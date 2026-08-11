"""Factorial ablation for the v10 regional-accent and regime-transition layers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_active_presidential_model as active  # noqa: E402
from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine import incumbent_shock_adjustment  # noqa: E402
from presidential_issue_engine import mega_issue_adjustment  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "regional_accent_regime_v10_ablation"


def _late_stages(
    structural: pd.DataFrame,
    policy: dict[str, object],
    *,
    modern_regime: bool,
) -> pd.DataFrame:
    postprocess = policy["postprocess"]
    profile = pd.read_csv(active.CANDIDATE_ISSUE_PROFILE, encoding="utf-8-sig")
    intensity = pd.read_csv(active.MEGA_ISSUE_INTENSITY, encoding="utf-8-sig")
    direct_scores = mega_issue_adjustment.compile_direct_mega_scores(
        profile,
        intensity,
        active.nested.engine.ELECTION_DATES,
        minimum_intensity=float(postprocess["direct_mega_minimum_intensity"]),
        score_cap=float(postprocess["direct_mega_score_cap"]),
    )
    mega = mega_issue_adjustment.apply_direct_mega_shift(
        structural,
        direct_scores,
        prediction_column="layer_pred",
        gain=float(postprocess["direct_mega_logit_gain"]),
        log_shift_cap=float(postprocess["direct_mega_log_shift_cap"]),
    )
    burden = incumbent_shock_adjustment.compile_government_burden_scores(
        profile, active.nested.engine.ELECTION_DATES
    )
    shock = incumbent_shock_adjustment.apply_incumbent_shock_response(
        mega,
        burden,
        intensity,
        active.nested.engine.ELECTION_DATES,
        prediction_column="layer_pred",
        government_burden_gain=float(postprocess["government_burden_gain"]),
        rupture_extra_gain=float(postprocess["rupture_extra_gain"]),
        conversion_buffer=float(postprocess["incumbent_conversion_buffer"]),
        log_shift_cap=float(postprocess["incumbent_shock_log_shift_cap"]),
    )
    regimes = contest_regime.derive_contest_regimes(
        shock,
        prediction_column="layer_pred",
        rejection_double_discount=not modern_regime,
    )
    if modern_regime:
        critical_elasticity = float(
            postprocess["contest_regime_critical_elasticity"]
        )
        swing_elasticity = float(postprocess["contest_regime_swing_elasticity"])
        swing_cap = float(postprocess["contest_regime_swing_log_shift_cap"])
    else:
        critical_elasticity = 1.0
        swing_elasticity = 1.0
        swing_cap = float(postprocess["contest_regime_log_shift_cap"])
    return contest_regime.apply_contest_regime_response(
        shock,
        regimes,
        prediction_column="layer_pred",
        expansion_gain=float(postprocess["contest_regime_expansion_gain"]),
        log_shift_cap=float(postprocess["contest_regime_log_shift_cap"]),
        critical_elasticity=critical_elasticity,
        swing_elasticity=swing_elasticity,
        swing_log_shift_cap=swing_cap,
    )


def run() -> dict[str, object]:
    policy = active.load_policy()
    with active.strict_input_policy():
        # Assignment priors are part of the evaluated pipeline. Regenerate them
        # under the same undated-input policy as the active run, otherwise the
        # third-candidate hierarchy can differ even when the Ridge fit is equal.
        active.regenerate_issue_seeds()
        active.regenerate_assignments()
        full = active.nested._prepare_rows()
        base = active.nested._base_layer_frame(require_frozen_reproduction=False)
        structural_policy = policy["structural_layers"]
        outer, _ = active.nested._build_outer_predictions(
            full,
            active.EXPECTED_VARIANT,
            layer_config_overrides=structural_policy["outer_config_overrides"],
        )
        layered = active.nested._attach_layers(base, outer)
        electorate = structural_policy["electorate_response"]
        intensity = pd.read_csv(active.MEGA_ISSUE_INTENSITY, encoding="utf-8-sig")
        terrain_gains, _ = active.structural_terrain_gain_by_target(
            layered, intensity, electorate["terrain_anchor"]
        )
        accent_gains, _ = active.regional_accent_gain_by_target(
            layered, electorate["regional_accent"]
        )

        structural_variants: dict[bool, pd.DataFrame] = {}
        for accent_enabled in (False, True):
            gains = accent_gains if accent_enabled else {
                election: 0.0 for election in active.nested.ELECTIONS
            }
            structural_variants[accent_enabled], _ = (
                active.nested._apply_nested_preference(
                    layered,
                    active.EXPECTED_VARIANT,
                    preference_gain_floor=float(electorate["preference_gain_floor"]),
                    terrain_gain_by_target=terrain_gains,
                    regional_accent_gain_by_target=gains,
                    regional_accent_signal_width=float(
                        electorate["regional_accent"]["signal_width"]
                    ),
                )
            )

        summaries: list[dict[str, object]] = []
        election_rows: list[pd.DataFrame] = []
        national_rows: list[pd.DataFrame] = []
        prediction_rows: list[pd.DataFrame] = []
        for accent_enabled in (False, True):
            for modern_regime in (False, True):
                label = (
                    f"accent_{int(accent_enabled)}_regime_{int(modern_regime)}"
                )
                predictions = _late_stages(
                    structural_variants[accent_enabled],
                    policy,
                    modern_regime=modern_regime,
                )
                summary, by_election, national = active.nested._metrics(
                    predictions, "layer_pred", label
                )
                summary["regional_accent_enabled"] = accent_enabled
                summary["modern_regime_transition"] = modern_regime
                summaries.append(summary)
                election_rows.append(by_election)
                national_rows.append(national)
                prediction_rows.append(
                    predictions.assign(ablation_variant=label)
                )

        active_predictions_path = active.OUTPUT_DIR / "nested_predictions.csv"
        if active_predictions_path.exists():
            active_predictions = pd.read_csv(
                active_predictions_path, encoding="utf-8-sig"
            )
            full_ablation = prediction_rows[-1]
            keys = ["election_id", "region_id", "source_slot"]
            matched = active_predictions[keys + ["layer_pred"]].merge(
                full_ablation[keys + ["layer_pred"]],
                on=keys,
                how="inner",
                suffixes=("_active", "_ablation"),
                validate="one_to_one",
            )
            if len(matched) != len(active_predictions):
                raise RuntimeError("active/full-ablation row coverage differs")
            maximum_difference = float(
                (
                    matched["layer_pred_active"]
                    - matched["layer_pred_ablation"]
                )
                .abs()
                .max()
            )
            if maximum_difference > 1e-12:
                raise RuntimeError(
                    "active/full-ablation predictions differ: "
                    f"maximum absolute difference={maximum_difference:.3e}"
                )
        else:
            maximum_difference = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(election_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(national_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "scope": "strict nested through-2022 development folds",
        "post_2022_outcomes_used": False,
        "active_full_ablation_maximum_difference": maximum_difference,
        "variants": summary_frame.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
