"""Strict nested evaluation of a bounded regional camp-core floor."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import electorate_layers as layers  # noqa: E402
import evaluate_electorate_layers as base_eval  # noqa: E402
import evaluate_nested_electorate_learning as gain_eval  # noqa: E402


ELECTIONS = base_eval.ALLOWED_ELECTIONS
CAMP_GAIN_GRID = (0.0, 0.025, 0.05, 0.075, 0.10)
MIN_TUNING_ELECTIONS = 2
MAX_PRIOR_WORSENING_PP = 0.05
OUTPUT_DIR = ROOT / "outputs" / "camp_core_anchor_experiment"


def _by_election(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    return base_eval.election_weighted_mae(frame, prediction_column).set_index(
        "election_id"
    )["weighted_row_mae_pp"]


def select_camp_gain(
    frame: pd.DataFrame,
    tuning_elections: tuple[str, ...],
    preference_gain: float,
    *,
    selection_label: str,
    cache: dict[tuple[float, float], pd.Series],
) -> tuple[float, list[dict[str, object]]]:
    if len(tuning_elections) < MIN_TUNING_ELECTIONS:
        return 0.0, []
    required_improved = math.ceil(len(tuning_elections) / 2.0)
    traces: list[dict[str, object]] = []
    candidates: list[tuple[float, float]] = []
    baseline: pd.Series | None = None
    for camp_gain in CAMP_GAIN_GRID:
        key = (float(preference_gain), float(camp_gain))
        if key not in cache:
            evaluated = base_eval.apply_config(
                frame,
                layers.ElectorateLayerConfig(
                    preference_gain=preference_gain,
                    camp_core_anchor_gain=camp_gain,
                    mass_profile="direct_party_layers",
                ),
            )
            cache[key] = _by_election(evaluated, "layer_pred")
        if camp_gain == 0.0:
            baseline = cache[key].loc[list(tuning_elections)]
        assert baseline is not None
        candidate = cache[key].loc[list(tuning_elections)]
        improvement = baseline - candidate
        improved_count = int((improvement > 1e-12).sum())
        max_worsening = float((-improvement).clip(lower=0.0).max())
        eligible = bool(
            camp_gain == 0.0
            or (
                improved_count >= required_improved
                and max_worsening <= MAX_PRIOR_WORSENING_PP
            )
        )
        macro = float(candidate.mean())
        traces.append(
            {
                "selection_label": selection_label,
                "tuning_elections": "|".join(tuning_elections),
                "preference_gain": float(preference_gain),
                "camp_core_anchor_gain": float(camp_gain),
                "weighted_macro_mae_pp": macro,
                "incremental_improvement_pp": float(baseline.mean() - macro),
                "improved_elections": improved_count,
                "maximum_prior_worsening_pp": max_worsening,
                "eligible": eligible,
            }
        )
        if eligible:
            candidates.append((macro, float(camp_gain)))
    _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected, traces


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = base_eval.prepare_frame(mass_profile="direct_party_layers")
    gain_cache: dict[float, pd.Series] = {}
    camp_cache: dict[tuple[float, float], pd.Series] = {}
    baseline_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    configs: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []

    for index, target in enumerate(ELECTIONS):
        prior = ELECTIONS[:index]
        preference_gain, _ = gain_eval.select_preference_gain(
            frame,
            prior,
            selection_label=f"preference_outer_{target}",
            gain_grid=gain_eval.CAPPED_GAIN_GRID,
            metric_cache=gain_cache,
        )
        camp_gain, trace = select_camp_gain(
            frame,
            prior,
            preference_gain,
            selection_label=f"camp_outer_{target}",
            cache=camp_cache,
        )
        traces.extend(trace)
        target_frame = frame.loc[frame["election_id"].eq(target)].copy()
        baseline_parts.append(
            base_eval.apply_config(
                target_frame,
                layers.ElectorateLayerConfig(
                    preference_gain=preference_gain,
                    mass_profile="direct_party_layers",
                ),
            )
        )
        candidate_parts.append(
            base_eval.apply_config(
                target_frame,
                layers.ElectorateLayerConfig(
                    preference_gain=preference_gain,
                    camp_core_anchor_gain=camp_gain,
                    mass_profile="direct_party_layers",
                ),
            )
        )
        configs.append(
            {
                "target_election": target,
                "tuning_elections": "|".join(prior),
                "preference_gain": float(preference_gain),
                "camp_core_anchor_gain": float(camp_gain),
                "target_excluded_from_tuning": target not in prior,
            }
        )

    baseline_nested = pd.concat(baseline_parts, ignore_index=True)
    candidate_nested = pd.concat(candidate_parts, ignore_index=True)
    baseline_by = _by_election(baseline_nested, "layer_pred")
    candidate_by = _by_election(candidate_nested, "layer_pred")
    comparison = pd.DataFrame(
        {
            "election_id": list(ELECTIONS),
            "active_nested_mae_pp": baseline_by.loc[list(ELECTIONS)].to_numpy(),
            "camp_nested_mae_pp": candidate_by.loc[list(ELECTIONS)].to_numpy(),
        }
    )
    comparison["improvement_pp"] = (
        comparison["active_nested_mae_pp"] - comparison["camp_nested_mae_pp"]
    )
    baseline_macro = float(comparison["active_nested_mae_pp"].mean())
    candidate_macro = float(comparison["camp_nested_mae_pp"].mean())
    improvement = baseline_macro - candidate_macro
    max_worsening = float((-comparison["improvement_pp"]).clip(lower=0.0).max())

    final_preference_gain, _ = gain_eval.select_preference_gain(
        frame,
        ELECTIONS,
        selection_label="preference_future_through2022",
        gain_grid=gain_eval.CAPPED_GAIN_GRID,
        metric_cache=gain_cache,
    )
    final_camp_gain, final_trace = select_camp_gain(
        frame,
        ELECTIONS,
        final_preference_gain,
        selection_label="camp_future_through2022",
        cache=camp_cache,
    )
    traces.extend(final_trace)
    nonzero_targets = [row for row in configs if float(row["camp_core_anchor_gain"]) > 0.0]
    gates = {
        "strict_nested_noninferior": improvement >= -1e-12,
        "strict_nested_improvement_at_least_0_002pp": improvement >= 0.002,
        "no_outer_election_worsens_more_than_0_05pp": max_worsening <= 0.05,
        "every_target_excluded_from_tuning": all(
            bool(row["target_excluded_from_tuning"]) for row in configs
        ),
        "at_least_two_outer_folds_select_nonzero_camp_gain": len(nonzero_targets) >= 2,
        "future_gain_within_declared_0_10_cap": 0.0 <= final_camp_gain <= 0.10,
    }
    payload = {
        "scope": {
            "scored_elections": list(ELECTIONS),
            "post_2022_presidential_outcomes_loaded": False,
            "selection_order": "preference_gain_then_camp_core_anchor_gain",
            "metric": "contest-vote weighted row MAE within election, equal-election macro",
        },
        "mechanism": {
            "unit": "region by political camp before candidate allocation",
            "candidate_claim": "official camp plus bounded multi-camp claim from PIT landscape",
            "floor_projection": "only baseline shares below allocated camp core are raised",
            "camp_gain_grid": list(CAMP_GAIN_GRID),
        },
        "active_nested_macro_mae_pp": baseline_macro,
        "camp_nested_macro_mae_pp": candidate_macro,
        "strict_nested_improvement_pp": improvement,
        "outer_configs": configs,
        "future_deployment_candidate": asdict(
            layers.ElectorateLayerConfig(
                preference_gain=final_preference_gain,
                camp_core_anchor_gain=final_camp_gain,
                mass_profile="direct_party_layers",
            )
        ),
        "adoption_gates": gates,
        "adopt_into_active_engine": bool(all(gates.values())),
        "caveat": (
            "The 0.10 safety cap was declared after inspecting fixed historical diagnostics; "
            "outer target outcomes remain excluded, but this is not an untouched design choice."
        ),
    }
    comparison.to_csv(OUTPUT_DIR / "nested_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(configs).to_csv(
        OUTPUT_DIR / "outer_selected_configs.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(traces).to_csv(
        OUTPUT_DIR / "selection_trace.csv", index=False, encoding="utf-8-sig"
    )
    candidate_nested.to_csv(
        OUTPUT_DIR / "nested_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
