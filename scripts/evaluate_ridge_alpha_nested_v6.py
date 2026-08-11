"""Diagnose Ridge shrinkage in the active strict-nested v6 pipeline.

Only ``ridge_alpha`` changes. Candidate assignment, point-in-time inputs,
predictors, structural layers, and every v6 postprocess remain fixed. Results
are diagnostic and must not automatically replace the active policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine import incumbent_shock_adjustment  # noqa: E402
from presidential_issue_engine import mega_issue_adjustment  # noqa: E402
from scripts import evaluate_preliminary_slot_shadow_nested as nested  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


ALPHA_MULTIPLIERS = (0.0, 0.01, 0.03, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0)
BASELINE_MULTIPLIER = 1.0
OUTPUT_DIR = ROOT / "outputs" / "ridge_alpha_nested_v6"
ACTIVE_PREDICTIONS = (
    ROOT / "outputs" / "active_presidential_nested" / "nested_predictions.csv"
)
ACTIVE_FOLD_AUDIT = (
    ROOT / "outputs" / "active_presidential_nested" / "fold_audit.csv"
)


def margin_diagnostics(
    national: pd.DataFrame, alpha_multiplier: float
) -> pd.DataFrame:
    """Measure every actual winner's predicted margin against its best rival."""

    rows: list[dict[str, object]] = []
    for election_id, group in national.groupby("election_id", sort=True):
        actual_winner = group.loc[group["actual_pct"].idxmax()]
        rivals = group.loc[
            ~group["candidate_key"].eq(actual_winner["candidate_key"])
        ]
        predicted_winner = group.loc[group["pred_pct"].idxmax()]
        predicted_margin = float(
            actual_winner["pred_pct"] - rivals["pred_pct"].max()
        )
        actual_margin = float(
            actual_winner["actual_pct"] - rivals["actual_pct"].max()
        )
        rows.append(
            {
                "alpha_multiplier": alpha_multiplier,
                "election_id": election_id,
                "actual_winner": actual_winner["candidate_name"],
                "predicted_winner": predicted_winner["candidate_name"],
                "winner_correct": bool(
                    predicted_winner["candidate_key"]
                    == actual_winner["candidate_key"]
                ),
                "predicted_actual_winner_margin_pp": predicted_margin,
                "actual_margin_pp": actual_margin,
                "margin_error_pp": predicted_margin - actual_margin,
                "abs_margin_error_pp": abs(predicted_margin - actual_margin),
            }
        )
    return pd.DataFrame(rows)


def _coefficient_norms(audit: pd.DataFrame) -> tuple[float, float]:
    columns = [
        f"standardized_coef_{name}" for name in nested.BASE_PREDICTORS
    ]
    values = audit[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    norms = np.sqrt(np.square(values.to_numpy(float)).sum(axis=1))
    return float(norms.mean()), float(norms.max())


def _baseline_reproduction_difference(predictions: pd.DataFrame) -> float:
    active_frame = pd.read_csv(ACTIVE_PREDICTIONS, encoding="utf-8-sig")
    keys = ["election_id", "region_id", "source_slot"]
    comparison = predictions[keys + ["layer_pred"]].merge(
        active_frame[keys + ["layer_pred"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_experiment", "_active"),
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        raise RuntimeError("alpha baseline and active prediction keys differ")
    return float(
        (
            comparison["layer_pred_experiment"]
            - comparison["layer_pred_active"]
        )
        .abs()
        .max()
    )


def run() -> dict[str, object]:
    policy = active.load_policy()
    active.regenerate_issue_seeds()
    active.regenerate_assignments()

    full = nested._prepare_rows()
    base = nested._base_layer_frame(require_frozen_reproduction=False)
    structural = policy["structural_layers"]
    electorate = structural["electorate_response"]
    postprocess = policy["postprocess"]
    intensity = pd.read_csv(active.MEGA_ISSUE_INTENSITY, encoding="utf-8-sig")
    profile = pd.read_csv(active.CANDIDATE_ISSUE_PROFILE, encoding="utf-8-sig")
    direct_mega_scores = mega_issue_adjustment.compile_direct_mega_scores(
        profile,
        intensity,
        nested.engine.ELECTION_DATES,
        minimum_intensity=float(postprocess["direct_mega_minimum_intensity"]),
        score_cap=float(postprocess["direct_mega_score_cap"]),
    )
    burden_scores = incumbent_shock_adjustment.compile_government_burden_scores(
        profile, nested.engine.ELECTION_DATES
    )
    active_audit = pd.read_csv(ACTIVE_FOLD_AUDIT, encoding="utf-8-sig")
    active_alpha_by_target = {
        str(row.target_election): float(row.layer_config_ridge_alpha)
        for row in active_audit.itertuples(index=False)
    }
    if set(active_alpha_by_target) != set(nested.ELECTIONS):
        raise RuntimeError("active fold alpha schedule does not match scored elections")

    summaries: list[dict[str, object]] = []
    predictions_all: list[pd.DataFrame] = []
    by_election_all: list[pd.DataFrame] = []
    national_all: list[pd.DataFrame] = []
    margins_all: list[pd.DataFrame] = []
    audits_all: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    baseline_difference: float | None = None

    for multiplier in ALPHA_MULTIPLIERS:
        try:
            overrides = dict(structural["outer_config_overrides"])
            alpha_by_target = {
                target: active_alpha_by_target[target] * multiplier
                for target in nested.ELECTIONS
            }
            outer, outer_audit = nested._build_outer_predictions(
                full,
                active.EXPECTED_VARIANT,
                layer_config_overrides=overrides,
                layer_config_overrides_by_target={
                    target: {"ridge_alpha": alpha}
                    for target, alpha in alpha_by_target.items()
                },
            )
            layered = nested._attach_layers(base, outer)
            terrain_gains, _ = active.structural_terrain_gain_by_target(
                layered, intensity, electorate["terrain_anchor"]
            )
            predictions, _ = nested._apply_nested_preference(
                layered,
                active.EXPECTED_VARIANT,
                preference_gain_floor=float(electorate["preference_gain_floor"]),
                terrain_gain_by_target=terrain_gains,
            )
            predictions = mega_issue_adjustment.apply_direct_mega_shift(
                predictions,
                direct_mega_scores,
                prediction_column="layer_pred",
                gain=float(postprocess["direct_mega_logit_gain"]),
                log_shift_cap=float(postprocess["direct_mega_log_shift_cap"]),
            )
            predictions = incumbent_shock_adjustment.apply_incumbent_shock_response(
                predictions,
                burden_scores,
                intensity,
                nested.engine.ELECTION_DATES,
                prediction_column="layer_pred",
                government_burden_gain=float(postprocess["government_burden_gain"]),
                rupture_extra_gain=float(postprocess["rupture_extra_gain"]),
                conversion_buffer=float(postprocess["incumbent_conversion_buffer"]),
                log_shift_cap=float(postprocess["incumbent_shock_log_shift_cap"]),
            )
            regimes = contest_regime.derive_contest_regimes(
                predictions, prediction_column="layer_pred"
            )
            predictions = contest_regime.apply_contest_regime_response(
                predictions,
                regimes,
                prediction_column="layer_pred",
                expansion_gain=float(postprocess["contest_regime_expansion_gain"]),
                log_shift_cap=float(postprocess["contest_regime_log_shift_cap"]),
            )
            summary, by_election, national = nested._metrics(
                predictions,
                "layer_pred",
                f"ridge_alpha_multiplier_{multiplier:g}",
            )
            margins = margin_diagnostics(national, multiplier)
            audit = pd.DataFrame(outer_audit)
            coefficient_mean, coefficient_max = _coefficient_norms(audit)
            target_margins = margins.loc[
                margins["election_id"].isin(["pres_2007", "pres_2017"])
            ]
            by_index = by_election.set_index("election_id")
            summary.update(
                {
                    "alpha_multiplier": multiplier,
                    "mean_ridge_alpha": float(np.mean(list(alpha_by_target.values()))),
                    "min_ridge_alpha": float(min(alpha_by_target.values())),
                    "max_ridge_alpha": float(max(alpha_by_target.values())),
                    "ridge_alpha_schedule": "|".join(
                        f"{target}:{alpha_by_target[target]:g}"
                        for target in nested.ELECTIONS
                    ),
                    "mean_standardized_coefficient_l2": coefficient_mean,
                    "max_standardized_coefficient_l2": coefficient_max,
                    "all_election_margin_bias_pp": float(
                        margins["margin_error_pp"].mean()
                    ),
                    "all_election_margin_mae_pp": float(
                        margins["abs_margin_error_pp"].mean()
                    ),
                    "target_2007_2017_margin_bias_pp": float(
                        target_margins["margin_error_pp"].mean()
                    ),
                    "pres_2007_margin_error_pp": float(
                        margins.loc[
                            margins["election_id"].eq("pres_2007"),
                            "margin_error_pp",
                        ].iloc[0]
                    ),
                    "pres_2017_margin_error_pp": float(
                        margins.loc[
                            margins["election_id"].eq("pres_2017"),
                            "margin_error_pp",
                        ].iloc[0]
                    ),
                    "pres_2007_national_mae_pp": float(
                        by_index.loc["pres_2007", "national_candidate_mae_pp"]
                    ),
                    "pres_2017_national_mae_pp": float(
                        by_index.loc["pres_2017", "national_candidate_mae_pp"]
                    ),
                }
            )
            if np.isclose(multiplier, BASELINE_MULTIPLIER):
                baseline_difference = _baseline_reproduction_difference(predictions)
                if baseline_difference > 1e-12:
                    raise RuntimeError(
                        "alpha multiplier=1.0 does not reproduce active v6: "
                        f"max difference={baseline_difference}"
                    )
            predictions = predictions.copy()
            predictions.insert(0, "alpha_multiplier", multiplier)
            by_election.insert(0, "alpha_multiplier", multiplier)
            national.insert(0, "alpha_multiplier", multiplier)
            audit.insert(0, "alpha_multiplier", multiplier)
            summaries.append(summary)
            predictions_all.append(
                predictions[
                    [
                        "alpha_multiplier",
                        "election_id",
                        "region_id",
                        "source_slot",
                        "candidate_name_x",
                        "contest_votes",
                        "actual",
                        "layer_pred",
                    ]
                ]
            )
            by_election_all.append(by_election)
            national_all.append(national)
            margins_all.append(margins)
            audits_all.append(audit)
        except np.linalg.LinAlgError as exc:
            failures.append(
                {
                    "alpha_multiplier": multiplier,
                    "status": "failed_singular_design",
                    "error": str(exc),
                }
            )

    if baseline_difference is None:
        raise RuntimeError("alpha multiplier=1.0 baseline was not evaluated")
    summary_frame = pd.DataFrame(summaries).sort_values("alpha_multiplier")
    payload = {
        "status": "diagnostic_only_no_promotion",
        "policy_version": policy["policy_version"],
        "changed_parameter": "ridge_alpha_schedule_multiplier_only",
        "active_alpha_by_target": active_alpha_by_target,
        "alpha_multiplier_grid": list(ALPHA_MULTIPLIERS),
        "baseline_multiplier": BASELINE_MULTIPLIER,
        "baseline_reproduction_max_difference": baseline_difference,
        "post_2022_outcomes_used": False,
        "target_excluded_from_each_outer_fit": True,
        "successful_runs": int(len(summary_frame)),
        "failures": failures,
        "selection_warning": (
            "This grid reads 2002-2022 development outcomes and cannot select a "
            "new production alpha without an additional nested selection rule."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(by_election_all, ignore_index=True).to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(national_all, ignore_index=True).to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(margins_all, ignore_index=True).to_csv(
        OUTPUT_DIR / "margin_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(predictions_all, ignore_index=True).to_csv(
        OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(audits_all, ignore_index=True).to_csv(
        OUTPUT_DIR / "fold_audit.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
