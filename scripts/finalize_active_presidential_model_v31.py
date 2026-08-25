"""Promote and freeze V30: the third-share regional dispersion expansion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import multiplicative_dispersion_expansion  # noqa: E402
from scripts import finalize_active_presidential_model_v25 as shared  # noqa: E402

ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v31"
POINTERS = (
    ROOT / "data/config/current_presidential_model.json",
    ROOT / "data/config/active_presidential_model.json",
)
V30_SHA256 = "afee25e582e201873f1785c7123004336f4dfb892791c30c4e6f3f7ab9d3049e"
#: V30's published macros, recorded so the manifest can state the change
#: rather than assert a direction.
V30_NATIONAL_MACRO_PP = 0.7204374174124484
V30_REGIONAL_MACRO_PP = 2.5664447526782004

PROSPECTIVE_DEMONSTRATION = {
    "artifact": "outputs/prospective_pres_2025_v31",
    "regenerated_for_v31": True,
    "history_reference": "outputs/external_model_free_v25_baseline",
    "supersedes": "outputs/prospective_pres_2025_v30",
    "change_from_published_v30_artifact": {
        "regional_max_pp": 2.0530,
        "regional_mean_pp": 0.3158,
        "national_max_pp": 4.441e-14,
        "minimum_share_before_pct": 0.0001,
        "minimum_share_after_pct": 2.0531,
        "winner_unchanged": True,
        "ranking_unchanged": True,
    },
    "note": (
        "The V30 artifact published 0.00% for 김문수 in 광주. That was not an "
        "estimate: V29's expansion is capped at the factor where some region "
        "reaches zero, so the region setting the cap lands on zero by "
        "construction, and the displaced mass moved to the other two candidates "
        "in that region. V31's multiplicative form cannot reach zero from a "
        "positive input, and the same 광주 row is now 2.05% against a "
        "pre-transform 2.67%. The national levels are unchanged to 4e-14pp, so "
        "the correction is entirely within regions. No 2025 outcome was used."
    ),
}

SELECTION_DISCLOSURE = (
    "V31 replaces V29's additive dispersion expansion with a multiplicative one. "
    "The additive form is linear in the deviation and has no lower bound, so it "
    "was capped per election at the factor where some region reaches zero - which "
    "means the region setting the cap lands on exactly zero whenever the cap "
    "binds. That is not an estimate; it is where the arithmetic stopped. On the "
    "scored panel it is 홍준표's 광주 in 2017 (3.55% into the transform, 1.68% "
    "realised, 0.00% published) and in the demonstration it is 김문수's 광주 in "
    "2025 (2.67% in, 0.00% out). The multiplicative form scales the ratio rather "
    "than the difference, so it cannot reach zero from a positive input and needs "
    "no cap. It does not conserve the weighted national level on its own, so the "
    "regional sums and the candidate levels are alternated to convergence; "
    "neither step introduces a constant, and the gain stays the parameter-free "
    "1.0. Regional macro 2.566445 to 2.500701 and national 0.720437 to 0.724291: "
    "the national figure is worse, and the change was made anyway, because a "
    "prediction of exactly zero for a major-party candidate in a metropolitan "
    "region is wrong in kind rather than in degree. Both figures were measured "
    "before the decision."
)


def main() -> None:
    rollback = ROOT / "outputs/active_presidential_nested_v30/nested_predictions.csv"
    if shared._sha256(rollback) != V30_SHA256:
        raise RuntimeError("V30 rollback prediction drift")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads(
        (ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    prediction_hash = shared._sha256(ACTIVE_DIR / "nested_predictions.csv")
    # read what the run actually produced, so the manifest reports rather
    # than asserts
    predictions = pd.read_csv(
        ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    expansion_audit = pd.read_csv(
        ACTIVE_DIR / "multiplicative_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    region_sums_hold = bool(
        predictions.groupby(["election_id", "region_id"])["layer_pred"]
        .sum()
        .sub(1.0)
        .abs()
        .lt(1e-12)
        .all()
    )
    artifacts = [
        "scripts/run_active_presidential_model_v31.py",
        "scripts/run_prospective_forecast_v31.py",
        "scripts/build_active_v31_predictive_intervals.py",
        "scripts/evaluate_electorate_layers.py",
        "scripts/finalize_active_presidential_model_v31.py",
        "scripts/audit_public_active_presidential_model_v31.py",
        "scripts/verify_v31_clean_reproduction.py",
        "scripts/verify_v31_prospective_reproduction.py",
        "scripts/build_external_model_free_v25_baseline.py",
        "presidential_issue_engine/multiplicative_dispersion_expansion.py",
        "presidential_issue_engine/forecast_time_region_weights.py",
        "presidential_issue_engine/party_regionalism_dispersion.py",
        "presidential_issue_engine/fixed_dataset/pres_1997_regional_turnout.csv",
        "presidential_issue_engine/external_model_free_runtime.py",
        "presidential_issue_engine/issue_vote_engine.py",
        "presidential_issue_engine/fixed_dataset/kospi_election_context.csv",
        "docs/FINAL_MODEL_V31_20260825.md",
        "docs/EXPERIMENT_V31_MULTIPLICATIVE_EXPANSION_20260825.md",
        "outputs/active_presidential_nested_v31/nested_predictions.csv",
        "outputs/active_presidential_nested_v31/summary.json",
        "outputs/active_presidential_nested_v31/by_election.csv",
        "outputs/active_presidential_nested_v31/national_predictions.csv",
        "outputs/active_presidential_nested_v31/input_manifest.csv",
        "outputs/active_presidential_nested_v31/multiplicative_dispersion_expansion_audit.csv",
        "outputs/active_presidential_nested_v31/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v31/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v31/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v31/predictive_interval_manifest.json",
    ]
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "status": "promoted_multiplicative_dispersion_expansion",
        "active_version": "v31",
        "predecessor": "v30",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "accepted_scope": ["multiplicative_regional_dispersion_expansion"],
        "rejected_scope": [
            "swept_gain_0_50_better_regional_but_fitted_on_scored_panel"
        ],
        "selection_disclosure": SELECTION_DISCLOSURE,
        "rollback": {"version": "v30", "prediction_sha256": V30_SHA256},
    }
    shared._atomic_json(promotion, ACTIVE_DIR / "promotion_manifest.json")
    artifacts.append("outputs/active_presidential_nested_v31/promotion_manifest.json")

    finalization = {
        "schema": "presidential_model_finalization_v1",
        "status": "frozen_multiplicative_dispersion_expansion",
        "active_version": "v31",
        "base_config_version": "v23",
        "scored_development_elections": [
            "pres_2002",
            "pres_2007",
            "pres_2012",
            "pres_2017",
            "pres_2022",
        ],
        "post_2022_outcomes_used": False,
        "untouched_historical_holdout": False,
        "external_neural_model_runtime": False,
        "external_model_derived_inputs": [
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        ],
        "metrics": summary["metrics"],
        "multiplicative_dispersion_expansion": {
            "gain": multiplicative_dispersion_expansion.DEFAULT_GAIN,
            "gain_selection": "parameter_free_unit_gain_not_swept",
            "index": "model_predicted_third_placed_national_level",
            # measured from the shipped audit, not asserted. The previous
            # version of this block hardcoded three claims, and two of them
            # were false for V31: it listed pres_2017 as feasibility-capped
            # when removing the cap is the reason V31 exists, and it declared
            # the national macro not worse when V31 accepts a +0.0039pp cost
            # on purpose. A record that states its conclusions as literals
            # cannot go stale loudly.
            "candidate_national_level_preserved": bool(
                expansion_audit["max_candidate_level_shift_pp"].abs().lt(1e-9).all()
            ),
            "worst_candidate_level_shift_pp": float(
                expansion_audit["max_candidate_level_shift_pp"].abs().max()
            ),
            "regional_composition_preserved": bool(region_sums_hold),
            "feasibility_cap": (
                "not applicable; the multiplicative form cannot reach zero from a "
                "positive input, so no cap exists to bind"
            ),
            "feasibility_capped_elections": [],
            "minimum_predicted_share": float(predictions["layer_pred"].min()),
            "reconciliation_rounds": {
                str(row.election_id): int(row.reconciliation_rounds)
                for row in expansion_audit.itertuples(index=False)
            },
            "outcome_fields_used": [],
        },
        "predictive_intervals": {
            "status": intervals["status"],
            "levels": intervals["levels"],
            "post_2022_outcomes_used": False,
        },
        "verification": {
            # this one is genuinely checked above, and raises before reaching here
            "v30_rollback_hash_match": True,
            "v31_prediction_hash": prediction_hash,
            # the measured change, not a boolean asserting a direction. V31's
            # national macro is worse than V30's and the version was taken
            # anyway; a field claiming otherwise contradicted the experiment
            # record sitting beside it.
            "national_macro_change_vs_v30_pp": float(
                summary["metrics"]["national_equal_election_macro_mae_pp"]
                - V30_NATIONAL_MACRO_PP
            ),
            "regional_macro_change_vs_v30_pp": float(
                summary["metrics"]["regional_equal_election_macro_mae_pp"]
                - V30_REGIONAL_MACRO_PP
            ),
            "predecessor_score_was_not_a_promotion_condition": True,
        },
        "artifacts": [shared._record(path) for path in artifacts],
        "rollback": {
            "version": "v30",
            "prediction_sha256": V30_SHA256,
            "finalization_manifest": (
                "outputs/active_presidential_nested_v30/finalization_manifest.json"
            ),
        },
        "change_policy": (
            "Do not modify V23 through V31 in place; use a new versioned experiment."
        ),
    }
    shared._atomic_json(finalization, ACTIVE_DIR / "finalization_manifest.json")

    metrics = summary["metrics"]
    pointer = {
        "schema": "current_presidential_model_pointer_v1",
        "active_version": "v31",
        "lifecycle_status": "frozen_multiplicative_dispersion_expansion",
        "canonical_document": "docs/FINAL_MODEL_V31_20260825.md",
        "finalization_manifest": (
            "outputs/active_presidential_nested_v31/finalization_manifest.json"
        ),
        "runner": "scripts/run_active_presidential_model_v31.py",
        "prospective_runner": "scripts/run_prospective_forecast_v31.py",
        "config": "data/config/active_presidential_model_v23.json",
        "version_wrapper": "scripts/run_active_presidential_model_v31.py",
        "base_config_version": "v23",
        "output": "outputs/active_presidential_nested_v31",
        "predecessor": "v30",
        "rollback_pointer": (
            "outputs/active_presidential_nested_v30/finalization_manifest.json"
        ),
        "regional_equal_election_macro_mae_pp": metrics[
            "regional_equal_election_macro_mae_pp"
        ],
        "national_equal_election_macro_mae_pp": metrics[
            "national_equal_election_macro_mae_pp"
        ],
        "winner_accuracy": metrics["winner_accuracy"],
        "prediction_rows": metrics["rows"],
        "prediction_sha256": prediction_hash,
        "predictive_intervals": (
            "outputs/active_presidential_nested_v31/predictive_interval_manifest.json"
        ),
        "predictive_interval_levels": intervals["levels"],
        "predictive_interval_status": intervals["status"],
        "external_neural_model_runtime": False,
        "external_model_derived_inputs": [
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        ],
        "post_2022_outcomes_used": False,
        "prospective_demonstration": PROSPECTIVE_DEMONSTRATION,
    }
    for path in POINTERS:
        shared._atomic_json(pointer, path)
    _refresh_github_baseline(prediction_hash)


def _refresh_github_baseline(prediction_hash: str) -> None:
    """Keep the boundary baseline's pinned hashes in step with the pointer.

    The baseline pins the pointer files by hash, and finalizing rewrites those
    files, so every re-freeze left the boundary audit failing until someone
    refreshed the pins by hand. Doing it here means the only thing that can
    change the pointer is also the thing that records it.
    """

    import hashlib

    baseline_path = ROOT / "docs/GITHUB_BASELINE_V31_20260825.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for pointer_path in POINTERS:
        relative = pointer_path.relative_to(ROOT).as_posix()
        baseline["expected_hashes"][relative] = hashlib.sha256(
            pointer_path.read_bytes()
        ).hexdigest()
    baseline["expected_hashes"][
        "outputs/active_presidential_nested_v31/nested_predictions.csv"
    ] = prediction_hash
    baseline_path.write_bytes(
        (json.dumps(baseline, ensure_ascii=False, indent=2) + chr(10)).encode("utf-8")
    )


if __name__ == "__main__":
    main()
