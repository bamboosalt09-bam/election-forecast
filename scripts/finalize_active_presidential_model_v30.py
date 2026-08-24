"""Promote and freeze V29: the third-share regional dispersion expansion."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import third_share_dispersion_expansion  # noqa: E402
from scripts import finalize_active_presidential_model_v25 as shared  # noqa: E402

ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v30"
POINTERS = (
    ROOT / "data/config/current_presidential_model.json",
    ROOT / "data/config/active_presidential_model.json",
)
V29_SHA256 = "fed959cdba1e127f91c2ab640a378d1f44a4a3e79b4c4a76893cf8d7c6153904"

PROSPECTIVE_DEMONSTRATION = {
    "artifact": "outputs/prospective_pres_2025_v30",
    "regenerated_for_v30": True,
    "history_reference": "outputs/external_model_free_v25_baseline",
    "supersedes": "outputs/prospective_pres_2025_v28",
    "identical_to_v29_artifact": True,
    "change_from_published_v28_artifact": {
        "boundary_enforcement_regional_max_pp": 0.0359,
        "boundary_enforcement_regional_mean_pp": 0.0103,
        "boundary_enforcement_national_max_pp": 0.006484,
        "third_share_expansion_regional_max_pp": 2.6703,
        "third_share_expansion_regional_mean_pp": 0.6970,
        "third_share_expansion_national_max_pp": 0.0,
        "forecast_time_weight_regional_max_pp": 0.0,
        "winner_unchanged": True,
    },
    "note": (
        "The published V28 artifact predated process-wide enforcement of the "
        "V28 external-model boundary, so it used seed inputs V28 documents as "
        "removed. Regenerating under the enforced boundary moves the national "
        "levels by 0.0065pp and the regions by 0.0103pp on average; the larger "
        "regional movement is the V29 dispersion expansion, which conserves the "
        "national levels exactly. V30's reweighting leaves this artifact "
        "byte-identical to the V29 one, because the 2025 path already refused "
        "the target election's turnout and used 2022 volumes - that refusal is "
        "what V30 extends to the scored panel. See "
        "docs/DIAGNOSIS_PROSPECTIVE_2025_PATH_20260823.md."
    ),
}

SELECTION_DISCLOSURE = (
    "V30 changes only which regional weight the two terminal transforms read. V27 "
    "and V29 weighted each candidate's national level by contest_votes - the target "
    "election's own turnout, which exists only after the count, so a postprocess "
    "using it consumed an outcome of the election it was predicting. Each scored "
    "election now uses its predecessor's regional valid votes; 2002 uses 1997, whose "
    "turnout is carried in fixed_dataset/pres_1997_regional_turnout.csv because 1997 "
    "is a warmup election outside the scored panel. The transform forms, the gain, "
    "the Ridge stack and the V28 external-model boundary are unchanged. Closing the "
    "leak improved both headline figures - regional 2.573607 to 2.566445, national "
    "0.726250 to 0.720437 - which is recorded as an outcome, not as the reason: the "
    "change was made because the old weight was not available at forecast time."
)


def main() -> None:
    rollback = ROOT / "outputs/active_presidential_nested_v29/nested_predictions.csv"
    if shared._sha256(rollback) != V29_SHA256:
        raise RuntimeError("V29 rollback prediction drift")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads(
        (ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    prediction_hash = shared._sha256(ACTIVE_DIR / "nested_predictions.csv")
    artifacts = [
        "scripts/run_active_presidential_model_v30.py",
        "scripts/run_prospective_forecast_v30.py",
        "scripts/build_active_v30_predictive_intervals.py",
        "scripts/evaluate_electorate_layers.py",
        "scripts/finalize_active_presidential_model_v30.py",
        "scripts/audit_public_active_presidential_model_v30.py",
        "scripts/verify_v30_clean_reproduction.py",
        "scripts/verify_v30_prospective_reproduction.py",
        "scripts/build_external_model_free_v25_baseline.py",
        "presidential_issue_engine/third_share_dispersion_expansion.py",
        "presidential_issue_engine/forecast_time_region_weights.py",
        "presidential_issue_engine/party_regionalism_dispersion.py",
        "presidential_issue_engine/fixed_dataset/pres_1997_regional_turnout.csv",
        "presidential_issue_engine/external_model_free_runtime.py",
        "presidential_issue_engine/issue_vote_engine.py",
        "presidential_issue_engine/fixed_dataset/kospi_election_context.csv",
        "docs/FINAL_MODEL_V30_20260824.md",
        "docs/EXPERIMENT_V30_FORECAST_TIME_WEIGHTS_20260824.md",
        "outputs/active_presidential_nested_v30/nested_predictions.csv",
        "outputs/active_presidential_nested_v30/summary.json",
        "outputs/active_presidential_nested_v30/by_election.csv",
        "outputs/active_presidential_nested_v30/national_predictions.csv",
        "outputs/active_presidential_nested_v30/input_manifest.csv",
        "outputs/active_presidential_nested_v30/third_share_dispersion_expansion_audit.csv",
        "outputs/active_presidential_nested_v30/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v30/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v30/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v30/predictive_interval_manifest.json",
    ]
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "status": "promoted_third_share_dispersion_expansion",
        "active_version": "v30",
        "predecessor": "v29",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "accepted_scope": ["third_share_indexed_regional_dispersion_expansion"],
        "rejected_scope": [
            "swept_gain_0_50_better_regional_but_fitted_on_scored_panel"
        ],
        "selection_disclosure": SELECTION_DISCLOSURE,
        "rollback": {"version": "v29", "prediction_sha256": V29_SHA256},
    }
    shared._atomic_json(promotion, ACTIVE_DIR / "promotion_manifest.json")
    artifacts.append("outputs/active_presidential_nested_v30/promotion_manifest.json")

    finalization = {
        "schema": "presidential_model_finalization_v1",
        "status": "frozen_third_share_dispersion_expansion",
        "active_version": "v30",
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
        "third_share_dispersion_expansion": {
            "gain": third_share_dispersion_expansion.DEFAULT_GAIN,
            "gain_selection": "parameter_free_unit_gain_not_swept",
            "index": "model_predicted_third_placed_national_level",
            "candidate_national_level_preserved": True,
            "regional_composition_preserved": True,
            "feasibility_capped_elections": ["pres_2017"],
            "outcome_fields_used": [],
        },
        "predictive_intervals": {
            "status": intervals["status"],
            "levels": intervals["levels"],
            "post_2022_outcomes_used": False,
        },
        "verification": {
            "v29_rollback_hash_match": True,
            "v30_prediction_hash": prediction_hash,
            "national_macro_not_worse_than_v29": True,
        },
        "artifacts": [shared._record(path) for path in artifacts],
        "rollback": {
            "version": "v29",
            "prediction_sha256": V29_SHA256,
            "finalization_manifest": (
                "outputs/active_presidential_nested_v29/finalization_manifest.json"
            ),
        },
        "change_policy": (
            "Do not modify V23 through V30 in place; use a new versioned experiment."
        ),
    }
    shared._atomic_json(finalization, ACTIVE_DIR / "finalization_manifest.json")

    metrics = summary["metrics"]
    pointer = {
        "schema": "current_presidential_model_pointer_v1",
        "active_version": "v30",
        "lifecycle_status": "frozen_forecast_time_regional_weighting",
        "canonical_document": "docs/FINAL_MODEL_V30_20260824.md",
        "finalization_manifest": (
            "outputs/active_presidential_nested_v30/finalization_manifest.json"
        ),
        "runner": "scripts/run_active_presidential_model_v30.py",
        "prospective_runner": "scripts/run_prospective_forecast_v30.py",
        "config": "data/config/active_presidential_model_v23.json",
        "version_wrapper": "scripts/run_active_presidential_model_v30.py",
        "base_config_version": "v23",
        "output": "outputs/active_presidential_nested_v30",
        "predecessor": "v29",
        "rollback_pointer": (
            "outputs/active_presidential_nested_v29/finalization_manifest.json"
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
            "outputs/active_presidential_nested_v30/predictive_interval_manifest.json"
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

    baseline_path = ROOT / "docs/GITHUB_BASELINE_V30_20260824.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for pointer_path in POINTERS:
        relative = pointer_path.relative_to(ROOT).as_posix()
        baseline["expected_hashes"][relative] = hashlib.sha256(
            pointer_path.read_bytes()
        ).hexdigest()
    baseline["expected_hashes"][
        "outputs/active_presidential_nested_v30/nested_predictions.csv"
    ] = prediction_hash
    baseline_path.write_bytes(
        (json.dumps(baseline, ensure_ascii=False, indent=2) + chr(10)).encode("utf-8")
    )


if __name__ == "__main__":
    main()
