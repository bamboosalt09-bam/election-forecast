"""Write V27 pointer, promotion, finalization, and repository baseline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_active_presidential_model_v25 as shared  # noqa: E402

ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v27"
POINTER = ROOT / "data/config/current_presidential_model.json"
BASELINE = ROOT / "docs/GITHUB_BASELINE_V27_20260822.json"
V23_SHA256 = shared.V23_SHA256
V24_SHA256 = shared.V24_SHA256
V25_SHA256 = shared.V25_SHA256
V26_SHA256 = "9b66b813f97c3c2804a178ebb5b9104fa4a58553c75812f75affbb3b17773dd3"


def main() -> None:
    for version, expected in (("v23", V23_SHA256), ("v24", V24_SHA256), ("v25", V25_SHA256), ("v26", V26_SHA256)):
        path = ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv"
        if shared._sha256(path) != expected:
            raise RuntimeError(f"{version} rollback prediction drift")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads((ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8"))
    prediction_hash = shared._sha256(ACTIVE_DIR / "nested_predictions.csv")
    artifacts = [
        "scripts/run_active_presidential_model_v27.py",
        "scripts/run_prospective_forecast_v27.py",
        "scripts/build_active_v27_predictive_intervals.py",
        "scripts/audit_public_active_presidential_model_v27.py",
        "presidential_issue_engine/party_regionalism_dispersion.py",
        "docs/FINAL_MODEL_V27_20260822.md",
        "docs/EXPERIMENT_CORE_WEIGHTED_REGIONAL_POLARIZATION_20260822.md",
        "outputs/active_presidential_nested_v27/nested_predictions.csv",
        "outputs/active_presidential_nested_v27/summary.json",
        "outputs/active_presidential_nested_v27/by_election.csv",
        "outputs/active_presidential_nested_v27/national_predictions.csv",
        "outputs/active_presidential_nested_v27/party_regionalism_dispersion_audit.csv",
        "outputs/active_presidential_nested_v27/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v27/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v27/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v27/predictive_interval_manifest.json",
    ]
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "status": "promoted_frozen_pre_2025_evaluation",
        "active_version": "v27",
        "predecessor": "v26",
        "review_decision": "explicit_user_authorized_pointer_promotion",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "accepted_scope": ["core_weighted_inherited_party_regional_dispersion_gain_1"],
        "rejected_scope": ["regional_share_floor", "panel_optimal_dispersion_gain_3"],
        "selection_disclosure": "The defect and mechanism were developed on the scored panel. Gain 1 is theory-fixed; the panel-optimal gain 3 was rejected.",
        "rollback": {"version": "v26", "prediction_sha256": V26_SHA256},
    }
    shared._atomic_json(promotion, ACTIVE_DIR / "promotion_manifest.json")
    artifacts.append("outputs/active_presidential_nested_v27/promotion_manifest.json")
    finalization = {
        "schema": "presidential_model_finalization_v1",
        "status": "frozen_pre_2025_evaluation",
        "active_version": "v27",
        "base_config_version": "v23",
        "scored_development_elections": ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"],
        "post_2022_outcomes_used": False,
        "untouched_historical_holdout": False,
        "metrics": summary["metrics"],
        "predictive_intervals": {"status": intervals["status"], "levels": intervals["levels"], "post_2022_outcomes_used": False},
        "verification": {
            "v23_rollback_hash_match": True, "v24_rollback_hash_match": True,
            "v25_rollback_hash_match": True, "v26_rollback_hash_match": True,
            "v27_prediction_hash": prediction_hash,
        },
        "artifacts": [shared._record(path) for path in artifacts],
        "rollback": {"version": "v26", "prediction_sha256": V26_SHA256, "finalization_manifest": "outputs/active_presidential_nested_v26/finalization_manifest.json"},
        "change_policy": "Do not modify V23 through V27 in place; use a new versioned experiment.",
    }
    shared._atomic_json(finalization, ACTIVE_DIR / "finalization_manifest.json")
    pointer = {
        "schema": "current_presidential_model_pointer_v1",
        "active_version": "v27",
        "lifecycle_status": "frozen_pre_2025_evaluation",
        "canonical_document": "docs/FINAL_MODEL_V27_20260822.md",
        "finalization_manifest": "outputs/active_presidential_nested_v27/finalization_manifest.json",
        "runner": "scripts/run_active_presidential_model_v27.py",
        "prospective_runner": "scripts/run_prospective_forecast_v27.py",
        "config": "data/config/active_presidential_model_v23.json",
        "version_wrapper": "scripts/run_active_presidential_model_v27.py",
        "base_config_version": "v23",
        "output": "outputs/active_presidential_nested_v27",
        "predecessor": "v26",
        "rollback_pointer": "outputs/active_presidential_nested_v26/finalization_manifest.json",
        "regional_equal_election_macro_mae_pp": summary["metrics"]["regional_equal_election_macro_mae_pp"],
        "national_equal_election_macro_mae_pp": summary["metrics"]["national_equal_election_macro_mae_pp"],
        "winner_accuracy": summary["metrics"]["winner_accuracy"],
        "prediction_rows": summary["metrics"]["rows"],
        "prediction_sha256": prediction_hash,
        "predictive_intervals": "outputs/active_presidential_nested_v27/predictive_interval_manifest.json",
        "predictive_interval_levels": intervals["levels"],
        "predictive_interval_status": intervals["status"],
        "post_2022_outcomes_used": False,
    }
    shared._atomic_json(pointer, POINTER)
    previous_baseline = json.loads(
        (ROOT / "docs/GITHUB_BASELINE_20260822.json").read_text(encoding="utf-8")
    )
    baseline = {
        "schema": "github_repository_baseline_v1",
        "created_date": "2026-08-22",
        "source_workspace": str(ROOT),
        "active_version": "v27",
        "active_policy": "active_v27_core_weighted_party_regional_dispersion",
        "post_2022_outcomes_used": False,
        "license": previous_baseline["license"],
        "required_repository_files": ["NOTICE", "docs/FINAL_MODEL_V27_20260822.md", "scripts/audit_public_active_presidential_model_v27.py"],
        "expected_hashes": {
            "data/config/current_presidential_model.json": "written_after_baseline",
            "outputs/active_presidential_nested_v23/nested_predictions.csv": V23_SHA256,
            "outputs/active_presidential_nested_v24/nested_predictions.csv": V24_SHA256,
            "outputs/active_presidential_nested_v25/nested_predictions.csv": V25_SHA256,
            "outputs/active_presidential_nested_v26/nested_predictions.csv": V26_SHA256,
            "outputs/active_presidential_nested_v27/nested_predictions.csv": prediction_hash,
        },
        "verified_at_promotion": finalization["verification"],
        "tracked_file_max_bytes": previous_baseline["tracked_file_max_bytes"],
        "allowed_output_prefixes": sorted(set(previous_baseline["allowed_output_prefixes"]) | {"outputs/active_presidential_nested_v27/", "outputs/prospective_pres_2025_v27/"}),
    }
    # Resolve the pointer hash after it has been written.
    baseline["expected_hashes"]["data/config/current_presidential_model.json"] = shared._sha256(POINTER)
    shared._atomic_json(baseline, BASELINE)


if __name__ == "__main__":
    main()
