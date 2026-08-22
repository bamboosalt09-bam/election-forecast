"""Promote and freeze the external-model-runtime-free V28 runtime."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_active_presidential_model_v25 as shared  # noqa: E402

ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v28"
POINTERS = (
    ROOT / "data/config/current_presidential_model.json",
    ROOT / "data/config/active_presidential_model.json",
)
V27_SHA256 = "f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b"


def main() -> None:
    if shared._sha256(ROOT / "outputs/active_presidential_nested_v27/nested_predictions.csv") != V27_SHA256:
        raise RuntimeError("V27 rollback prediction drift")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads((ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8"))
    prediction_hash = shared._sha256(ACTIVE_DIR / "nested_predictions.csv")
    artifacts = [
        "scripts/run_active_presidential_model_v28.py",
        "scripts/run_prospective_forecast_v28.py",
        "scripts/build_active_v28_predictive_intervals.py",
        "scripts/finalize_active_presidential_model_v28.py",
        "scripts/audit_public_active_presidential_model_v28.py",
        "scripts/verify_v28_clean_reproduction.py",
        "presidential_issue_engine/external_model_free_runtime.py",
        "presidential_issue_engine/issue_vote_engine.py",
        "presidential_issue_engine/fixed_dataset/kospi_election_context.csv",
        "docs/FINAL_MODEL_V28_20260822.md",
        "docs/EXPERIMENT_REMOVE_EXTERNAL_MODEL_OVERLAY_20260822.md",
        "outputs/active_presidential_nested_v28/nested_predictions.csv",
        "outputs/active_presidential_nested_v28/summary.json",
        "outputs/active_presidential_nested_v28/by_election.csv",
        "outputs/active_presidential_nested_v28/national_predictions.csv",
        "outputs/active_presidential_nested_v28/input_manifest.csv",
        "outputs/active_presidential_nested_v28/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v28/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v28/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v28/predictive_interval_manifest.json",
    ]
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "status": "promoted_external_model_runtime_free",
        "active_version": "v28",
        "predecessor": "v27",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "accepted_scope": ["remove_external_model_derived_stance_overlay"],
        "selection_disclosure": "Adopted to remove neural runtime, model weights, sentence corpora, the direct stance overlay and unused descendants. The frozen historical candidate-issue aggregate remains disclosed because full removal materially degrades the development panel; historical predictions remain byte-identical to V27.",
        "rollback": {"version": "v27", "prediction_sha256": V27_SHA256},
    }
    shared._atomic_json(promotion, ACTIVE_DIR / "promotion_manifest.json")
    artifacts.append("outputs/active_presidential_nested_v28/promotion_manifest.json")
    finalization = {
        "schema": "presidential_model_finalization_v1",
        "status": "frozen_external_model_runtime_free",
        "active_version": "v28",
        "base_config_version": "v23",
        "scored_development_elections": ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"],
        "post_2022_outcomes_used": False,
        "untouched_historical_holdout": False,
        "external_neural_model_runtime": False,
        "external_model_derived_inputs": [
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        ],
        "metrics": summary["metrics"],
        "predictive_intervals": {"status": intervals["status"], "levels": intervals["levels"], "post_2022_outcomes_used": False},
        "verification": {
            "v27_rollback_hash_match": True,
            "v28_prediction_hash": prediction_hash,
            "v28_historical_predictions_byte_identical_to_v27": prediction_hash == V27_SHA256,
        },
        "artifacts": [shared._record(path) for path in artifacts],
        "rollback": {"version": "v27", "prediction_sha256": V27_SHA256, "finalization_manifest": "outputs/active_presidential_nested_v27/finalization_manifest.json"},
        "change_policy": "Do not modify V23 through V28 in place; use a new versioned experiment.",
    }
    shared._atomic_json(finalization, ACTIVE_DIR / "finalization_manifest.json")
    pointer = {
        "schema": "current_presidential_model_pointer_v1",
        "active_version": "v28",
        "lifecycle_status": "frozen_external_model_runtime_free",
        "canonical_document": "docs/FINAL_MODEL_V28_20260822.md",
        "finalization_manifest": "outputs/active_presidential_nested_v28/finalization_manifest.json",
        "runner": "scripts/run_active_presidential_model_v28.py",
        "prospective_runner": "scripts/run_prospective_forecast_v28.py",
        "config": "data/config/active_presidential_model_v23.json",
        "version_wrapper": "scripts/run_active_presidential_model_v28.py",
        "base_config_version": "v23",
        "output": "outputs/active_presidential_nested_v28",
        "predecessor": "v27",
        "rollback_pointer": "outputs/active_presidential_nested_v27/finalization_manifest.json",
        "regional_equal_election_macro_mae_pp": summary["metrics"]["regional_equal_election_macro_mae_pp"],
        "national_equal_election_macro_mae_pp": summary["metrics"]["national_equal_election_macro_mae_pp"],
        "winner_accuracy": summary["metrics"]["winner_accuracy"],
        "prediction_rows": summary["metrics"]["rows"],
        "prediction_sha256": prediction_hash,
        "predictive_intervals": "outputs/active_presidential_nested_v28/predictive_interval_manifest.json",
        "predictive_interval_levels": intervals["levels"],
        "predictive_interval_status": intervals["status"],
        "external_neural_model_runtime": False,
        "external_model_derived_inputs": [
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        ],
        "post_2022_outcomes_used": False,
    }
    for path in POINTERS:
        shared._atomic_json(pointer, path)


if __name__ == "__main__":
    main()
