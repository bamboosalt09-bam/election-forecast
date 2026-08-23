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

ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v29"
POINTERS = (
    ROOT / "data/config/current_presidential_model.json",
    ROOT / "data/config/active_presidential_model.json",
)
V28_SHA256 = "23d6efd825244caa1f7b06b84e94cf581f00c6184aeb80769d8bb3d4c2a19fba"

PROSPECTIVE_DEMONSTRATION = {
    "artifact": "outputs/prospective_pres_2025_v28",
    "regenerated_for_v29": False,
    "blocked_by": "docs/DIAGNOSIS_PROSPECTIVE_2025_PATH_20260823.md",
    "reason": (
        "The prospective harness has been unrunnable since the V28 external-model "
        "boundary was enforced process-wide, because it asserts byte-identical "
        "reproduction of a V25 history frozen before that boundary. The published "
        "artifact predates the enforcement and therefore used the seed inputs V28 "
        "excludes; regenerating it changes the published forecast."
    ),
}

SELECTION_DISCLOSURE = (
    "Adopted at gain 1.0, where the expansion factor is the predicted third share "
    "itself and no constant is selected. A swept gain of 0.50 gives a better "
    "regional macro (2.555129 against 2.573607) and was rejected because it is a "
    "constant chosen on the same five scored outcomes it is then measured against. "
    "The national macro is unchanged to nine decimals at every gain, so this choice "
    "costs nothing on that axis. The decision to build a dispersion correction at "
    "all was made by reading the residuals of those same five outcomes and is "
    "in-sample."
)


def main() -> None:
    rollback = ROOT / "outputs/active_presidential_nested_v28/nested_predictions.csv"
    if shared._sha256(rollback) != V28_SHA256:
        raise RuntimeError("V28 rollback prediction drift")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads(
        (ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    prediction_hash = shared._sha256(ACTIVE_DIR / "nested_predictions.csv")
    artifacts = [
        "scripts/run_active_presidential_model_v29.py",
        "scripts/run_prospective_forecast_v29.py",
        "scripts/build_active_v29_predictive_intervals.py",
        "scripts/evaluate_electorate_layers.py",
        "scripts/finalize_active_presidential_model_v29.py",
        "scripts/audit_public_active_presidential_model_v29.py",
        "scripts/verify_v29_clean_reproduction.py",
        "presidential_issue_engine/third_share_dispersion_expansion.py",
        "presidential_issue_engine/external_model_free_runtime.py",
        "presidential_issue_engine/issue_vote_engine.py",
        "presidential_issue_engine/fixed_dataset/kospi_election_context.csv",
        "docs/FINAL_MODEL_V29_20260823.md",
        "docs/EXPERIMENT_V29_THIRD_SHARE_DISPERSION_20260823.md",
        "outputs/active_presidential_nested_v29/nested_predictions.csv",
        "outputs/active_presidential_nested_v29/summary.json",
        "outputs/active_presidential_nested_v29/by_election.csv",
        "outputs/active_presidential_nested_v29/national_predictions.csv",
        "outputs/active_presidential_nested_v29/input_manifest.csv",
        "outputs/active_presidential_nested_v29/third_share_dispersion_expansion_audit.csv",
        "outputs/active_presidential_nested_v29/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v29/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v29/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v29/predictive_interval_manifest.json",
    ]
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "status": "promoted_third_share_dispersion_expansion",
        "active_version": "v29",
        "predecessor": "v28",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "accepted_scope": ["third_share_indexed_regional_dispersion_expansion"],
        "rejected_scope": [
            "swept_gain_0_50_better_regional_but_fitted_on_scored_panel"
        ],
        "selection_disclosure": SELECTION_DISCLOSURE,
        "rollback": {"version": "v28", "prediction_sha256": V28_SHA256},
    }
    shared._atomic_json(promotion, ACTIVE_DIR / "promotion_manifest.json")
    artifacts.append("outputs/active_presidential_nested_v29/promotion_manifest.json")

    finalization = {
        "schema": "presidential_model_finalization_v1",
        "status": "frozen_third_share_dispersion_expansion",
        "active_version": "v29",
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
            "v28_rollback_hash_match": True,
            "v29_prediction_hash": prediction_hash,
            "national_macro_unchanged_from_v28": True,
        },
        "artifacts": [shared._record(path) for path in artifacts],
        "rollback": {
            "version": "v28",
            "prediction_sha256": V28_SHA256,
            "finalization_manifest": (
                "outputs/active_presidential_nested_v28/finalization_manifest.json"
            ),
        },
        "change_policy": (
            "Do not modify V23 through V29 in place; use a new versioned experiment."
        ),
    }
    shared._atomic_json(finalization, ACTIVE_DIR / "finalization_manifest.json")

    metrics = summary["metrics"]
    pointer = {
        "schema": "current_presidential_model_pointer_v1",
        "active_version": "v29",
        "lifecycle_status": "frozen_third_share_dispersion_expansion",
        "canonical_document": "docs/FINAL_MODEL_V29_20260823.md",
        "finalization_manifest": (
            "outputs/active_presidential_nested_v29/finalization_manifest.json"
        ),
        "runner": "scripts/run_active_presidential_model_v29.py",
        "prospective_runner": "scripts/run_prospective_forecast_v29.py",
        "config": "data/config/active_presidential_model_v23.json",
        "version_wrapper": "scripts/run_active_presidential_model_v29.py",
        "base_config_version": "v23",
        "output": "outputs/active_presidential_nested_v29",
        "predecessor": "v28",
        "rollback_pointer": (
            "outputs/active_presidential_nested_v28/finalization_manifest.json"
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
            "outputs/active_presidential_nested_v29/predictive_interval_manifest.json"
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


if __name__ == "__main__":
    main()
