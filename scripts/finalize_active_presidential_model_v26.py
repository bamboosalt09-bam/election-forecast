"""Write V26 promotion, finalization, and GitHub baseline records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finalize_active_presidential_model_v25 as shared  # noqa: E402

ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v26"
OLD_BASELINE = ROOT / "docs/GITHUB_BASELINE_20260821.json"
NEW_BASELINE = ROOT / "docs/GITHUB_BASELINE_20260822.json"
V23_SHA256 = shared.V23_SHA256
V24_SHA256 = shared.V24_SHA256
V25_SHA256 = shared.V25_SHA256
CANONICAL_DOCUMENT = "docs/FINAL_MODEL_V26_20260822.md"
EXPERIMENT_DOCUMENT = "docs/EXPERIMENT_V25_INTENSITY_LADDER_20260822.md"

_sha256 = shared._sha256
_record = shared._record
_atomic_json = shared._atomic_json
_created_at = shared._created_at


def main() -> None:
    for name, version, expected in (
        ("V23", "v23", V23_SHA256),
        ("V24", "v24", V24_SHA256),
        ("V25", "v25", V25_SHA256),
    ):
        path = ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv"
        if _sha256(path) != expected:
            raise RuntimeError(f"{name} prediction artifact drift")

    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads(
        (ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    v26_sha256 = _sha256(ACTIVE_DIR / "nested_predictions.csv")

    promotion_path = ACTIVE_DIR / "promotion_manifest.json"
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "created_at_local": _created_at(promotion_path),
        "status": "promoted_frozen_pre_2025_evaluation",
        "active_version": "v26",
        "predecessor": "v25",
        "review_decision": "explicit_user_authorized_pointer_promotion",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "accepted_scope": [
            "graded_mega_issue_intensity_from_classifier_gate_proximity",
            "event_class_alignment_on_the_scored_path",
        ],
        "rejected_scope": [
            "margin_proportional_intensity_from_above",
            "graded_intensity_without_event_class_alignment",
            "target_specificity_gating_of_direct_mega_score",
        ],
        "selection_disclosure": (
            "The pairing was chosen by comparing the same five scored outcomes "
            "that measure it. No 2025 outcome was read, and no constant was "
            "introduced: the ceiling, the floors and both gates already existed."
        ),
        "interval_record": {
            "type": intervals["interval_type"],
            "levels": intervals["levels"],
            "residual_scale": intervals["residual_scale"],
            "candidate_outcomes": intervals["candidate_outcomes"],
            "status": intervals["status"],
        },
        "rollback": {
            "version": "v25",
            "prediction_sha256": V25_SHA256,
            "finalization_manifest": "outputs/active_presidential_nested_v25/finalization_manifest.json",
        },
        "artifacts": [
            _record("outputs/active_presidential_nested_v26/nested_predictions.csv"),
            _record("outputs/active_presidential_nested_v26/summary.json"),
            _record("outputs/active_presidential_nested_v26/input_manifest.csv"),
            _record("outputs/active_presidential_nested_v26/predictive_interval_manifest.json"),
            _record(EXPERIMENT_DOCUMENT),
        ],
    }
    _atomic_json(promotion, promotion_path)

    artifacts = [
        "data/config/current_presidential_model.json",
        "data/config/active_presidential_model_v23.json",
        "scripts/run_current_presidential_model.py",
        "scripts/run_active_presidential_model_v26.py",
        "scripts/build_automatic_controls_v26.py",
        "scripts/build_active_v26_predictive_intervals.py",
        "scripts/audit_public_active_presidential_model_v26.py",
        "presidential_issue_engine/mega_issue_intensity_ladder.py",
        "presidential_issue_engine/mega_issue_adjustment.py",
        "presidential_issue_engine/automatic_controls_v22.py",
        "presidential_issue_engine/v24_calibration.py",
        "presidential_issue_engine/strong_incumbent_veto.py",
        "presidential_issue_engine/third_candidate_lineage_constraint.py",
        "presidential_issue_engine/weak_same_lane_refusal.py",
        CANONICAL_DOCUMENT,
        EXPERIMENT_DOCUMENT,
        "docs/REPRODUCIBILITY.md",
        "outputs/automatic_controls_v26/mega_issue_intensity.csv",
        "outputs/active_presidential_nested_v26/input_manifest.csv",
        "outputs/active_presidential_nested_v26/nested_predictions.csv",
        "outputs/active_presidential_nested_v26/summary.json",
        "outputs/active_presidential_nested_v26/by_election.csv",
        "outputs/active_presidential_nested_v26/national_predictions.csv",
        "outputs/active_presidential_nested_v26/weak_same_lane_refusal_audit.csv",
        "outputs/active_presidential_nested_v26/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v26/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v26/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v26/predictive_interval_manifest.json",
        "outputs/active_presidential_nested_v26/promotion_manifest.json",
    ]
    finalization_path = ACTIVE_DIR / "finalization_manifest.json"
    finalization = {
        "schema": "presidential_model_finalization_v1",
        "created_at_local": _created_at(finalization_path),
        "status": "frozen_pre_2025_evaluation",
        "active_version": "v26",
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
        "metrics": summary["metrics"],
        "predictive_intervals": {
            "status": intervals["status"],
            "levels": intervals["levels"],
            "residual_scale": intervals["residual_scale"],
            "candidate_outcomes": intervals["candidate_outcomes"],
            "target_outcomes_used_to_construct_bounds": False,
            "post_2022_outcomes_used": False,
        },
        "verification": {
            "v23_rollback_hash_match": True,
            "v24_rollback_hash_match": True,
            "v25_rollback_hash_match": True,
            "v26_prediction_hash": v26_sha256,
            "public_active_v26_audit": "pass_after_finalization",
        },
        "artifacts": [_record(path) for path in artifacts],
        "rollback": {
            "version": "v25",
            "prediction_sha256": V25_SHA256,
            "finalization_manifest": "outputs/active_presidential_nested_v25/finalization_manifest.json",
        },
        "change_policy": (
            "Do not modify V23, V24, V25, or V26 in place; use a new versioned experiment."
        ),
    }
    _atomic_json(finalization, finalization_path)

    old = json.loads(OLD_BASELINE.read_text(encoding="utf-8"))
    baseline = {
        "schema": "github_repository_baseline_v1",
        "created_date": "2026-08-22",
        "source_workspace": str(ROOT),
        "active_version": "v26",
        "active_policy": "active_v26_graded_mega_intensity_event_aligned_pre_2025",
        "post_2022_outcomes_used": False,
        "license": old["license"],
        "required_repository_files": [
            "NOTICE",
            CANONICAL_DOCUMENT,
            "scripts/audit_public_active_presidential_model_v26.py",
        ],
        "expected_hashes": {
            "data/config/current_presidential_model.json": _sha256(
                ROOT / "data/config/current_presidential_model.json"
            ),
            "outputs/active_presidential_nested_v23/nested_predictions.csv": V23_SHA256,
            "outputs/active_presidential_nested_v24/nested_predictions.csv": V24_SHA256,
            "outputs/active_presidential_nested_v25/nested_predictions.csv": V25_SHA256,
            "outputs/active_presidential_nested_v26/finalization_manifest.json": _sha256(
                finalization_path
            ),
            "outputs/active_presidential_nested_v26/nested_predictions.csv": v26_sha256,
        },
        "verified_at_promotion": finalization["verification"],
        "tracked_file_max_bytes": old["tracked_file_max_bytes"],
        "allowed_output_prefixes": sorted(
            set(old["allowed_output_prefixes"])
            | {
                "outputs/active_presidential_nested_v26/",
                "outputs/automatic_controls_v26/",
            }
        ),
    }
    _atomic_json(baseline, NEW_BASELINE)
    print(json.dumps(finalization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
