"""Write the reviewed V24 promotion, finalization, and GitHub baseline records."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v24"
OLD_BASELINE = ROOT / "docs" / "GITHUB_BASELINE_20260810.json"
NEW_BASELINE = ROOT / "docs" / "GITHUB_BASELINE_20260820.json"
V23_SHA256 = "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b"
V24_SHA256 = "edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"finalization artifact missing: {relative}")
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _created_at(path: Path) -> str:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["created_at_local"]
    return datetime.now().astimezone().isoformat()


def main() -> None:
    v23 = ROOT / "outputs" / "active_presidential_nested_v23" / "nested_predictions.csv"
    v24 = ACTIVE_DIR / "nested_predictions.csv"
    if _sha256(v23) != V23_SHA256:
        raise RuntimeError("V23 rollback artifact drift")
    if _sha256(v24) != V24_SHA256:
        raise RuntimeError("V24 prediction artifact drift")

    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads(
        (ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    promotion_path = ACTIVE_DIR / "promotion_manifest.json"
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "created_at_local": _created_at(promotion_path),
        "status": "promoted_frozen_pre_2025_evaluation",
        "active_version": "v24",
        "predecessor": "v23",
        "review_decision": "explicit_user_authorized_pointer_promotion",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "panel_comparison_warning": (
            "V24 has 232 rows after restoring weak third candidates; V23 has 199 rows. "
            "Headline differences are not a clean same-panel ablation."
        ),
        "accepted_scope": [
            "uniform_1pp_scored_floor",
            "ballot_faithful_withdrawal_slots",
            "continuous_nonmajor_organization_strength",
            "third_candidate_lineage_ceiling",
            "strong_incumbent_veto",
            "weak_same_lane_refusal_with_1pp_theoretical_floor",
        ],
        "interval_record": {
            "type": intervals["interval_type"],
            "levels": intervals["levels"],
            "residual_scale": intervals["residual_scale"],
            "scale_policy": intervals["residual_scale_policy"],
            "candidate_outcomes": intervals["candidate_outcomes"],
            "status": intervals["status"],
        },
        "manual_selection_disclosure": {
            "strict_nested_postprocess_selection": summary[
                "strict_nested_postprocess_selection"
            ],
            "candidate_numeric_parameters_historically_development_selected": summary[
                "candidate_numeric_parameters_historically_development_selected"
            ],
            "untouched_historical_holdout": summary["untouched_historical_holdout"],
        },
        "rollback": {
            "version": "v23",
            "prediction_sha256": V23_SHA256,
            "finalization_manifest": (
                "outputs/active_presidential_nested_v23/finalization_manifest.json"
            ),
        },
        "artifacts": [
            _record("outputs/active_presidential_nested_v24/nested_predictions.csv"),
            _record("outputs/active_presidential_nested_v24/summary.json"),
            _record("outputs/active_presidential_nested_v24/input_manifest.csv"),
            _record("outputs/active_presidential_nested_v24/predictive_interval_manifest.json"),
            _record("docs/EXPERIMENT_V24_LINEAGE_20260819.md"),
        ],
    }
    _atomic_json(promotion, promotion_path)

    artifact_paths = [
        "data/config/current_presidential_model.json",
        "data/config/active_presidential_model_v23.json",
        "scripts/run_current_presidential_model.py",
        "scripts/run_active_presidential_model_v24.py",
        "scripts/build_active_v24_predictive_intervals.py",
        "scripts/audit_public_active_presidential_model_v24.py",
        "presidential_issue_engine/v24_calibration.py",
        "presidential_issue_engine/strong_incumbent_veto.py",
        "presidential_issue_engine/third_candidate_lineage_constraint.py",
        "presidential_issue_engine/weak_same_lane_refusal.py",
        "docs/FINAL_MODEL_V24_20260820.md",
        "docs/EXPERIMENT_V24_LINEAGE_20260819.md",
        "docs/REPRODUCIBILITY.md",
        "outputs/active_presidential_nested_v24/input_manifest.csv",
        "outputs/active_presidential_nested_v24/nested_predictions.csv",
        "outputs/active_presidential_nested_v24/summary.json",
        "outputs/active_presidential_nested_v24/by_election.csv",
        "outputs/active_presidential_nested_v24/national_predictions.csv",
        "outputs/active_presidential_nested_v24/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v24/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v24/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v24/predictive_interval_manifest.json",
        "outputs/active_presidential_nested_v24/promotion_manifest.json",
    ]
    for path in sorted((ROOT / "presidential_issue_engine/fixed_dataset/v24").glob("*")):
        if path.is_file():
            artifact_paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    finalization_path = ACTIVE_DIR / "finalization_manifest.json"
    finalization = {
        "schema": "presidential_model_finalization_v1",
        "created_at_local": _created_at(finalization_path),
        "status": "frozen_pre_2025_evaluation",
        "active_version": "v24",
        "base_config_version": "v23",
        "warmup_elections": ["pres_1992", "pres_1997"],
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
            "target_outcomes_used_to_construct_bounds": intervals[
                "target_outcomes_used_to_construct_bounds"
            ],
            "post_2022_outcomes_used": intervals["post_2022_outcomes_used"],
        },
        "verification": {
            "v24_exact_reproduction_to_temporary_directory": "pass",
            "regression_suite": "595 passed",
            "public_active_v24_audit": "pass",
            "github_boundary_audit": "pass",
            "v23_rollback_hash_match": True,
            "v24_prediction_hash": V24_SHA256,
        },
        "artifacts": [_record(path) for path in artifact_paths],
        "rollback": {
            "version": "v23",
            "prediction_sha256": V23_SHA256,
            "finalization_manifest": (
                "outputs/active_presidential_nested_v23/finalization_manifest.json"
            ),
        },
        "change_policy": (
            "Do not modify V23 or V24 in place; use a new versioned experiment "
            "and an explicit promotion change."
        ),
    }
    _atomic_json(finalization, finalization_path)

    old = json.loads(OLD_BASELINE.read_text(encoding="utf-8"))
    baseline = {
        "schema": "github_repository_baseline_v1",
        "created_date": "2026-08-20",
        "source_workspace": str(ROOT),
        "active_version": "v24",
        "active_policy": "active_v24_structural_residual_pre_2025",
        "post_2022_outcomes_used": False,
        "license": old["license"],
        "required_repository_files": [
            "NOTICE",
            "docs/FINAL_MODEL_V24_20260820.md",
            "scripts/audit_public_active_presidential_model_v24.py",
        ],
        "expected_hashes": {
            "data/config/current_presidential_model.json": _sha256(
                ROOT / "data/config/current_presidential_model.json"
            ),
            "outputs/active_presidential_nested_v23/finalization_manifest.json": _sha256(
                ROOT / "outputs/active_presidential_nested_v23/finalization_manifest.json"
            ),
            "outputs/active_presidential_nested_v23/nested_predictions.csv": V23_SHA256,
            "outputs/active_presidential_nested_v24/finalization_manifest.json": _sha256(
                finalization_path
            ),
            "outputs/active_presidential_nested_v24/nested_predictions.csv": V24_SHA256,
        },
        "verified_at_promotion": finalization["verification"],
        "tracked_file_max_bytes": old["tracked_file_max_bytes"],
        "allowed_output_prefixes": sorted(
            set(old["allowed_output_prefixes"])
            | {
                "outputs/active_presidential_nested_v24/",
                "outputs/v24_floor_recalibration_hypotheses/",
                "outputs/v24_structural_residual_hypotheses/",
            }
        ),
    }
    _atomic_json(baseline, NEW_BASELINE)
    print(json.dumps(finalization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
