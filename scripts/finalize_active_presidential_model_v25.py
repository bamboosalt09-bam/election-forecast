"""Write V25 promotion, finalization, and GitHub baseline records."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v25"
OLD_BASELINE = ROOT / "docs/GITHUB_BASELINE_20260820.json"
NEW_BASELINE = ROOT / "docs/GITHUB_BASELINE_20260821.json"
V23_SHA256 = "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b"
V24_SHA256 = "edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a"
V25_SHA256 = "218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b"
NORMALIZED_TEXT_PREFIXES = (
    ".github/", "data/config/", "docs/", "presidential_issue_engine/", "scripts/", "tests/",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise RuntimeError(f"finalization artifact missing: {relative}")
    normalized = relative.replace("\\", "/").startswith(NORMALIZED_TEXT_PREFIXES)
    content = path.read_bytes()
    if normalized:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return {
        "path": relative.replace("\\", "/"),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "hash_mode": "normalized_text_lf" if normalized else "raw_bytes",
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
    checks = {
        "V23": (ROOT / "outputs/active_presidential_nested_v23/nested_predictions.csv", V23_SHA256),
        "V24": (ROOT / "outputs/active_presidential_nested_v24/nested_predictions.csv", V24_SHA256),
        "V25": (ACTIVE_DIR / "nested_predictions.csv", V25_SHA256),
    }
    for name, (path, expected) in checks.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"{name} prediction artifact drift")

    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads((ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8"))
    promotion_path = ACTIVE_DIR / "promotion_manifest.json"
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "created_at_local": _created_at(promotion_path),
        "status": "promoted_frozen_pre_2025_evaluation",
        "active_version": "v25",
        "predecessor": "v24",
        "review_decision": "explicit_user_authorized_pointer_promotion",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "accepted_scope": [
            "bounded_v23_runtime_lineage_repair",
            "v24_ballot_and_scored_scope",
            "v24_prediction_tilted_weak_c_route",
            "preserved_v24_third_candidate_profile_pressure_paths",
        ],
        "rejected_scope": [
            "affinity_only_weak_c_route",
            "v23_automatic_third_candidate_profile_pressure_rebind",
            "v24_conversion_context_upper_route_rebind",
        ],
        "interval_record": {
            "type": intervals["interval_type"],
            "levels": intervals["levels"],
            "residual_scale": intervals["residual_scale"],
            "candidate_outcomes": intervals["candidate_outcomes"],
            "status": intervals["status"],
        },
        "rollback": {
            "version": "v24",
            "prediction_sha256": V24_SHA256,
            "finalization_manifest": "outputs/active_presidential_nested_v24/finalization_manifest.json",
        },
        "artifacts": [
            _record("outputs/active_presidential_nested_v25/nested_predictions.csv"),
            _record("outputs/active_presidential_nested_v25/summary.json"),
            _record("outputs/active_presidential_nested_v25/input_manifest.csv"),
            _record("outputs/active_presidential_nested_v25/predictive_interval_manifest.json"),
            _record("docs/V24_RUNTIME_LINEAGE_DEFECT_20260821.md"),
        ],
    }
    _atomic_json(promotion, promotion_path)

    artifacts = [
        "data/config/current_presidential_model.json",
        "data/config/active_presidential_model_v23.json",
        "scripts/run_current_presidential_model.py",
        "scripts/run_active_presidential_model_v25.py",
        "scripts/build_active_v25_predictive_intervals.py",
        "scripts/audit_public_active_presidential_model_v25.py",
        "presidential_issue_engine/v24_calibration.py",
        "presidential_issue_engine/strong_incumbent_veto.py",
        "presidential_issue_engine/third_candidate_lineage_constraint.py",
        "presidential_issue_engine/weak_same_lane_refusal.py",
        "docs/FINAL_MODEL_V25_20260821.md",
        "docs/V24_RUNTIME_LINEAGE_DEFECT_20260821.md",
        "docs/REPRODUCIBILITY.md",
        "outputs/active_presidential_nested_v25/input_manifest.csv",
        "outputs/active_presidential_nested_v25/nested_predictions.csv",
        "outputs/active_presidential_nested_v25/summary.json",
        "outputs/active_presidential_nested_v25/by_election.csv",
        "outputs/active_presidential_nested_v25/national_predictions.csv",
        "outputs/active_presidential_nested_v25/weak_same_lane_refusal_audit.csv",
        "outputs/active_presidential_nested_v25/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v25/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v25/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v25/predictive_interval_manifest.json",
        "outputs/active_presidential_nested_v25/promotion_manifest.json",
    ]
    finalization_path = ACTIVE_DIR / "finalization_manifest.json"
    finalization = {
        "schema": "presidential_model_finalization_v1",
        "created_at_local": _created_at(finalization_path),
        "status": "frozen_pre_2025_evaluation",
        "active_version": "v25",
        "base_config_version": "v23",
        "scored_development_elections": ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"],
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
            "v25_exact_reproduction_to_same_hash": "pass",
            "regression_suite": "601 passed",
            "public_active_v25_audit": "pass_after_finalization",
            "v23_rollback_hash_match": True,
            "v24_rollback_hash_match": True,
            "v25_prediction_hash": V25_SHA256,
            "prospective_historical_reproduction_rows": 232,
        },
        "artifacts": [_record(path) for path in artifacts],
        "rollback": {
            "version": "v24",
            "prediction_sha256": V24_SHA256,
            "finalization_manifest": "outputs/active_presidential_nested_v24/finalization_manifest.json",
        },
        "change_policy": "Do not modify V23, V24, or V25 in place; use a new versioned experiment.",
    }
    _atomic_json(finalization, finalization_path)

    old = json.loads(OLD_BASELINE.read_text(encoding="utf-8"))
    baseline = {
        "schema": "github_repository_baseline_v1",
        "created_date": "2026-08-21",
        "source_workspace": str(ROOT),
        "active_version": "v25",
        "active_policy": "active_v25_bounded_runtime_repair_pre_2025",
        "post_2022_outcomes_used": False,
        "license": old["license"],
        "required_repository_files": [
            "NOTICE", "docs/FINAL_MODEL_V25_20260821.md", "scripts/audit_public_active_presidential_model_v25.py",
        ],
        "expected_hashes": {
            "data/config/current_presidential_model.json": _sha256(ROOT / "data/config/current_presidential_model.json"),
            "outputs/active_presidential_nested_v23/nested_predictions.csv": V23_SHA256,
            "outputs/active_presidential_nested_v24/nested_predictions.csv": V24_SHA256,
            "outputs/active_presidential_nested_v25/finalization_manifest.json": _sha256(finalization_path),
            "outputs/active_presidential_nested_v25/nested_predictions.csv": V25_SHA256,
        },
        "verified_at_promotion": finalization["verification"],
        "tracked_file_max_bytes": old["tracked_file_max_bytes"],
        "allowed_output_prefixes": sorted(
            set(old["allowed_output_prefixes"])
            | {
                "outputs/active_presidential_nested_v25/",
                "outputs/prospective_pres_2025_v24/",
                "outputs/prospective_pres_2025_v25/",
            }
        ),
    }
    _atomic_json(baseline, NEW_BASELINE)
    print(json.dumps(finalization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
