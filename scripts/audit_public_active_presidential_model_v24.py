"""Audit the frozen public V24 pointer, lineage, intervals, and V23 rollback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v24"
V23_PREDICTIONS = (
    ROOT / "outputs" / "active_presidential_nested_v23" / "nested_predictions.csv"
)
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
EXTERNAL_INPUTS = (
    ROOT / "data" / "raw" / "official_sources" / "external_active_inputs.json"
)
V23_SHA256 = "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b"
V24_SHA256 = "edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a"
ORDER = ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_rollback() -> dict[str, object]:
    _require(_sha256(V23_PREDICTIONS) == V23_SHA256, "V23 rollback prediction drift")
    return {"v23_rollback_sha256": V23_SHA256}


def _audit_pointer() -> dict[str, object]:
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    expected = {
        "active_version": "v24",
        "runner": "scripts/run_active_presidential_model_v24.py",
        "output": "outputs/active_presidential_nested_v24",
        "canonical_document": "docs/FINAL_MODEL_V24_20260820.md",
        "finalization_manifest": (
            "outputs/active_presidential_nested_v24/finalization_manifest.json"
        ),
        "config": "data/config/active_presidential_model_v23.json",
        "base_config_version": "v23",
        "predecessor": "v23",
    }
    for key, value in expected.items():
        _require(pointer.get(key) == value, f"active pointer mismatch: {key}")
    _require(pointer.get("post_2022_outcomes_used") is False, "pointer permits post-2022 outcomes")
    return {"active_version": pointer["active_version"]}


def _audit_input_manifest() -> dict[str, object]:
    manifest = pd.read_csv(ACTIVE_DIR / "input_manifest.csv", encoding="utf-8-sig")
    external_payload = json.loads(EXTERNAL_INPUTS.read_text(encoding="utf-8"))
    external = {
        str(row["path"]).replace("\\", "/"): row
        for row in external_payload.get("inputs", [])
    }
    for row in manifest.itertuples(index=False):
        relative = str(row.path).replace("\\", "/")
        if relative.startswith("generated:"):
            continue
        path = ROOT / relative
        if path.exists():
            _require(_sha256(path) == str(row.sha256), f"input hash drift: {relative}")
            continue
        record = external.get(relative)
        _require(record is not None, f"manifest input missing from clone and record: {relative}")
        _require(record.get("status") == "excluded_from_git", f"bad external status: {relative}")
        _require(
            int(record.get("bytes", -1)) == int(row.bytes)
            and str(record.get("sha256", "")) == str(row.sha256),
            f"external input record drift: {relative}",
        )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT,
            capture_output=True,
        )
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=ROOT,
            capture_output=True,
        )
        _require(tracked.returncode != 0, f"external input is tracked: {relative}")
        _require(ignored.returncode == 0, f"external input is not ignored: {relative}")
    return {"v24_input_manifest_files": len(manifest)}


def _audit_predictions() -> dict[str, object]:
    path = ACTIVE_DIR / "nested_predictions.csv"
    _require(_sha256(path) == V24_SHA256, "V24 canonical prediction drift")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    _require(len(frame) == 232, "V24 prediction row count is not 232")
    _require(set(frame["election_id"].astype(str)) == set(ORDER), "V24 election panel drift")
    _require(
        not frame["election_id"].astype(str).str.contains("2025").any(),
        "2025 row found in active V24 predictions",
    )
    totals = frame.groupby(["election_id", "region_id"])["layer_pred"].sum()
    _require(np.allclose(totals, 1.0, atol=1e-12), "V24 regional predictions are not compositional")

    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    by_election = pd.read_csv(ACTIVE_DIR / "by_election.csv", encoding="utf-8-sig")
    _require(metrics["rows"] == len(frame), "summary row count drift")
    _require(
        np.isclose(
            metrics["regional_equal_election_macro_mae_pp"],
            by_election["regional_weighted_mae_pp"].mean(),
            atol=1e-12,
        ),
        "regional macro metric drift",
    )
    _require(
        np.isclose(
            metrics["national_equal_election_macro_mae_pp"],
            by_election["national_candidate_mae_pp"].mean(),
            atol=1e-12,
        ),
        "national macro metric drift",
    )
    return {
        "v24_prediction_rows": len(frame),
        "v24_prediction_sha256": V24_SHA256,
    }


def _audit_intervals() -> dict[str, object]:
    manifest_path = ACTIVE_DIR / "predictive_interval_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest["input_sha256"] == V24_SHA256, "interval input hash drift")
    _require(manifest["post_2022_outcomes_used"] is False, "post-2022 interval outcome used")
    _require(
        manifest["target_outcomes_used_to_construct_bounds"] is False,
        "target result used to construct an interval",
    )
    _require(
        manifest["residual_scale_policy"] == "fixed_unscaled_not_selected_on_coverage",
        "interval scale policy drift",
    )

    detail = pd.read_csv(ACTIVE_DIR / "national_predictive_intervals.csv")
    _require(len(detail) == 44, "interval row count is not 44")
    _require(set(detail["nominal_level"].round(2)) == {0.5, 0.8, 0.9, 0.95}, "interval levels drift")
    _require(set(detail["election_id"]) == set(ORDER[1:]), "interval target folds drift")
    _require(
        not detail["target_outcome_used_to_construct_bounds"].astype(bool).any(),
        "interval detail marks target outcome use",
    )
    position = {election: index for index, election in enumerate(ORDER)}
    for row in detail.itertuples(index=False):
        training = str(row.training_elections).split("|")
        _require(
            all(position[election] < position[row.election_id] for election in training),
            f"nonchronological interval training: {row.election_id}",
        )
    for keys, group in detail.groupby(["election_id", "slot"], sort=False):
        ordered = group.sort_values("nominal_level")
        _require(ordered["lower_share"].is_monotonic_decreasing, f"lower interval nesting drift: {keys}")
        _require(ordered["upper_share"].is_monotonic_increasing, f"upper interval nesting drift: {keys}")
    return {"interval_candidate_level_rows": len(detail)}


def _audit_finalization() -> dict[str, object]:
    path = ACTIVE_DIR / "finalization_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest["active_version"] == "v24", "finalization version drift")
    _require(manifest["post_2022_outcomes_used"] is False, "finalization uses post-2022 outcome")
    for record in manifest["artifacts"]:
        artifact = ROOT / record["path"]
        _require(artifact.is_file(), f"finalized artifact missing: {record['path']}")
        _require(_sha256(artifact) == record["sha256"], f"finalized artifact drift: {record['path']}")
    return {"finalized_artifacts": len(manifest["artifacts"])}


def main() -> None:
    results: dict[str, object] = {}
    for audit in (
        _audit_rollback,
        _audit_pointer,
        _audit_input_manifest,
        _audit_predictions,
        _audit_intervals,
        _audit_finalization,
    ):
        results.update(audit())
    print("[active V24 public audit: PASS]")
    for key, value in results.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
