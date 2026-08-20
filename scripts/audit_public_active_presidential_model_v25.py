"""Audit the frozen public V25 pointer and V23/V24 rollback boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v25"
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
EXTERNAL_INPUTS = ROOT / "data/raw/official_sources/external_active_inputs.json"
V23_SHA256 = "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b"
V24_SHA256 = "edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a"
V25_SHA256 = "218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b"
ORDER = ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_sha256(path: Path, hash_mode: str) -> str:
    content = path.read_bytes()
    if hash_mode == "normalized_text_lf":
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    elif hash_mode != "raw_bytes":
        raise RuntimeError(f"unknown finalization hash mode: {hash_mode}")
    return hashlib.sha256(content).hexdigest()


def _audit_rollbacks() -> dict[str, object]:
    v23 = ROOT / "outputs/active_presidential_nested_v23/nested_predictions.csv"
    v24 = ROOT / "outputs/active_presidential_nested_v24/nested_predictions.csv"
    _require(_sha256(v23) == V23_SHA256, "V23 rollback prediction drift")
    _require(_sha256(v24) == V24_SHA256, "V24 rollback prediction drift")
    return {"v23_rollback_sha256": V23_SHA256, "v24_rollback_sha256": V24_SHA256}


def _audit_pointer() -> dict[str, object]:
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    expected = {
        "active_version": "v25",
        "runner": "scripts/run_active_presidential_model_v25.py",
        "output": "outputs/active_presidential_nested_v25",
        "canonical_document": "docs/FINAL_MODEL_V25_20260821.md",
        "finalization_manifest": "outputs/active_presidential_nested_v25/finalization_manifest.json",
        "config": "data/config/active_presidential_model_v23.json",
        "base_config_version": "v23",
        "predecessor": "v24",
    }
    for key, value in expected.items():
        _require(pointer.get(key) == value, f"active pointer mismatch: {key}")
    _require(pointer.get("post_2022_outcomes_used") is False, "pointer permits post-2022 outcomes")
    return {"active_version": "v25"}


def _audit_inputs() -> dict[str, object]:
    manifest = pd.read_csv(ACTIVE_DIR / "input_manifest.csv", encoding="utf-8-sig")
    paths = manifest["path"].astype(str).str.replace("\\", "/", regex=False)
    _require("data/raw/third_candidate_profile.csv" in set(paths), "V24 third profile not preserved")
    _require("data/raw/third_candidate_pressure.csv" in set(paths), "V24 third pressure not preserved")
    _require(
        "outputs/automatic_controls_v23/third_candidate_profile.csv" not in set(paths),
        "rejected V23 third profile rebind is active",
    )
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
        _require(record is not None, f"manifest input missing: {relative}")
        _require(record.get("status") == "excluded_from_git", f"bad external status: {relative}")
        _require(
            int(record.get("bytes", -1)) == int(row.bytes)
            and str(record.get("sha256", "")) == str(row.sha256),
            f"external input record drift: {relative}",
        )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True
        )
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=ROOT, capture_output=True
        )
        _require(tracked.returncode != 0, f"external input is tracked: {relative}")
        _require(ignored.returncode == 0, f"external input is not ignored: {relative}")
    return {"v25_input_manifest_files": len(manifest)}


def _audit_predictions() -> dict[str, object]:
    path = ACTIVE_DIR / "nested_predictions.csv"
    _require(_sha256(path) == V25_SHA256, "V25 canonical prediction drift")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    _require(len(frame) == 232, "V25 prediction row count is not 232")
    _require(set(frame["election_id"].astype(str)) == set(ORDER), "V25 election panel drift")
    _require(not frame["election_id"].astype(str).str.contains("2025").any(), "2025 row in V25 history")
    totals = frame.groupby(["election_id", "region_id"])["layer_pred"].sum()
    _require(np.allclose(totals, 1.0, atol=1e-12), "V25 predictions are not compositional")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    _require(summary["metrics"]["rows"] == 232, "V25 summary row drift")
    _require(np.isclose(summary["metrics"]["winner_accuracy"], 0.8), "V25 winner gate drift")
    refusal = pd.read_csv(ACTIVE_DIR / "weak_same_lane_refusal_audit.csv", encoding="utf-8-sig")
    _require(set(refusal["recipient_weight_mode"].astype(str)) == {"prediction_tilted"}, "weak-C route drift")
    return {"v25_prediction_rows": 232, "v25_prediction_sha256": V25_SHA256}


def _audit_intervals() -> dict[str, object]:
    manifest = json.loads((ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8"))
    _require(manifest["model_version"] == "v25", "interval model version drift")
    _require(manifest["input_sha256"] == V25_SHA256, "interval input hash drift")
    _require(manifest["post_2022_outcomes_used"] is False, "post-2022 interval outcome used")
    _require(manifest["target_outcomes_used_to_construct_bounds"] is False, "target built interval")
    detail = pd.read_csv(ACTIVE_DIR / "national_predictive_intervals.csv")
    _require(len(detail) == 44, "interval row count is not 44")
    _require(set(detail["nominal_level"].round(2)) == {0.5, 0.8, 0.9, 0.95}, "interval levels drift")
    position = {election: index for index, election in enumerate(ORDER)}
    for row in detail.itertuples(index=False):
        _require(
            all(position[e] < position[row.election_id] for e in str(row.training_elections).split("|")),
            f"nonchronological interval training: {row.election_id}",
        )
    return {"interval_candidate_level_rows": len(detail)}


def _audit_finalization() -> dict[str, object]:
    manifest = json.loads((ACTIVE_DIR / "finalization_manifest.json").read_text(encoding="utf-8"))
    _require(manifest["active_version"] == "v25", "finalization version drift")
    _require(manifest["post_2022_outcomes_used"] is False, "finalization outcome boundary drift")
    for record in manifest["artifacts"]:
        path = ROOT / record["path"]
        _require(path.is_file(), f"finalized artifact missing: {record['path']}")
        _require(
            _artifact_sha256(path, str(record.get("hash_mode", "raw_bytes"))) == record["sha256"],
            f"finalized artifact drift: {record['path']}",
        )
    return {"finalized_artifacts": len(manifest["artifacts"])}


def main() -> None:
    results: dict[str, object] = {}
    for audit in (_audit_rollbacks, _audit_pointer, _audit_inputs, _audit_predictions, _audit_intervals, _audit_finalization):
        results.update(audit())
    print("[active V25 public audit: PASS]")
    for key, value in results.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
