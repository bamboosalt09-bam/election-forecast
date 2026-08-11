"""Audit the promoted V23 automatic-control inputs and active reproduction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import audit_point_in_time as pit  # noqa: E402
from presidential_issue_engine import issue_vote_engine as engine  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402


AUTOMATIC_DIR = ROOT / "outputs" / "automatic_controls_v23"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v23"
EXPERIMENT_DIR = (
    ROOT
    / "outputs"
    / "automatic_controls_v23_ablation_v3"
    / "v23_unified_profile_transfer_generation"
    / "active_run"
)
REGISTRY = AUTOMATIC_DIR / "withdrawal_transfer_registry.csv"
GENERATION = AUTOMATIC_DIR / "election_generation_weights.csv"
LEGACY_TRANSFER_INPUTS = {
    "data/raw/withdrawn_candidate_transfers.csv",
    "data/raw/withdrawal_event_profiles.csv",
    "presidential_issue_engine/fixed_dataset/coalition_events.csv",
}
EXTERNAL_INPUTS = ROOT / "data" / "raw" / "official_sources" / "external_active_inputs.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_active_manifest() -> dict[str, int]:
    manifest = pd.read_csv(ACTIVE_DIR / "input_manifest.csv", encoding="utf-8-sig")
    external_payload = json.loads(EXTERNAL_INPUTS.read_text(encoding="utf-8"))
    external_inputs = {
        str(row["path"]).replace("\\", "/"): row
        for row in external_payload.get("inputs", [])
    }
    paths = manifest["path"].astype(str).str.replace("\\", "/", regex=False)
    _require(not (set(paths) & LEGACY_TRANSFER_INPUTS), "legacy transfer input is active")
    required = {
        "outputs/automatic_controls_v23/withdrawal_transfer_registry.csv",
        "outputs/automatic_controls_v23/election_generation_weights.csv",
        "outputs/automatic_controls_v23/candidate_political_profiles.csv",
        "data/raw/withdrawal_events.csv",
        "data/raw/official_sources/assembly_candidate_attention_history.csv",
        "data/raw/official_sources/nec_age_turnout_composition_history.csv",
    }
    _require(required.issubset(set(paths)), "V23 manifest is missing required inputs")
    for row in manifest.itertuples(index=False):
        path_text = str(row.path).replace("\\", "/")
        if path_text.startswith("generated:"):
            continue
        path = ROOT / path_text
        if not path.exists():
            external = external_inputs.get(path_text)
            _require(external is not None, f"manifest input is missing: {path_text}")
            _require(
                external.get("status") == "excluded_from_git",
                f"external input has an invalid status: {path_text}",
            )
            _require(
                int(external.get("bytes", -1)) == int(row.bytes),
                f"external input byte count drift: {path_text}",
            )
            _require(
                str(external.get("sha256", "")) == str(row.sha256),
                f"external input hash drift: {path_text}",
            )
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", path_text],
                cwd=ROOT,
                capture_output=True,
            )
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", path_text],
                cwd=ROOT,
                capture_output=True,
            )
            _require(tracked.returncode != 0, f"external input is tracked: {path_text}")
            _require(ignored.returncode == 0, f"external input is not ignored: {path_text}")
            continue
        _require(_sha256(path) == str(row.sha256), f"manifest hash drift: {path_text}")
    return {"active_manifest_files": len(manifest)}


def _audit_automatic_outputs() -> dict[str, int]:
    names = [
        "candidate_political_profiles.csv",
        "candidate_political_landscape.csv",
        "third_candidate_profile.csv",
        "third_candidate_pressure.csv",
        "withdrawal_transfer_registry.csv",
        "election_generation_weights.csv",
    ]
    rows = 0
    for name in names:
        frame = pd.read_csv(AUTOMATIC_DIR / name, encoding="utf-8-sig")
        rows += len(frame)
        if "election_id" in frame:
            _require(
                not frame["election_id"].astype(str).str.contains("2025").any(),
                f"2025 row found in {name}",
            )
        if "target_outcome_used" in frame:
            used = frame["target_outcome_used"].astype(str).str.lower()
            _require(
                not used.isin({"1", "true", "yes", "y"}).any(),
                f"target outcome used in {name}",
            )
        if {"election_id", "available_date"}.issubset(frame.columns):
            available = pd.to_datetime(frame["available_date"], errors="coerce")
            cutoff = pd.to_datetime(
                frame["election_id"].map(engine.ELECTION_DATES), errors="coerce"
            )
            _require(available.notna().all(), f"invalid available_date in {name}")
            _require(cutoff.notna().all(), f"unknown election cutoff in {name}")
            _require(available.le(cutoff).all(), f"post-cutoff row in {name}")

    generation = pd.read_csv(GENERATION, encoding="utf-8-sig")
    generation_date = pd.to_datetime(generation["available_date"], errors="raise")
    election_date = pd.to_datetime(
        generation["election_id"].map(engine.ELECTION_DATES), errors="raise"
    )
    _require(generation_date.lt(election_date).all(), "generation report is not strictly prior")
    weight_sum = generation[["young_weight", "middle_weight", "senior_weight"]].sum(axis=1)
    _require(np.allclose(weight_sum, 1.0, atol=1e-12), "generation weights do not sum to one")
    return {"automatic_rows_checked": rows}


def _audit_attention_cutoff() -> dict[str, int]:
    attention = pd.read_csv(
        ROOT
        / "data"
        / "raw"
        / "official_sources"
        / "assembly_candidate_attention_history.csv",
        encoding="utf-8-sig",
    )
    events = pd.read_csv(
        ROOT / "data" / "raw" / "withdrawal_events.csv", encoding="utf-8-sig"
    )
    merged = attention.merge(
        events[["election_id", "candidate_id", "event_timestamp"]]
    )
    _require(not merged.empty, "candidate attention evidence did not match an event")
    last_evidence = pd.to_datetime(merged["last_evidence_date"], errors="raise")
    event_date = pd.to_datetime(merged["event_timestamp"], errors="raise", utc=True).dt.tz_convert(None)
    _require(last_evidence.lt(event_date).all(), "attention evidence reaches event or future date")
    return {"attention_rows_checked": len(merged)}


def _audit_reproduction() -> dict[str, int]:
    _require(
        _sha256(ACTIVE_DIR / "nested_predictions.csv")
        == _sha256(EXPERIMENT_DIR / "nested_predictions.csv"),
        "active V23 does not reproduce the gated experiment",
    )
    active_metrics = json.loads(
        (ACTIVE_DIR / "summary.json").read_text(encoding="utf-8")
    )["metrics"]
    experiment_metrics = json.loads(
        (EXPERIMENT_DIR / "summary.json").read_text(encoding="utf-8")
    )["metrics"]
    _require(active_metrics == experiment_metrics, "active V23 metrics differ from experiment")
    return {"active_prediction_rows": int(active_metrics["rows"])}


def _audit_outcome_invariance() -> dict[str, int]:
    attributes = [
        (engine, "WITHDRAWAL_TRANSFER_REGISTRY", str(REGISTRY)),
        (engine, "ELECTION_GENERATION_WEIGHTS", str(GENERATION)),
        (
            engine,
            "CANDIDATE_POLITICAL_LANDSCAPE",
            str(AUTOMATIC_DIR / "candidate_political_landscape.csv"),
        ),
        (
            engine,
            "ENHANCED_MEGA_ISSUE_INTENSITY",
            str(AUTOMATIC_DIR / "mega_issue_intensity.csv"),
        ),
        (engine, "MEGA_ISSUE_TAXONOMY", str(AUTOMATIC_DIR / "mega_issue_taxonomy.csv")),
        (engine, "ECONOMIC_SLOT_ALIGNMENT", str(AUTOMATIC_DIR / "economic_slot_alignment.csv")),
        (engine, "HOUSING_SLOT_ALIGNMENT", str(AUTOMATIC_DIR / "housing_slot_alignment.csv")),
    ]
    with patching.patched(attributes):
        return pit.audit_target_outcome_invariance()


def main() -> None:
    results: dict[str, int] = {}
    for audit in (
        _audit_active_manifest,
        _audit_automatic_outputs,
        _audit_attention_cutoff,
        _audit_reproduction,
        _audit_outcome_invariance,
    ):
        results.update(audit())
    print("[active V23 audit: PASS]")
    for key, value in results.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
