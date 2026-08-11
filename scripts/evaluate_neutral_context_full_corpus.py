"""Evaluate the fixed neutral-context configuration on the full stance corpus.

This is a read-only, non-PIT shadow experiment. The full extractor labels
availability with a meeting-date proxy and retrospectively maps candidate
identities, so this output must not be promoted into the active engine without
a separate point-in-time eligibility policy.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "presidential_issue_engine"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ENGINE_DIR))

import issue_vote_engine as engine  # noqa: E402
from scripts.evaluate_neutral_context_protocols import (  # noqa: E402
    ELECTIONS,
    SHADOW_SCALE,
    full_fit_rows,
    loeo_rows,
    national_points,
    national_summary,
    rolling_rows,
    row_summary,
)
from scripts.evaluate_stance_pilot_3000_sensitivity import CONFIGS, build_features  # noqa: E402


CONFIG_NAME = "person_party_speaker_confirmed_conf3_context050_issueglobal025_gate2"
FULL_INPUT = ROOT / "outputs" / "assembly_stance" / "full_15_22" / "assembly_stance_rows_15_22.csv"
EXTRACTION_STATE = ROOT / "outputs" / "assembly_stance" / "full_15_22" / "state.json"
SAMPLE_INPUT = ROOT / "data" / "raw" / "assembly_neutral_issue_context.csv"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "full_corpus_neutral_context_experiment"
PROGRESS_EVERY = 250_000


def _corpus_fingerprint(state: dict[str, object]) -> str:
    completed = state.get("completed", {})
    if not isinstance(completed, dict):
        raise ValueError("extraction state lacks completed part metadata")
    digest = hashlib.sha256()
    for source_id, metadata in sorted(completed.items()):
        if not isinstance(metadata, dict):
            continue
        digest.update(str(source_id).encode("utf-8"))
        digest.update(str(metadata.get("part_sha256", "")).encode("ascii", "ignore"))
        digest.update(str(metadata.get("row_count", "")).encode("ascii", "ignore"))
    return digest.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def _write_json_atomic(payload: dict[str, object], path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)


def _validate_extraction_state() -> dict[str, object]:
    if not FULL_INPUT.exists() or not EXTRACTION_STATE.exists():
        raise FileNotFoundError("full stance corpus or extraction state is missing")
    state = json.loads(EXTRACTION_STATE.read_text(encoding="utf-8"))
    completed = state.get("completed", {})
    if not isinstance(completed, dict) or not completed:
        raise ValueError("extraction state has no completed sources")
    row_count = sum(int(value.get("row_count", 0)) for value in completed.values())
    if row_count <= 0:
        raise ValueError("extraction state has no completed rows")
    return {
        "source_parts": len(completed),
        "source_rows": row_count,
        "source_bytes": FULL_INPUT.stat().st_size,
        "source_mtime_ns": FULL_INPUT.stat().st_mtime_ns,
        "corpus_fingerprint": _corpus_fingerprint(state),
    }


def _signal_table(features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "stance_shadow_signal",
        "evidence_count",
        "context_neutral_count",
        "context_issue_overlap_count",
        "global_context_neutral_count",
        "global_context_issue_overlap_count",
        "global_context_structure_strength",
        "global_context_content_strength",
        "global_context_strength",
        "global_context_relative_strength",
        "coverage_gate_passed",
    ]
    out = features.loc[features["election_id"].isin(ELECTIONS), columns].copy()
    if out.duplicated(["election_id", "slot"]).any():
        raise RuntimeError("full-corpus features contain duplicate election-slot rows")
    return out.sort_values(["election_id", "slot"]).reset_index(drop=True)


def _sample_comparison(full: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_INPUT)[
        ["election_id", "slot", "candidate_name", "assembly_neutral_issue_signal"]
    ].rename(columns={"assembly_neutral_issue_signal": "sample_signal"})
    out = full.rename(columns={"stance_shadow_signal": "full_signal"}).merge(
        sample,
        on=["election_id", "slot", "candidate_name"],
        how="left",
    )
    out["direction_agrees"] = (
        np.sign(out["full_signal"].fillna(0.0)) == np.sign(out["sample_signal"].fillna(0.0))
    )
    out["signal_difference"] = out["full_signal"] - out["sample_signal"]
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = _validate_extraction_state()
    run_state = {
        **source,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "config": CONFIG_NAME,
        "shadow_scale": SHADOW_SCALE,
        "point_in_time_eligible": False,
    }
    _write_json_atomic(run_state, OUTPUT_DIR / "run_state.json")
    try:
        config = next(value for value in CONFIGS if value["name"] == CONFIG_NAME)
        features = build_features(
            config,
            pilot_input=FULL_INPUT,
            progress_every=PROGRESS_EVERY,
        )
        signals = _signal_table(features)
        assembled = engine.assemble()
        frame = assembled.loc[assembled["election_id"].isin(ELECTIONS)].copy()
        protocol_rows = pd.concat(
            [
                full_fit_rows(frame, signals),
                loeo_rows(frame, signals),
                rolling_rows(frame, signals),
            ],
            ignore_index=True,
        )
        row_metrics = row_summary(protocol_rows)
        points = national_points(protocol_rows)
        national_metrics = national_summary(points)
        comparison = _sample_comparison(signals)

        _write_csv_atomic(signals, OUTPUT_DIR / "candidate_features.csv")
        _write_csv_atomic(comparison, OUTPUT_DIR / "sample_vs_full_signals.csv")
        _write_csv_atomic(protocol_rows, OUTPUT_DIR / "protocol_row_predictions.csv")
        _write_csv_atomic(row_metrics, OUTPUT_DIR / "protocol_row_summary.csv")
        _write_csv_atomic(points, OUTPUT_DIR / "protocol_national_points.csv")
        _write_csv_atomic(national_metrics, OUTPUT_DIR / "protocol_national_summary.csv")

        run_state.update(
            {
                "status": "complete",
                "completed_at": datetime.now().astimezone().isoformat(),
                "feature_rows": len(signals),
                "protocol_rows": len(protocol_rows),
                "signal_direction_agreement": float(comparison["direction_agrees"].mean()),
            }
        )
        _write_json_atomic(run_state, OUTPUT_DIR / "run_state.json")
        print(row_metrics.to_string(index=False))
        print()
        print(national_metrics.to_string(index=False))
        print()
        print(comparison.to_string(index=False))
    except BaseException as exc:
        run_state.update(
            {
                "status": "failed",
                "failed_at": datetime.now().astimezone().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json_atomic(run_state, OUTPUT_DIR / "run_state.json")
        raise


if __name__ == "__main__":
    main()
