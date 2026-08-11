"""Collect and compile official NEC candidate history through 2022.

The integrated name search may return post-cutoff records. They are discarded
in memory before any checkpoint is written, so the persisted dataset and all
derived features remain bounded by ``--max-source-date``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from news_collector.sources.nec_api import (  # noqa: E402
    CANDIDATE_SEARCH_BASE,
    search_candidate,
)
from presidential_issue_engine.official_candidate_history import (  # noqa: E402
    build_candidate_reference,
    build_official_candidate_regional_base,
    resolve_candidate_history,
)


DEFAULT_RESULTS = (
    ROOT / "presidential_issue_engine" / "fixed_dataset" / "presidential_results_standardized.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "official_sources"
DEFAULT_MAX_SOURCE_DATE = "2022-12-31"


def _record_date(record: dict[str, Any]) -> pd.Timestamp | None:
    text = "".join(character for character in str(record.get("sgId", "")) if character.isdigit())
    if len(text) != 8:
        return None
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _safe_slug(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def collect_records_by_name(
    names: list[str],
    *,
    checkpoint_dir: Path,
    max_source_date: pd.Timestamp,
    offline: bool,
    refresh: bool,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Collect sanitized per-name checkpoints for interruption-safe resumes."""

    records_by_name: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    for name in names:
        checkpoint = checkpoint_dir / f"{_safe_slug(name)}.json"
        if checkpoint.exists() and not refresh:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("candidate_name") != name:
                raise ValueError(f"candidate checkpoint identity mismatch: {checkpoint}")
            records = list(saved.get("records", []))
            records_by_name[name] = records
            audit.append(
                {
                    "candidate_name": name,
                    "source": "sanitized_checkpoint",
                    "retained_record_count": len(records),
                    "post_cutoff_records_discarded": int(
                        saved.get("post_cutoff_records_discarded", 0)
                    ),
                }
            )
            continue
        if offline:
            raise FileNotFoundError(f"missing sanitized candidate checkpoint: {checkpoint}")
        raw_records = search_candidate(name, cache_dir=None, refresh=refresh)
        retained: list[dict[str, Any]] = []
        discarded = 0
        for record in raw_records:
            source_date = _record_date(record)
            if source_date is not None and source_date <= max_source_date:
                retained.append(record)
            else:
                discarded += 1
        _atomic_json(
            checkpoint,
            {
                "candidate_name": name,
                "source_url": f"{CANDIDATE_SEARCH_BASE}/getCndaSrchInqire",
                "max_source_date": max_source_date.date().isoformat(),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "post_cutoff_records_discarded": discarded,
                "records": retained,
                "checkpoint_schema": "nec_candidate_search_sanitized_v1",
            },
        )
        records_by_name[name] = retained
        audit.append(
            {
                "candidate_name": name,
                "source": "live_api",
                "retained_record_count": len(retained),
                "post_cutoff_records_discarded": discarded,
            }
        )
    return records_by_name, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-source-date", default=DEFAULT_MAX_SOURCE_DATE)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    maximum = pd.Timestamp(args.max_source_date)
    if maximum > pd.Timestamp(DEFAULT_MAX_SOURCE_DATE):
        raise ValueError(
            "this project is frozen at 2022; --max-source-date must not exceed 2022-12-31"
        )
    results = pd.read_csv(
        args.results,
        encoding="utf-8",
        usecols=["election_id", "slot", "candidate_name", "party_name"],
    )
    reference = build_candidate_reference(results)
    reference = reference.loc[
        pd.to_datetime(reference["target_election_date"]).le(maximum)
    ].copy()
    names = sorted(reference["candidate_name"].astype(str).unique())
    checkpoint_dir = args.output_dir / "checkpoints" / "nec_candidate_search"
    records_by_name, collection_audit = collect_records_by_name(
        names,
        checkpoint_dir=checkpoint_dir,
        max_source_date=maximum,
        offline=args.offline,
        refresh=args.refresh,
    )
    source_url = f"{CANDIDATE_SEARCH_BASE}/getCndaSrchInqire"
    history, resolution_audit = resolve_candidate_history(
        reference,
        records_by_name,
        source_url=source_url,
        max_source_date=maximum,
    )
    regional_base = build_official_candidate_regional_base(history)

    _atomic_csv(args.output_dir / "presidential_candidate_reference.csv", reference)
    _atomic_csv(args.output_dir / "nec_candidate_history.csv", history)
    _atomic_csv(args.output_dir / "nec_candidate_resolution_audit.csv", resolution_audit)
    _atomic_csv(
        args.output_dir / "automatic_candidate_regional_base_official.csv",
        regional_base,
    )
    _atomic_csv(
        args.output_dir / "nec_candidate_collection_audit.csv",
        pd.DataFrame(collection_audit),
    )
    manifest = {
        "schema": "official_candidate_automation_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_source_date": maximum.date().isoformat(),
        "post_2022_records_persisted": False,
        "target_outcome_columns_read": [],
        "candidate_reference_columns": [
            "election_id",
            "slot",
            "candidate_name",
            "party_name",
        ],
        "source_url": source_url,
        "rows": {
            "candidate_reference": len(reference),
            "candidate_history": len(history),
            "candidate_resolution_audit": len(resolution_audit),
            "automatic_candidate_regional_base": len(regional_base),
        },
        "active_model_promoted": False,
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
