"""Build forecast-only 2025 Assembly context from the verified full corpus.

The script streams the sentence corpus, retains every ``pres_2025`` row
available by 2025-06-02, and emits only sufficient statistics. It never reads
presidential results or evaluates forecast error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine.election_scope import (
    ELECTION_DATES,
    FORECAST_ONLY_ELECTIONS,
    SCORED_ELECTIONS,
)
from presidential_issue_engine.point_in_time import forecast_cutoff


TARGET_ELECTION = "pres_2025"
DEFAULT_OUTPUT_DIR = ROOT / "data/raw/official_sources/assembly_pres_2025_context"
DEFAULT_CANDIDATE_REGISTRY = (
    ROOT / "data/raw/official_sources/pres_2025_candidate_registry.csv"
)
DEFAULT_STATE = "state.json"
REQUIRED_COLUMNS = {
    "election_id",
    "assembly_daesu",
    "source_id",
    "source_file",
    "meeting_date",
    "available_date",
    "period",
    "issue_name",
    "issue_weight",
    "target_type",
    "target_name",
    "target_model_eligible",
    "stance_label",
    "stance_polarity",
    "stance_confidence",
    "text_sha256",
}
FORBIDDEN_OUTCOME_COLUMNS = {
    "actual_vote_share",
    "candidate_votes",
    "error",
    "mae",
    "mean_vote_share",
    "pred",
    "vote_share",
    "votes",
    "winner",
}
CANDIDATE_REGISTRY_REQUIRED_COLUMNS = {
    "election_id",
    "candidate_id",
    "candidate_name",
    "party_name",
    "ballot_number",
    "available_date",
    "source_url",
    "source_type",
}


@dataclass
class Aggregate:
    rows: int = 0
    issue_weight: float = 0.0
    signed_weight: float = 0.0
    absolute_directional_weight: float = 0.0
    first_date: str = ""
    last_date: str = ""

    def add(
        self,
        observed: str,
        weight: float,
        polarity: float,
        confidence: float,
    ) -> None:
        self.rows += 1
        self.issue_weight += weight
        directional = weight * polarity * confidence
        self.signed_weight += directional
        self.absolute_directional_weight += abs(directional)
        self.first_date = observed if not self.first_date or observed < self.first_date else self.first_date
        self.last_date = observed if not self.last_date or observed > self.last_date else self.last_date


def _number(value: object, *, default: float = 0.0) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric input: {value!r}")
    return number


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _atomic_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    """Return a portable source identifier without publishing local paths."""

    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return f"external://{resolved.name}"
    return f"repo://{relative.as_posix()}"


def _source_fingerprint(source: Path) -> tuple[str, str]:
    state = source.parent / DEFAULT_STATE
    if state.exists():
        payload = json.loads(state.read_text(encoding="utf-8"))
        if payload.get("final_valid") is True and payload.get("final_sha256"):
            return str(payload["final_sha256"]), f"verified sibling {DEFAULT_STATE}"
    return _sha256(source), "computed from source bytes"


def _load_candidate_registry(path: Path, cutoff_date: date) -> list[dict[str, str]]:
    """Load the full official roster without using withdrawals or outcomes."""

    frame = pd.read_csv(path, dtype="string", encoding="utf-8-sig").fillna("")
    missing = sorted(CANDIDATE_REGISTRY_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"candidate registry is missing columns: {missing}")
    forbidden = sorted(set(frame.columns) & FORBIDDEN_OUTCOME_COLUMNS)
    if forbidden:
        raise ValueError(f"candidate registry contains outcome columns: {forbidden}")
    if set(frame["election_id"].astype(str)) != {TARGET_ELECTION}:
        raise ValueError("candidate registry contains elections other than pres_2025")

    parsed_dates = pd.to_datetime(frame["available_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("candidate registry has invalid available_date values")
    if (parsed_dates.dt.date > cutoff_date).any():
        raise ValueError("candidate registry contains post-cutoff rows")
    if frame["candidate_id"].duplicated().any() or frame["candidate_name"].duplicated().any():
        raise ValueError("candidate registry has duplicate candidate identities")

    rows = frame.to_dict("records")
    if not rows:
        raise ValueError("candidate registry is empty")
    return [{str(key): str(value) for key, value in row.items()} for row in rows]


def _registered_candidate_links(
    target_type: str,
    target_name: str,
    registry: list[dict[str, str]],
) -> list[tuple[dict[str, str], str]]:
    """Resolve explicit person/party targets against the complete roster."""

    if target_type == "person":
        return [
            (candidate, "registered_candidate_person")
            for candidate in registry
            if candidate["candidate_name"] == target_name
        ]
    if target_type == "party" and target_name != "무소속":
        return [
            (candidate, "registered_candidate_party")
            for candidate in registry
            if candidate["party_name"] == target_name
        ]
    return []


def _normalize_sources(source: Path | Iterable[Path]) -> list[Path]:
    if isinstance(source, Path):
        paths = [source]
    else:
        paths = [Path(path) for path in source]
    if not paths:
        raise ValueError("at least one Assembly source is required")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Assembly source not found: {path}")
    return paths


def _iter_validated_rows(source: Path) -> Iterable[dict[str, str]]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Assembly corpus is missing columns: {missing}")
        forbidden = sorted(columns & FORBIDDEN_OUTCOME_COLUMNS)
        if forbidden:
            raise ValueError(f"Assembly corpus contains outcome columns: {forbidden}")
        yield from reader


def build_context(
    source: Path | Iterable[Path],
    output_dir: Path,
    candidate_registry: Path = DEFAULT_CANDIDATE_REGISTRY,
) -> dict[str, object]:
    """Stream the full corpus and build PIT-safe 2025 sufficient statistics."""

    if TARGET_ELECTION not in FORECAST_ONLY_ELECTIONS:
        raise RuntimeError("2025 is not registered as forecast-only")
    if TARGET_ELECTION in SCORED_ELECTIONS:
        raise RuntimeError("2025 entered the scored election set")
    cutoff = forecast_cutoff(TARGET_ELECTION, ELECTION_DATES)
    if cutoff is None:
        raise RuntimeError("2025 forecast cutoff is unavailable")
    cutoff_date = cutoff.date()
    registry = _load_candidate_registry(candidate_registry, cutoff_date)
    source_paths = _normalize_sources(source)

    issue_groups: dict[tuple[str, str, str], Aggregate] = defaultdict(Aggregate)
    target_groups: dict[tuple[str, str, str, str, str, bool, str], Aggregate] = defaultdict(Aggregate)
    scanned = 0
    target_rows = 0
    excluded_post_cutoff = 0
    invalid_dates = 0
    duplicate_rows_excluded = 0
    source_min_date = ""
    source_max_date = ""
    min_date = ""
    max_date = ""
    assemblies: dict[str, int] = defaultdict(int)
    seen_rows: set[tuple[str, str, str, str, str]] = set()

    for source_path in source_paths:
        for row in _iter_validated_rows(source_path):
            scanned += 1
            if str(row.get("election_id", "")).strip() != TARGET_ELECTION:
                continue
            target_rows += 1
            meeting_text = str(row.get("meeting_date", "")).strip()
            available_text = str(row.get("available_date", "")).strip()
            try:
                meeting = date.fromisoformat(meeting_text)
                available = date.fromisoformat(available_text)
            except ValueError:
                invalid_dates += 1
                continue
            source_min_date = (
                meeting_text
                if not source_min_date or meeting_text < source_min_date
                else source_min_date
            )
            source_max_date = (
                meeting_text
                if not source_max_date or meeting_text > source_max_date
                else source_max_date
            )
            if available < meeting:
                raise ValueError(
                    f"availability predates meeting: {meeting_text} -> {available_text}"
                )
            if available > cutoff_date:
                excluded_post_cutoff += 1
                continue

            row_key = (
                str(row.get("source_id", "")).strip(),
                str(row.get("source_row_id", "")).strip(),
                str(row.get("sentence_index", "")).strip(),
                str(row.get("issue_name", "")).strip(),
                str(row.get("text_sha256", "")).strip(),
            )
            if row_key in seen_rows:
                duplicate_rows_excluded += 1
                continue
            seen_rows.add(row_key)

            issue_name = str(row.get("issue_name", "")).strip()
            period = str(row.get("period", "")).strip()
            assembly = str(row.get("assembly_daesu", "")).strip()
            if not issue_name or not period or not assembly:
                raise ValueError("2025 Assembly row lacks issue, period, or Assembly term")
            weight = max(_number(row.get("issue_weight")), 0.0)
            polarity = max(min(_number(row.get("stance_polarity")), 1.0), -1.0)
            confidence = max(min(_number(row.get("stance_confidence")), 1.0), 0.0)
            assemblies[assembly] += 1
            min_date = meeting_text if not min_date or meeting_text < min_date else min_date
            max_date = meeting_text if not max_date or meeting_text > max_date else max_date

            issue_groups[(assembly, period, issue_name)].add(
                meeting_text, weight, polarity, confidence
            )
            target_type = str(row.get("target_type", "")).strip()
            target_name = str(row.get("target_name", "")).strip()
            target_model_eligible = _truthy(row.get("target_model_eligible"))
            stance_label = str(row.get("stance_label", "")).strip() or "neutral"
            if target_type and target_name:
                target_groups[
                    (
                        assembly,
                        period,
                        issue_name,
                        target_type,
                        target_name,
                        target_model_eligible,
                        stance_label,
                    )
                ].add(meeting_text, weight, polarity, confidence)

    if invalid_dates:
        raise ValueError(f"2025 Assembly corpus has {invalid_dates} invalid dated rows")
    if not target_rows:
        raise ValueError("Assembly corpus contains no pres_2025 rows")

    issue_rows: list[dict[str, object]] = []
    for (assembly, period, issue_name), value in sorted(issue_groups.items()):
        issue_rows.append(
            {
                "election_id": TARGET_ELECTION,
                "assembly_daesu": assembly,
                "period": period,
                "issue_name": issue_name,
                "sentence_count": value.rows,
                "weighted_mentions": value.issue_weight,
                "first_observed_date": value.first_date,
                "last_observed_date": value.last_date,
                "available_date": value.last_date,
                "source": "assembly_stance_full_corpus_v1",
            }
        )

    target_context_rows: list[dict[str, object]] = []
    candidate_context_rows: list[dict[str, object]] = []
    for key, value in sorted(target_groups.items()):
        (
            assembly,
            period,
            issue_name,
            target_type,
            target_name,
            target_model_eligible,
            stance_label,
        ) = key
        balance = (
            value.signed_weight / value.absolute_directional_weight
            if value.absolute_directional_weight > 0.0
            else 0.0
        )
        target_row = {
                "election_id": TARGET_ELECTION,
                "assembly_daesu": assembly,
                "period": period,
                "issue_name": issue_name,
                "target_type": target_type,
                "target_name": target_name,
                "target_model_eligible": target_model_eligible,
                "stance_label": stance_label,
                "sentence_count": value.rows,
                "weighted_mentions": value.issue_weight,
                "signed_weight": value.signed_weight,
                "absolute_directional_weight": value.absolute_directional_weight,
                "directional_balance": balance,
                "first_observed_date": value.first_date,
                "last_observed_date": value.last_date,
                "available_date": value.last_date,
                "source": "assembly_stance_full_corpus_v1",
            }
        target_context_rows.append(target_row)
        for candidate, linkage_basis in _registered_candidate_links(
            target_type, target_name, registry
        ):
            link_available = max(value.last_date, candidate["available_date"])
            candidate_context_rows.append(
                {
                    **target_row,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_name": candidate["candidate_name"],
                    "candidate_party_name": candidate["party_name"],
                    "candidate_ballot_number": candidate["ballot_number"],
                    "candidate_registry_available_date": candidate["available_date"],
                    "candidate_link_eligible": True,
                    "candidate_linkage_basis": linkage_basis,
                    "source_target_type": target_type,
                    "source_target_name": target_name,
                    "source_observed_available_date": value.last_date,
                    "available_date": link_available,
                }
            )

    issue_path = output_dir / "issue_salience_weekly.csv"
    target_path = output_dir / "explicit_target_context_weekly.csv"
    candidate_path = output_dir / "candidate_target_context_weekly.csv"
    model_salience_path = output_dir / "model_issue_salience.csv"
    model_link_path = output_dir / "model_candidate_issue_link.csv"
    _atomic_csv(issue_rows, issue_path)
    _atomic_csv(target_context_rows, target_path)
    _atomic_csv(candidate_context_rows, candidate_path)

    model_salience = pd.DataFrame(issue_rows)
    model_salience = (
        model_salience.groupby(
            ["election_id", "issue_name", "period"], as_index=False
        )
        .agg(
            raw_value=("weighted_mentions", "sum"),
            available_date=("available_date", "max"),
        )
    )
    salience_peak = float(model_salience["raw_value"].max()) if len(model_salience) else 0.0
    model_salience["salience_score"] = (
        model_salience["raw_value"] / salience_peak if salience_peak > 0.0 else 0.0
    )
    model_salience["instrument"] = "assembly_speech_forecast_only"
    model_salience = model_salience[
        [
            "election_id",
            "issue_name",
            "period",
            "raw_value",
            "salience_score",
            "instrument",
            "available_date",
        ]
    ]
    _atomic_csv(model_salience.to_dict("records"), model_salience_path)

    candidate_frame = pd.DataFrame(candidate_context_rows)
    if candidate_frame.empty:
        model_link = pd.DataFrame(
            columns=[
                "election_id",
                "candidate_id",
                "candidate_name",
                "party_name",
                "issue_name",
                "mentions",
                "emphasis_volume",
                "emphasis_within",
                "signed_weight",
                "absolute_directional_weight",
                "directional_balance",
                "available_date",
                "linkage_basis",
            ]
        )
    else:
        model_link = (
            candidate_frame.groupby(
                [
                    "election_id",
                    "candidate_id",
                    "candidate_name",
                    "candidate_party_name",
                    "issue_name",
                ],
                as_index=False,
            )
            .agg(
                mentions=("weighted_mentions", "sum"),
                signed_weight=("signed_weight", "sum"),
                absolute_directional_weight=("absolute_directional_weight", "sum"),
                available_date=("available_date", "max"),
                linkage_basis=("candidate_linkage_basis", lambda values: "|".join(sorted(set(values)))),
            )
            .rename(columns={"candidate_party_name": "party_name"})
        )
        election_peak = model_link.groupby("election_id")["mentions"].transform("max")
        candidate_total = model_link.groupby(
            ["election_id", "candidate_id"]
        )["mentions"].transform("sum")
        model_link["emphasis_volume"] = model_link["mentions"] / election_peak.replace(0.0, pd.NA)
        model_link["emphasis_within"] = model_link["mentions"] / candidate_total.replace(0.0, pd.NA)
        model_link["directional_balance"] = (
            model_link["signed_weight"]
            / model_link["absolute_directional_weight"].replace(0.0, pd.NA)
        ).fillna(0.0)
        model_link[["emphasis_volume", "emphasis_within"]] = model_link[
            ["emphasis_volume", "emphasis_within"]
        ].fillna(0.0)
        model_link = model_link[
            [
                "election_id",
                "candidate_id",
                "candidate_name",
                "party_name",
                "issue_name",
                "mentions",
                "emphasis_volume",
                "emphasis_within",
                "signed_weight",
                "absolute_directional_weight",
                "directional_balance",
                "available_date",
                "linkage_basis",
            ]
        ]
    _atomic_csv(model_link.to_dict("records"), model_link_path)
    source_entries = []
    for source_path in source_paths:
        source_hash, source_hash_basis = _source_fingerprint(source_path)
        source_entries.append(
            {
                "path": _manifest_path(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": source_hash,
                "sha256_basis": source_hash_basis,
            }
        )
    combined_source_hash = hashlib.sha256(
        json.dumps(source_entries, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": "pres_2025_assembly_forecast_context_v1",
        "status": "forecast_only_not_scored",
        "target_election": TARGET_ELECTION,
        "election_date": ELECTION_DATES[TARGET_ELECTION],
        "forecast_cutoff": cutoff_date.isoformat(),
        "source_path": _manifest_path(source_paths[0]),
        "source_paths": [_manifest_path(path) for path in source_paths],
        "source_bytes": sum(path.stat().st_size for path in source_paths),
        "source_sha256": combined_source_hash,
        "source_sha256_basis": "sha256 of canonical source entry manifest",
        "sources": source_entries,
        "candidate_registry_path": _manifest_path(candidate_registry),
        "candidate_registry_sha256": _sha256(candidate_registry),
        "candidate_registry_rows": len(registry),
        "candidate_registry_policy": "complete_registered_roster_not_outcome_selected",
        "candidate_status_or_withdrawal_used": False,
        "source_rows_scanned": scanned,
        "target_rows_seen": target_rows,
        "target_rows_included": sum(item.rows for item in issue_groups.values()),
        "post_cutoff_rows_excluded": excluded_post_cutoff,
        "duplicate_rows_excluded": duplicate_rows_excluded,
        "invalid_date_rows": invalid_dates,
        "source_first_meeting_date": source_min_date,
        "source_last_meeting_date": source_max_date,
        "first_meeting_date": min_date,
        "last_meeting_date": max_date,
        "assembly_rows": dict(sorted(assemblies.items())),
        "outcome_columns_read": [],
        "pres_2025_outcome_used": False,
        "performance_metrics_computed": False,
        "scored_elections_unchanged": list(SCORED_ELECTIONS),
        "outputs": {
            issue_path.name: {
                "rows": len(issue_rows),
                "sha256": _sha256(issue_path),
            },
            target_path.name: {
                "rows": len(target_context_rows),
                "sha256": _sha256(target_path),
            },
            candidate_path.name: {
                "rows": len(candidate_context_rows),
                "sha256": _sha256(candidate_path),
            },
            model_salience_path.name: {
                "rows": len(model_salience),
                "sha256": _sha256(model_salience_path),
            },
            model_link_path.name: {
                "rows": len(model_link),
                "sha256": _sha256(model_link_path),
            },
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--supplement-source",
        type=Path,
        action="append",
        default=[],
        help="Additional compatible corpus; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=DEFAULT_CANDIDATE_REGISTRY,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [args.source.resolve(), *(path.resolve() for path in args.supplement_source)]
    manifest = build_context(
        sources,
        args.output_dir.resolve(),
        args.candidate_registry.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
