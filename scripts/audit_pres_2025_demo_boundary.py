"""Fail-fast audit for the outcome-free 2025 presidential demo boundary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import issue_vote_engine  # noqa: E402
from presidential_issue_engine.election_scope import (  # noqa: E402
    ELECTION_DATES,
    FORECAST_ONLY_ELECTIONS,
    SCORED_ELECTIONS,
)
from presidential_issue_engine.forecast_only_inputs import (  # noqa: E402
    DEFAULT_CONTEXT_DIR,
    FORBIDDEN_OUTCOME_COLUMNS,
    load_forecast_only_assembly_inputs,
)
from presidential_issue_engine.point_in_time import forecast_cutoff  # noqa: E402


REGISTRY = ROOT / "data/raw/official_sources/pres_2025_candidate_registry.csv"
MINUTES_DIR = ROOT / "data/raw/official_sources/assembly_pres_2025_minutes"
MINUTES_MANIFEST = MINUTES_DIR / "manifest.json"
MINUTES_MEETINGS = MINUTES_DIR / "meeting_manifest.csv"
MINUTES_ROWS = MINUTES_DIR / "assembly_stance_rows_2025_h1.csv"
RESULTS = ROOT / "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
ACTIVE_PREDICTIONS = ROOT / "outputs/active_presidential_nested_v23/nested_predictions.csv"
AUDIT_OUTPUT = ROOT / "docs/PRES_2025_DEMO_BOUNDARY_AUDIT.json"
EXPECTED_CANDIDATES = {
    "이재명",
    "김문수",
    "이준석",
    "권영국",
    "구주와",
    "황교안",
    "송진호",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cutoff = forecast_cutoff("pres_2025", ELECTION_DATES)
    _require(cutoff is not None, "2025 cutoff is missing")
    _require("pres_2025" in FORECAST_ONLY_ELECTIONS, "2025 is not forecast-only")
    _require("pres_2025" not in SCORED_ELECTIONS, "2025 entered scored elections")
    _require("pres_2025" not in issue_vote_engine.ORDER, "2025 entered engine ORDER")
    _require(
        "pres_2025" not in issue_vote_engine.WEIGHT_SELECTION_ELECTIONS,
        "2025 entered weight selection",
    )

    results = pd.read_csv(RESULTS, encoding="utf-8-sig")
    _require(
        not results["election_id"].astype(str).eq("pres_2025").any(),
        "2025 outcome exists in historical result input",
    )
    active = pd.read_csv(ACTIVE_PREDICTIONS, encoding="utf-8-sig")
    _require(
        not active["election_id"].astype(str).eq("pres_2025").any(),
        "2025 row exists in active historical predictions",
    )

    registry = pd.read_csv(REGISTRY, encoding="utf-8-sig", dtype="string")
    _require(
        set(registry["candidate_name"].astype(str)) == EXPECTED_CANDIDATES,
        "candidate registry is not the complete official seven-candidate roster",
    )
    _require(
        not (set(registry.columns) & FORBIDDEN_OUTCOME_COLUMNS),
        "candidate registry contains outcome fields",
    )
    registry_dates = pd.to_datetime(registry["available_date"], errors="raise")
    _require(registry_dates.le(cutoff).all(), "candidate registry is post-cutoff")

    salience, candidate_link, manifest = load_forecast_only_assembly_inputs()
    _require(
        pd.to_datetime(salience["available_date"], errors="raise").le(cutoff).all(),
        "salience contains post-cutoff rows",
    )
    _require(
        pd.to_datetime(candidate_link["available_date"], errors="raise").le(cutoff).all(),
        "candidate link contains post-cutoff rows",
    )
    _require(
        set(candidate_link["candidate_id"].astype(str)).issubset(
            set(registry["candidate_id"].astype(str))
        ),
        "candidate links contain identities outside the official roster",
    )

    minutes_manifest = json.loads(MINUTES_MANIFEST.read_text(encoding="utf-8"))
    _require(
        minutes_manifest.get("pres_2025_outcome_used") is False,
        "official-minute collection does not certify outcome exclusion",
    )
    _require(
        minutes_manifest.get("performance_metrics_computed") is False,
        "official-minute collection computed performance metrics",
    )
    meeting_manifest = pd.read_csv(MINUTES_MEETINGS, encoding="utf-8-sig")
    _require(
        len(meeting_manifest) == int(minutes_manifest["meetings_discovered"]),
        "official-minute meeting manifest is incomplete",
    )
    _require(
        meeting_manifest["minutes_id"].nunique() == len(meeting_manifest),
        "official-minute meeting IDs are duplicated",
    )
    meeting_dates = pd.to_datetime(meeting_manifest["meeting_date"], errors="raise")
    _require(meeting_dates.le(cutoff).all(), "post-cutoff meetings entered collection")
    _require(
        _sha256(MINUTES_MEETINGS)
        == minutes_manifest["outputs"]["meeting_manifest.csv"]["sha256"],
        "official-minute meeting manifest hash drift",
    )
    _require(
        _sha256(MINUTES_ROWS)
        == minutes_manifest["outputs"]["assembly_stance_rows_2025_h1.csv"]["sha256"],
        "official-minute derived corpus hash drift",
    )
    minute_rows = pd.read_csv(MINUTES_ROWS, encoding="utf-8-sig", low_memory=False)
    _require(
        not (set(minute_rows.columns) & FORBIDDEN_OUTCOME_COLUMNS),
        "official-minute derived corpus contains outcome fields",
    )
    minute_available = pd.to_datetime(minute_rows["available_date"], errors="raise")
    _require(
        int(minute_available.gt(cutoff).sum()) == int(manifest["post_cutoff_rows_excluded"]),
        "context builder exclusion count differs from official-minute corpus",
    )
    context_sources = {entry["sha256"] for entry in manifest.get("sources", [])}
    _require(
        _sha256(MINUTES_ROWS) in context_sources,
        "official-minute supplement is absent from forecast context sources",
    )

    payload = {
        "schema": "pres_2025_demo_boundary_audit_v2",
        "status": "pass",
        "target_election": "pres_2025",
        "forecast_cutoff": cutoff.date().isoformat(),
        "historical_scored_elections": list(SCORED_ELECTIONS),
        "historical_results_rows": int(len(results)),
        "historical_active_prediction_rows": int(len(active)),
        "candidate_registry_rows": int(len(registry)),
        "candidate_registry_sha256": _sha256(REGISTRY),
        "assembly_source_sha256": manifest["source_sha256"],
        "assembly_source_rows_scanned": manifest["source_rows_scanned"],
        "assembly_pres_2025_rows_included": manifest["target_rows_included"],
        "assembly_first_meeting_date": manifest["first_meeting_date"],
        "assembly_last_meeting_date": manifest["last_meeting_date"],
        "assembly_source_first_meeting_date": manifest["source_first_meeting_date"],
        "assembly_source_last_meeting_date": manifest["source_last_meeting_date"],
        "official_minutes_discovered": int(minutes_manifest["meetings_discovered"]),
        "official_minutes_completed": int(minutes_manifest["meetings_completed"]),
        "official_minutes_eligible_at_cutoff": int(
            minutes_manifest["meetings_eligible_at_cutoff"]
        ),
        "official_minutes_excluded_by_availability": int(
            minutes_manifest["meetings_excluded_by_availability"]
        ),
        "official_minutes_derived_rows": int(len(minute_rows)),
        "official_minutes_eligible_rows": int(minute_available.le(cutoff).sum()),
        "official_minutes_post_cutoff_rows": int(minute_available.gt(cutoff).sum()),
        "model_salience_rows": int(len(salience)),
        "model_candidate_link_rows": int(len(candidate_link)),
        "outcome_columns_used": [],
        "performance_metrics_computed": False,
        "known_source_gap": (
            "The official site exposes no exact first-publication timestamp. "
            "PDF CreationDate plus one day is used as a conservative proxy; "
            "records not provably eligible are retained but excluded."
        ),
    }
    AUDIT_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
