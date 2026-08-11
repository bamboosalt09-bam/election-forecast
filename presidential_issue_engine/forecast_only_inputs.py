"""Load outcome-free inputs for elections that are forecast but never scored."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from presidential_issue_engine.election_scope import (
    ELECTION_DATES,
    FORECAST_ONLY_ELECTIONS,
)
from presidential_issue_engine.point_in_time import filter_available_by_election


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_DIR = (
    ROOT / "data/raw/official_sources/assembly_pres_2025_context"
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_table(
    context_dir: Path,
    manifest: dict[str, object],
    name: str,
    election_id: str,
) -> pd.DataFrame:
    path = context_dir / name
    recorded = dict(manifest["outputs"])[name]
    if _sha256(path) != str(recorded["sha256"]):
        raise RuntimeError(f"forecast-only input hash drift: {name}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    forbidden = sorted(set(frame.columns) & FORBIDDEN_OUTCOME_COLUMNS)
    if forbidden:
        raise RuntimeError(f"forecast-only input contains outcome columns: {forbidden}")
    if set(frame["election_id"].astype(str)) != {election_id}:
        raise RuntimeError(f"forecast-only input has mixed elections: {name}")
    return filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name=f"forecast-only {name}",
    )


def load_forecast_only_assembly_inputs(
    election_id: str = "pres_2025",
    context_dir: Path = DEFAULT_CONTEXT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return PIT-filtered salience and candidate-link inputs for a demo run."""

    if election_id not in FORECAST_ONLY_ELECTIONS:
        raise ValueError(f"not a forecast-only election: {election_id}")
    manifest = json.loads((context_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "forecast_only_not_scored":
        raise RuntimeError("forecast-only context has an invalid status")
    if manifest.get("target_election") != election_id:
        raise RuntimeError("forecast-only context targets a different election")
    if manifest.get("pres_2025_outcome_used") is not False:
        raise RuntimeError("forecast-only context does not certify outcome exclusion")
    if manifest.get("performance_metrics_computed") is not False:
        raise RuntimeError("forecast-only context was evaluated during input construction")

    salience = _load_table(
        context_dir, manifest, "model_issue_salience.csv", election_id
    )
    candidate_link = _load_table(
        context_dir, manifest, "model_candidate_issue_link.csv", election_id
    )
    return salience, candidate_link, manifest


def attach_preliminary_slots(
    candidate_link: pd.DataFrame,
    preliminary_slots: pd.DataFrame,
) -> pd.DataFrame:
    """Attach outcome-blind slot assignments to candidate-id issue profiles."""

    required = {"election_id", "candidate_id", "slot", "available_date"}
    missing = sorted(required - set(preliminary_slots.columns))
    if missing:
        raise ValueError(f"preliminary slot registry is missing columns: {missing}")
    eligible_slots = filter_available_by_election(
        preliminary_slots,
        ELECTION_DATES,
        source_name="forecast-only preliminary slots",
    )
    if eligible_slots.duplicated(["election_id", "candidate_id"]).any():
        raise ValueError("preliminary slot registry has duplicate candidates")
    out = candidate_link.merge(
        eligible_slots[["election_id", "candidate_id", "slot"]],
        on=["election_id", "candidate_id"],
        how="left",
        validate="many_to_one",
    )
    if out["slot"].isna().any():
        missing_ids = sorted(out.loc[out["slot"].isna(), "candidate_id"].unique())
        raise ValueError(f"candidate issue profiles lack preliminary slots: {missing_ids}")
    return out
