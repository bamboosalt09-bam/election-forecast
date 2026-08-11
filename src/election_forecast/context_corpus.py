"""Point-in-time validation for external parliamentary context corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ELECTION_CUTOFFS = {
    "pres_2002": pd.Timestamp("2002-12-19"),
    "pres_2007": pd.Timestamp("2007-12-19"),
    "pres_2012": pd.Timestamp("2012-12-19"),
    "pres_2017": pd.Timestamp("2017-05-09"),
    "pres_2022": pd.Timestamp("2022-03-09"),
}

OUTCOME_COLUMN_FRAGMENTS = (
    "actual",
    "result",
    "vote_share",
    "candidate_votes",
    "valid_votes",
    "winner",
    "margin",
    "득표",
    "개표",
    "당선",
)


@dataclass(frozen=True)
class ContextCorpusValidation:
    rows: int
    earliest_available_date: str
    latest_available_date: str
    election_ids: tuple[str, ...]


def outcome_like_columns(columns: Iterable[object]) -> list[str]:
    """Return columns that could leak election outcomes into text training."""

    flagged: list[str] = []
    for value in columns:
        column = str(value)
        lowered = column.lower()
        if "sign_margin" in lowered:
            continue
        if any(fragment in lowered for fragment in OUTCOME_COLUMN_FRAGMENTS):
            flagged.append(column)
    return flagged


def validate_context_corpus(frame: pd.DataFrame) -> ContextCorpusValidation:
    """Enforce election-specific availability and outcome-blind inputs."""

    required = {"election_id", "available_date", "text"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"context corpus is missing required columns: {missing}")
    forbidden = outcome_like_columns(frame.columns)
    if forbidden:
        raise ValueError(f"context corpus contains outcome-like columns: {forbidden}")
    if frame.empty:
        raise ValueError("context corpus is empty")

    election_ids = frame["election_id"].astype(str)
    unknown = sorted(set(election_ids).difference(ELECTION_CUTOFFS))
    if unknown:
        raise ValueError(f"context corpus contains unsupported elections: {unknown}")
    dates = pd.to_datetime(frame["available_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("context corpus contains invalid available_date values")
    cutoffs = election_ids.map(ELECTION_CUTOFFS)
    late = dates > cutoffs
    if late.any():
        sample = frame.loc[late, ["election_id", "available_date"]].head(5)
        raise ValueError(
            "context corpus contains post-cutoff rows: "
            f"{sample.to_dict(orient='records')}"
        )
    if (dates > ELECTION_CUTOFFS["pres_2022"]).any():
        raise ValueError("context corpus contains post-2022 presidential-election rows")

    return ContextCorpusValidation(
        rows=int(len(frame)),
        earliest_available_date=dates.min().date().isoformat(),
        latest_available_date=dates.max().date().isoformat(),
        election_ids=tuple(sorted(set(election_ids))),
    )


def load_context_corpus(path: Path) -> tuple[pd.DataFrame, ContextCorpusValidation]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame, validate_context_corpus(frame)
