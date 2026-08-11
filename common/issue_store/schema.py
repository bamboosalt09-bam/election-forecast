"""The issue/event store row — one dated issue observation in the "memory".

Designed as a superset of the engine's existing ``issue_scores.csv`` so the
forecast keeps working, plus the dimensions needed for the A/B/C/alpha slot model
and the cross-election roadmap.

A row can represent either:
- a single dated **event/사건** (e.g. a scandal break, an endorsement), or
- an aggregated **issue-period** observation (e.g. "housing" salience over a week).

The store is *append-only memory*: rows accumulate over time and the rollup
applies a time-decay so older issues fade. ``available_date`` is mandatory for
leakage-free backtesting — an issue known only after the forecast cutoff is
dropped.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from pydantic import BaseModel, field_validator

from common.election_slot_schema.slots import SLOTS

# Issue/event categories. Open set in spirit, but a controlled vocabulary keeps
# the curated populator consistent. ``policy`` vs ``scandal`` etc. lets the
# rollup route favorable vs risk signals.
ISSUE_TYPES = (
    "policy",         # 공약/정책 입장
    "scandal",        # 의혹/논란 (risk-bearing)
    "endorsement",    # 지지선언
    "unification",    # 단일화
    "withdrawal",     # 사퇴
    "gaffe",          # 실언/막말 (risk-bearing)
    "achievement",    # 성과/치적
    "external_shock", # 외부 사건(경제·안보 등)
    "other",
)

POPULATORS = ("curated", "aggregate", "corpus", "manual")

# Issue types whose direction is treated as a risk/negative burden by default.
RISK_ISSUE_TYPES = frozenset({"scandal", "gaffe"})


class IssueEventRow(BaseModel):
    """One issue/event observation tied to a slot in an election.

    Column names for the scoring fields match ``issue_scores.csv`` exactly
    (``salience_score`` / ``direction_score`` / ``candidate_link_score`` /
    ``media_reliability_score`` / ``final_issue_score``) so the existing engine
    math in ``election_forecast.issue_score`` consumes it unchanged.
    """

    issue_id: str
    election_id: str
    issue_name: str
    issue_type: str = "policy"
    event_date: str
    available_date: str

    slot: str = "A"                     # analytic slot the issue attaches to
    candidate_id: str | None = None     # optional, for candidate-level (MVP) engine
    region_scope: str = "ALL"           # "ALL" or a specific region_id

    # Scoring fields (normalized to [-1, 1] except salience/reliability in [0, 1]).
    salience_score: float = 0.0         # 노출량/부각 (0..1)
    direction_score: float = 0.0        # 호오/방향 (-1..1): favorable + / unfavorable -
    candidate_link_score: float = 0.0   # 후보 연결도 (0..1)
    media_reliability_score: float = 1.0  # 매체 신뢰도 (0..1)
    final_issue_score: float | None = None  # optional precomputed; else derived

    populator: str = "curated"          # provenance: curated|aggregate|corpus|manual
    confidence: float = 1.0
    source_note: str | None = None

    @field_validator("slot")
    @classmethod
    def _known_slot(cls, value: str) -> str:
        if value not in SLOTS:
            raise ValueError(f"unknown slot {value!r}; expected one of {SLOTS}")
        return value

    @field_validator("issue_type")
    @classmethod
    def _known_issue_type(cls, value: str) -> str:
        if value not in ISSUE_TYPES:
            raise ValueError(f"unknown issue_type {value!r}; expected one of {ISSUE_TYPES}")
        return value

    @field_validator("populator")
    @classmethod
    def _known_populator(cls, value: str) -> str:
        if value not in POPULATORS:
            raise ValueError(f"unknown populator {value!r}; expected one of {POPULATORS}")
        return value

    @field_validator("direction_score")
    @classmethod
    def _direction_range(cls, value: float) -> float:
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"direction_score must be in [-1, 1], got {value}")
        return value


def issue_columns() -> List[str]:
    return list(IssueEventRow.model_fields)


def validate_issue_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate every issue-store row, raising with row context on the first bad one."""

    required = {"issue_id", "election_id", "issue_name", "event_date", "available_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"issue store is missing required columns: {missing}")
    for idx, record in frame.to_dict("index").items():
        clean = {k: _nan_to_none(v) for k, v in record.items() if k in IssueEventRow.model_fields}
        try:
            IssueEventRow(**clean)
        except Exception as exc:  # noqa: BLE001 - re-raise with row context
            raise ValueError(f"issue row {idx} failed validation: {exc}") from exc
    return frame


def _nan_to_none(value: object) -> object:
    """Blank CSV cells load as NaN; coerce to None for optional fields."""

    if isinstance(value, (list, dict)):
        return value
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value
