"""The numericalization contract.

This is the single most important shared artifact. Both competitions turn raw
material (news, editorials, polls, results) into numbers, then feed those numbers
to a forecast engine. *How* they numericalize differs:

- statistics competition : rule-based — word/expression frequency, negative-keyword
  ratio, poll trends. Transparent, reproducible, AI-free.
- open-source competition: may additionally use an open-weight model to score
  stance / frame / issue-linkage more precisely.

Both MUST emit rows in THIS shape so the downstream engine and the evaluation
ruler never know — or care — who produced them. A "scorer" is therefore a
pluggable component; the engine binds to the schema, not to the scorer.

Values are normalized to ``[-1.0, 1.0]``: favorable positive, unfavorable
negative, unknown / not-applicable ``0``. ``available_date`` is mandatory so the
backtest can drop any feature known only after the forecast cutoff (leakage
prevention).

The seven variable names mirror ``election_forecast.presidential`` exactly, so
the existing engine consumes ``common`` features unchanged.
"""

from __future__ import annotations

from typing import List

import pandas as pd
from pydantic import BaseModel, field_validator

from common.shared_schema.election import AGGREGATION_RULES, ELECTION_TYPES
from common.election_slot_schema.slots import SLOTS

# Variable -> group. Groups document intent; weights are applied per-group-member
# by the engine's model_weights file, not here.
VARIABLE_GROUPS = {
    "regional_base": "structure",
    "candidate_strength": "candidate",
    "national_mood": "environment",
    "third_candidate_structure": "structure",
    "local_issue_fit": "issue",
    "turnout_structure": "turnout",
    "risk_or_negative": "risk",
}

VALID_VARIABLE_NAMES = tuple(VARIABLE_GROUPS)


class FeatureRow(BaseModel):
    """One numericalized variable for one slot in one contest at one time.

    Carries the cross-election dimensions so the same table serves presidential,
    legislative, and local data. Stats data leaves them at presidential defaults.
    """

    election_id: str
    election_type: str = "presidential"
    contest_id: str = ""  # blank => presidential single national contest
    region_id: str
    slot: str
    variable_name: str
    variable_value: float
    available_date: str
    aggregation_rule: str = "national_vote_share"
    scorer: str = "rule"  # provenance: "rule" | "openweight:<model>" | "manual"
    source_note: str | None = None

    @field_validator("variable_value")
    @classmethod
    def _within_unit_range(cls, value: float) -> float:
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"variable_value must be in [-1, 1], got {value}")
        return value

    @field_validator("variable_name")
    @classmethod
    def _known_variable(cls, value: str) -> str:
        if value not in VALID_VARIABLE_NAMES:
            raise ValueError(f"unknown variable_name {value!r}; expected one of {VALID_VARIABLE_NAMES}")
        return value

    @field_validator("slot")
    @classmethod
    def _known_slot(cls, value: str) -> str:
        if value not in SLOTS:
            raise ValueError(f"unknown slot {value!r}; expected one of {SLOTS}")
        return value

    @field_validator("election_type")
    @classmethod
    def _known_election_type(cls, value: str) -> str:
        if value not in ELECTION_TYPES:
            raise ValueError(f"unknown election_type {value!r}; expected one of {ELECTION_TYPES}")
        return value

    @field_validator("aggregation_rule")
    @classmethod
    def _known_rule(cls, value: str) -> str:
        if value not in AGGREGATION_RULES:
            raise ValueError(f"unknown aggregation_rule {value!r}; expected one of {AGGREGATION_RULES}")
        return value


def feature_columns() -> List[str]:
    return list(FeatureRow.model_fields)


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate every row against :class:`FeatureRow`, returning a clean frame.

    Raises ``ValueError`` with the offending row index on the first bad row so a
    scorer's output can be checked before it ever reaches the engine.
    """

    required = {"election_id", "region_id", "slot", "variable_name", "variable_value", "available_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature frame is missing required columns: {missing}")
    for idx, record in frame.to_dict("index").items():
        clean = {k: _nan_to_none(v) for k, v in record.items() if k in FeatureRow.model_fields}
        try:
            FeatureRow(**clean)
        except Exception as exc:  # noqa: BLE001 - re-raise with row context
            raise ValueError(f"feature row {idx} failed validation: {exc}") from exc
    return frame


def _nan_to_none(value: object) -> object:
    """Blank CSV cells load as NaN; coerce to None for optional fields."""

    if isinstance(value, (list, dict)):
        return value
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value
