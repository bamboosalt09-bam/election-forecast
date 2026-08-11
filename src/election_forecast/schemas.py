"""CSV schema declarations and lightweight validation helpers."""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Type

import pandas as pd
from pydantic import BaseModel


class RegionRow(BaseModel):
    region_id: str
    region_name: str
    province: str
    eligible_voters: float
    expected_turnout: float
    population: float


class ElectionResultRow(BaseModel):
    election_id: str
    election_type: str
    election_date: date
    region_id: str
    party_name: str
    camp: str
    candidate_name: str
    votes: float
    vote_share: float
    turnout: float
    available_date: date


class CandidateRow(BaseModel):
    candidate_id: str
    candidate_name: str
    party_name: str
    official_camp: str
    political_weight_score: float
    administrative_experience_score: float
    favorability_score: float
    unfavorability_score: float
    risk_score: float
    expansion_score: float
    available_date: date


class CandidatePartyVectorRow(BaseModel):
    candidate_id: str
    conservative: float
    liberal: float
    progressive: float
    centrist: float
    local_independent: float
    anti_party: float
    available_date: date


class PollRow(BaseModel):
    poll_id: str
    pollster: str
    published_date: date
    start_date: date
    end_date: date
    sample_size: float
    candidate_id: str
    support_rate: float
    pollster_weight: float
    available_date: date


class RegionIssueSensitivityRow(BaseModel):
    region_id: str
    issue_name: str
    sensitivity_score: float
    available_date: date


class CandidatePolicyPositionRow(BaseModel):
    candidate_id: str
    issue_name: str
    policy_direction: float
    candidate_credibility: float
    available_date: date


class IssueScoreRow(BaseModel):
    date: date
    candidate_id: str
    issue_name: str
    salience_score: float
    direction_score: float
    candidate_link_score: float
    media_reliability_score: float
    final_issue_score: float
    available_date: date


class EventEffectRow(BaseModel):
    event_id: str
    event_date: date
    available_date: date
    event_type: str
    source_candidate_id: str | None = None
    target_candidate_id: str | None = None
    region_id: str | None = None
    transfer_rate: float
    voter_compliance: float
    effect_strength: float


class CandidateIssueProfileRow(BaseModel):
    election_id: str
    candidate_id: str | None = None
    slot: str | None = None
    candidate_name: str | None = None
    issue_name: str
    association_strength: float
    direction: float
    available_date: date
    source_type: str
    confidence: float
    notes: str | None = None


class MegaIssueAxisRow(BaseModel):
    election_id: str
    mega_event: str
    primary_issue: str
    secondary_issue: str | None = None
    axis_weight: float
    regime_axis_weight: float
    available_date: date
    activation_method: str
    notes: str | None = None


class MegaIssueAttributionRow(BaseModel):
    election_id: str
    mega_event: str
    issue_name: str
    target_type: str
    target: str
    polarity: float
    weight: float
    available_date: date
    confidence: float
    notes: str | None = None


class IssueScopeWeightRow(BaseModel):
    issue_name: str
    national_weight: float
    local_weight: float
    notes: str | None = None


class IssueDirectionTermsRow(BaseModel):
    issue_name: str
    positive_terms: str | None = None
    negative_terms: str | None = None
    incumbent_positive_terms: str | None = None
    incumbent_negative_terms: str | None = None
    notes: str | None = None


class BlocHistoryResultRow(BaseModel):
    election_id: str
    election_type: str
    region_id: str
    bloc: str
    vote_share: float
    data_quality_weight: float


REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "regions": list(RegionRow.model_fields),
    "election_results": list(ElectionResultRow.model_fields),
    "candidates": list(CandidateRow.model_fields),
    "candidate_party_vectors": list(CandidatePartyVectorRow.model_fields),
    "polls": list(PollRow.model_fields),
    "region_issue_sensitivity": list(RegionIssueSensitivityRow.model_fields),
    "candidate_policy_positions": list(CandidatePolicyPositionRow.model_fields),
    "issue_scores": list(IssueScoreRow.model_fields),
    "event_effects": list(EventEffectRow.model_fields),
}

SCHEMA_MODELS: Dict[str, Type[BaseModel]] = {
    "regions": RegionRow,
    "election_results": ElectionResultRow,
    "candidates": CandidateRow,
    "candidate_party_vectors": CandidatePartyVectorRow,
    "polls": PollRow,
    "region_issue_sensitivity": RegionIssueSensitivityRow,
    "candidate_policy_positions": CandidatePolicyPositionRow,
    "issue_scores": IssueScoreRow,
    "event_effects": EventEffectRow,
}

OPTIONAL_COLUMNS: Dict[str, List[str]] = {
    "candidate_issue_profile": list(CandidateIssueProfileRow.model_fields),
    "mega_issue_axis": list(MegaIssueAxisRow.model_fields),
    "mega_issue_attribution": list(MegaIssueAttributionRow.model_fields),
    "issue_scope_weights": list(IssueScopeWeightRow.model_fields),
    "issue_direction_terms": list(IssueDirectionTermsRow.model_fields),
    "bloc_history_results": list(BlocHistoryResultRow.model_fields),
}

OPTIONAL_SCHEMA_MODELS: Dict[str, Type[BaseModel]] = {
    "candidate_issue_profile": CandidateIssueProfileRow,
    "mega_issue_axis": MegaIssueAxisRow,
    "mega_issue_attribution": MegaIssueAttributionRow,
    "issue_scope_weights": IssueScopeWeightRow,
    "issue_direction_terms": IssueDirectionTermsRow,
    "bloc_history_results": BlocHistoryResultRow,
}


def validate_required_columns(name: str, frame: pd.DataFrame) -> None:
    """Raise a clear error when a CSV is missing required columns."""

    missing = [col for col in REQUIRED_COLUMNS[name] if col not in frame.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing required columns: {missing}")


def validate_optional_columns(name: str, frame: pd.DataFrame) -> None:
    """Raise a clear error when an optional CSV is present but malformed."""

    missing = [col for col in OPTIONAL_COLUMNS[name] if col not in frame.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing required columns: {missing}")
    _validate_optional_ranges(name, frame)


def _validate_optional_ranges(name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if name == "candidate_issue_profile":
        _require_numeric_range(name, frame, "association_strength", 0.0, 1.0)
        _require_numeric_range(name, frame, "direction", -1.0, 1.0)
        _require_numeric_range(name, frame, "confidence", 0.0, 1.0)
    elif name == "mega_issue_axis":
        _require_numeric_range(name, frame, "axis_weight", 0.0, 2.0)
        _require_numeric_range(name, frame, "regime_axis_weight", 0.0, 2.0)
    elif name == "mega_issue_attribution":
        _require_values(name, frame, "target_type", {"candidate_id", "candidate_slot", "party", "camp", "incumbent_camp"})
        _require_numeric_range(name, frame, "polarity", -1.0, 1.0)
        _require_numeric_range(name, frame, "weight", 0.0, 1.0)
        _require_numeric_range(name, frame, "confidence", 0.0, 1.0)
    elif name == "issue_scope_weights":
        _require_numeric_range(name, frame, "national_weight", 0.0, 1.0)
        _require_numeric_range(name, frame, "local_weight", 0.0, 1.0)


def _require_numeric_range(name: str, frame: pd.DataFrame, column: str, low: float, high: float) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    bad = values.isna() | values.lt(low) | values.gt(high)
    if bad.any():
        indexes = list(frame.index[bad][:5])
        raise ValueError(f"{name}.csv column {column!r} must be in [{low}, {high}], bad rows: {indexes}")


def _require_values(name: str, frame: pd.DataFrame, column: str, allowed: set[str]) -> None:
    values = frame[column].fillna("").astype(str)
    bad = ~values.isin(allowed)
    if bad.any():
        indexes = list(frame.index[bad][:5])
        raise ValueError(f"{name}.csv column {column!r} has unsupported values, bad rows: {indexes}")
