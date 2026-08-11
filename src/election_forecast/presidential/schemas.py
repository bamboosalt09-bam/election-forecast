"""Schemas and validation helpers for presidential A/B/C/alpha data."""

from __future__ import annotations

from typing import Dict, List, Type

import pandas as pd
from pydantic import BaseModel


SLOTS = ("A", "B", "C", "alpha")
VALID_VARIABLE_NAMES = (
    "regional_base",
    "candidate_strength",
    "national_mood",
    "third_candidate_structure",
    "local_issue_fit",
    "turnout_structure",
    "risk_or_negative",
)


class CandidateSlotRow(BaseModel):
    election_id: str
    slot: str
    candidate_name: str
    party_name: str
    is_active_slot: bool
    notes: str | None = None


class PresidentialRegionRow(BaseModel):
    region_id: str
    region_name: str
    province: str


class RawPresidentialResultRow(BaseModel):
    election_id: str
    region_id: str
    candidate_name: str
    party_name: str
    votes: float


class StandardizedPresidentialResultRow(BaseModel):
    election_id: str
    region_id: str
    region_name: str
    province: str
    slot: str
    candidate_name: str
    party_name: str
    is_active_slot: bool
    votes: float
    vote_share: float


class VariableDictionaryRow(BaseModel):
    variable_name: str
    variable_group: str
    description: str
    direction: str
    scale_min: float
    scale_max: float
    notes: str | None = None


class PoliticalVariableRow(BaseModel):
    election_id: str
    region_id: str
    slot: str
    variable_name: str
    variable_value: float
    available_date: str
    source_note: str | None = None


class ModelWeightRow(BaseModel):
    model_name: str
    variable_name: str
    weight: float
    notes: str | None = None


class PartyControversyScoreRow(BaseModel):
    election_id: str
    slot: str
    party_name: str
    period_start: str
    period_end: str
    controversy_count: float
    controversy_z_score: float
    source_count: float
    score: float
    available_date: str
    source_note: str | None = None


class CandidateToneScoreRow(BaseModel):
    election_id: str
    slot: str
    candidate_name: str
    period_start: str
    period_end: str
    mention_count: float
    positive_frame_count: float
    negative_frame_count: float
    tone_z_score: float
    visibility_z_score: float
    score: float
    available_date: str
    source_note: str | None = None


class TransferEventRow(BaseModel):
    election_id: str
    event_date: str
    available_date: str
    source_slot: str
    target_slot: str
    region_id: str
    transfer_strength: float
    transfer_rate: float
    abstention_rate: float
    notes: str | None = None


class VariableUncertaintyRow(BaseModel):
    variable_name: str
    sigma: float
    distribution: str
    min_value: float
    max_value: float
    notes: str | None = None


class VariableModelPredictionRow(BaseModel):
    election_id: str
    region_id: str
    region_name: str
    province: str
    slot: str
    is_active_slot: bool
    model_name: str
    utility: float
    predicted_vote_share: float


class VariableContributionRow(BaseModel):
    election_id: str
    region_id: str
    slot: str
    model_name: str
    variable_name: str
    variable_value: float
    weight: float
    contribution: float


class VariableModelEvaluationRow(BaseModel):
    target_election_id: str
    model_name: str
    metric: str
    slot: str | None = None
    value: float
    notes: str | None = None


class RegionalErrorRow(BaseModel):
    target_election_id: str
    region_id: str
    region_name: str
    province: str
    slot: str
    is_active_slot: bool
    model_name: str
    predicted_vote_share: float
    actual_vote_share: float
    error: float
    abs_error: float


class TransferContributionRow(BaseModel):
    election_id: str
    region_id: str
    source_slot: str
    target_slot: str
    model_name: str
    transfer_strength: float
    transfer_rate: float
    abstention_rate: float
    utility_adjustment: float


class MonteCarloResultRow(BaseModel):
    simulation_id: int
    election_id: str
    model_name: str
    slot: str
    national_vote_share: float
    is_winner: bool


class MonteCarloSummaryRow(BaseModel):
    election_id: str
    model_name: str
    slot: str
    mean_vote_share: float
    lower_95: float
    upper_95: float
    win_probability: float


PRESIDENTIAL_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "candidate_slots": list(CandidateSlotRow.model_fields),
    "regions_master": list(PresidentialRegionRow.model_fields),
    "presidential_results_raw": list(RawPresidentialResultRow.model_fields),
    "presidential_results_standardized": list(StandardizedPresidentialResultRow.model_fields),
    "variable_dictionary": list(VariableDictionaryRow.model_fields),
    "political_variables": list(PoliticalVariableRow.model_fields),
    "manual_political_variables": list(PoliticalVariableRow.model_fields),
    "model_weights": list(ModelWeightRow.model_fields),
    "party_controversy_scores": list(PartyControversyScoreRow.model_fields),
    "candidate_tone_scores": list(CandidateToneScoreRow.model_fields),
    "transfer_events": list(TransferEventRow.model_fields),
    "variable_uncertainty": list(VariableUncertaintyRow.model_fields),
    "variable_model_predictions": list(VariableModelPredictionRow.model_fields),
    "variable_contributions": list(VariableContributionRow.model_fields),
    "variable_model_evaluation": list(VariableModelEvaluationRow.model_fields),
    "regional_errors": list(RegionalErrorRow.model_fields),
    "transfer_contributions": list(TransferContributionRow.model_fields),
    "monte_carlo_results": list(MonteCarloResultRow.model_fields),
    "monte_carlo_summary": list(MonteCarloSummaryRow.model_fields),
}


PRESIDENTIAL_SCHEMA_MODELS: Dict[str, Type[BaseModel]] = {
    "candidate_slots": CandidateSlotRow,
    "regions_master": PresidentialRegionRow,
    "presidential_results_raw": RawPresidentialResultRow,
    "presidential_results_standardized": StandardizedPresidentialResultRow,
    "variable_dictionary": VariableDictionaryRow,
    "political_variables": PoliticalVariableRow,
    "manual_political_variables": PoliticalVariableRow,
    "model_weights": ModelWeightRow,
    "party_controversy_scores": PartyControversyScoreRow,
    "candidate_tone_scores": CandidateToneScoreRow,
    "transfer_events": TransferEventRow,
    "variable_uncertainty": VariableUncertaintyRow,
    "variable_model_predictions": VariableModelPredictionRow,
    "variable_contributions": VariableContributionRow,
    "variable_model_evaluation": VariableModelEvaluationRow,
    "regional_errors": RegionalErrorRow,
    "transfer_contributions": TransferContributionRow,
    "monte_carlo_results": MonteCarloResultRow,
    "monte_carlo_summary": MonteCarloSummaryRow,
}


def validate_required_columns(name: str, frame: pd.DataFrame) -> None:
    """Raise a clear error when a presidential CSV is missing columns."""

    missing = [col for col in PRESIDENTIAL_REQUIRED_COLUMNS[name] if col not in frame.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing required columns: {missing}")


def normalize_slots(slots: pd.DataFrame) -> pd.DataFrame:
    """Normalize slot labels and booleans while validating A/B/C/alpha membership."""

    frame = slots.copy()
    validate_required_columns("candidate_slots", frame)
    frame["slot"] = frame["slot"].astype(str)
    invalid = sorted(set(frame["slot"]) - set(SLOTS))
    if invalid:
        raise ValueError(f"candidate_slots.csv contains unsupported slots: {invalid}")
    if frame.duplicated(["election_id", "slot"]).any():
        duplicates = frame.loc[frame.duplicated(["election_id", "slot"], keep=False), ["election_id", "slot"]]
        raise ValueError(f"candidate_slots.csv has duplicate election/slot rows: {duplicates.to_dict('records')}")
    frame["is_active_slot"] = frame["is_active_slot"].map(_to_bool)
    return frame


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
