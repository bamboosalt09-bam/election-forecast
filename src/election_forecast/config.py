"""Configuration objects for the presidential forecast MVP.

The MVP uses a temporary linear Utility specification. The config keeps
weights explicit so the model can evolve without hardcoding political
judgments in Python code.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ModelWeights(BaseModel):
    """Linear Utility weights for one ensemble member."""

    w_party_base: float = 1.0
    w_poll: float = 1.0
    w_candidate: float = 1.0
    w_local_policy: float = 1.0
    w_issue: float = 1.0
    w_event: float = 1.0
    w_turnout: float = 1.0


class InteractionWeights(BaseModel):
    """Expansion coefficients for future interaction-heavy models."""

    a_candidate_local: float = 0.0
    a_policy_region: float = 0.0
    a_party_candidate_fit: float = 0.0
    a_issue_region: float = 0.0
    a_turnout_group: float = 0.0


class ForecastConfig(BaseModel):
    """Runtime configuration for the forecast pipeline."""

    camp_columns: List[str] = Field(
        default_factory=lambda: [
            "conservative",
            "liberal",
            "progressive",
            "centrist",
            "local_independent",
            "anti_party",
        ]
    )
    election_type_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "presidential": 0.80,
            "national_assembly_pr": 1.45,
            "assembly_pr": 1.45,
            "national_assembly_district": 0.18,
            "assembly_district": 0.18,
            "metro_council_pr": 0.55,
            "local_council_pr": 0.11,
            "metro_council_district": 0.18,
            "local_council_district": 0.028,
            "metro_governor": 0.004,
            "local_governor": 0.003,
            "education_superintendent": 0.0,
            "education_council": 0.0,
        }
    )
    recency_half_life_years: float = 6.0
    poll_half_life_days: float = 14.0
    issue_half_life_days: float = 21.0
    softmax_temperature: float = 1.0
    interaction_weights: InteractionWeights = Field(default_factory=InteractionWeights)
    weights_config: Dict[str, ModelWeights] = Field(
        default_factory=lambda: {
            "balanced": ModelWeights(),
            "party_base_heavy": ModelWeights(w_party_base=1.8, w_poll=0.8),
            "poll_heavy": ModelWeights(w_poll=1.8, w_party_base=0.8),
            "candidate_heavy": ModelWeights(w_candidate=1.8),
            "local_policy_heavy": ModelWeights(w_local_policy=1.8),
            "issue_heavy": ModelWeights(w_issue=1.8),
            "event_heavy": ModelWeights(w_event=1.8),
            "turnout_heavy": ModelWeights(w_turnout=1.8),
        }
    )


DEFAULT_CONFIG = ForecastConfig()
