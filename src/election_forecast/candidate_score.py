"""Candidate-level general premium features."""

from __future__ import annotations

import pandas as pd


def _score(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]
    return float(numeric / 100.0 if abs(numeric) > 1.0 else numeric)


def compute_candidate_general_premium(candidates: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple candidate-wide premium from CSV-provided scores."""

    if candidates.empty:
        return pd.DataFrame(columns=["candidate_id", "candidate_general_premium"])

    frame = candidates.copy()
    components = []
    for _, row in frame.iterrows():
        positive = (
            0.25 * _score(row["political_weight_score"])
            + 0.20 * _score(row["administrative_experience_score"])
            + 0.25 * _score(row["favorability_score"])
            + 0.20 * _score(row["expansion_score"])
        )
        negative = 0.20 * _score(row["unfavorability_score"]) + 0.20 * _score(row["risk_score"])
        components.append(positive - negative)
    frame["candidate_general_premium"] = components
    return frame[["candidate_id", "candidate_general_premium"]]
