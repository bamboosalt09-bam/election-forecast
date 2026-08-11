"""Vote-share transformations and national aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_softmax_vote_share(
    utility_frame: pd.DataFrame,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Convert regional candidate Utility values into vote shares."""

    if temperature <= 0:
        raise ValueError("softmax temperature must be greater than 0")

    frame = utility_frame.copy()

    def _softmax(values: pd.Series) -> pd.Series:
        array = values.to_numpy(dtype=float) / temperature
        array = array - np.max(array)
        exp_values = np.exp(array)
        return pd.Series(exp_values / exp_values.sum(), index=values.index)

    frame["predicted_vote_share"] = frame.groupby("region_id")["utility"].transform(_softmax)
    return frame


def compute_national_vote_share(vote_share_frame: pd.DataFrame) -> pd.DataFrame:
    """Add national vote share weighted by expected regional voters."""

    frame = vote_share_frame.copy()
    frame["eligible_voters"] = pd.to_numeric(frame["eligible_voters"], errors="coerce").fillna(0.0)
    frame["expected_turnout"] = pd.to_numeric(frame["expected_turnout"], errors="coerce").fillna(0.0)
    frame["expected_turnout"] = frame["expected_turnout"].where(
        frame["expected_turnout"].abs().le(1.0), frame["expected_turnout"] / 100.0
    )
    frame["expected_votes"] = frame["eligible_voters"] * frame["expected_turnout"]
    frame["weighted_predicted_share"] = frame["predicted_vote_share"] * frame["expected_votes"]

    national = frame.groupby("candidate_id", as_index=False).agg(
        weighted_predicted_share=("weighted_predicted_share", "sum"),
        expected_votes=("expected_votes", "sum"),
    )
    total_expected_votes = frame[["region_id", "expected_votes"]].drop_duplicates()["expected_votes"].sum()
    national["national_vote_share"] = (
        national["weighted_predicted_share"] / total_expected_votes if total_expected_votes else 0.0
    )
    return frame.merge(national[["candidate_id", "national_vote_share"]], on="candidate_id", how="left")
