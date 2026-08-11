"""Local policy fit features."""

from __future__ import annotations

import pandas as pd


def compute_local_policy_fit(
    region_issue_sensitivity: pd.DataFrame,
    candidate_policy_positions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute local policy fit by candidate and region.

    MVP formula:
    ``sum_j RegionSensitivity(r,j) x PolicyPosition(c,j) x CandidateCredibility(c,j)``.
    """

    if region_issue_sensitivity.empty or candidate_policy_positions.empty:
        return pd.DataFrame(columns=["candidate_id", "region_id", "local_policy_fit"])

    joined = region_issue_sensitivity.merge(candidate_policy_positions, on="issue_name", how="inner")
    for col in ["sensitivity_score", "policy_direction", "candidate_credibility"]:
        joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0.0)
    joined["local_policy_component"] = (
        joined["sensitivity_score"] * joined["policy_direction"] * joined["candidate_credibility"]
    )
    return (
        joined.groupby(["candidate_id", "region_id"], as_index=False)["local_policy_component"]
        .sum()
        .rename(columns={"local_policy_component": "local_policy_fit"})
    )
