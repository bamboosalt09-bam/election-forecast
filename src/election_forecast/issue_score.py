"""Issue and media score feature calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_candidate_issue_scores(
    issue_scores: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    half_life_days: float,
) -> pd.DataFrame:
    """Aggregate manually entered issue scores up to the forecast date."""

    if issue_scores.empty:
        return pd.DataFrame(
            columns=["candidate_id", "issue_name", "issue_score", "issue_salience", "candidate_link_score"]
        )

    cutoff = pd.Timestamp(forecast_date)
    frame = issue_scores.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.loc[frame["date"].le(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=["candidate_id", "issue_name", "issue_score", "issue_salience", "candidate_link_score"]
        )

    for col in [
        "salience_score",
        "direction_score",
        "candidate_link_score",
        "media_reliability_score",
        "final_issue_score",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    missing_final = frame["final_issue_score"].eq(0.0)
    frame.loc[missing_final, "final_issue_score"] = (
        frame.loc[missing_final, "salience_score"]
        * frame.loc[missing_final, "direction_score"]
        * frame.loc[missing_final, "media_reliability_score"]
    )
    frame["final_issue_score"] = frame["final_issue_score"].clip(lower=-1.0, upper=1.0)
    age_days = (cutoff - frame["date"]).dt.days.clip(lower=0)
    frame["weight"] = np.exp(-age_days / half_life_days)
    frame["weighted_issue_score"] = frame["final_issue_score"] * frame["weight"]
    frame["weighted_salience"] = frame["salience_score"] * frame["weight"]
    frame["weighted_link"] = frame["candidate_link_score"] * frame["weight"]

    grouped = frame.groupby(["candidate_id", "issue_name"], as_index=False).agg(
        weighted_issue_score=("weighted_issue_score", "sum"),
        weighted_salience=("weighted_salience", "sum"),
        weighted_link=("weighted_link", "sum"),
        weight=("weight", "sum"),
    )
    grouped["issue_score"] = grouped["weighted_issue_score"] / grouped["weight"].replace(0, np.nan)
    grouped["issue_salience"] = grouped["weighted_salience"] / grouped["weight"].replace(0, np.nan)
    grouped["candidate_link_score"] = grouped["weighted_link"] / grouped["weight"].replace(0, np.nan)
    return grouped[
        ["candidate_id", "issue_name", "issue_score", "issue_salience", "candidate_link_score"]
    ].fillna(0.0)


def compute_issue_impact(
    candidate_issue_scores: pd.DataFrame,
    region_issue_sensitivity: pd.DataFrame,
    issue_scope_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute ``IssueImpact(c,r,t)`` from manual issue scores and region sensitivity."""

    columns = ["candidate_id", "region_id", "issue_impact", "issue_impact_national", "issue_impact_local"]
    if candidate_issue_scores.empty or region_issue_sensitivity.empty:
        return pd.DataFrame(columns=columns)

    joined = candidate_issue_scores.merge(region_issue_sensitivity, on="issue_name", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=columns)
    joined["sensitivity_score"] = pd.to_numeric(
        joined["sensitivity_score"],
        errors="coerce",
    ).fillna(0.0)
    for column in ["issue_score", "candidate_link_score"]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)

    if issue_scope_weights is not None and not issue_scope_weights.empty:
        scope = issue_scope_weights[["issue_name", "national_weight", "local_weight"]].copy()
        for column in ["national_weight", "local_weight"]:
            scope[column] = pd.to_numeric(scope[column], errors="coerce")
        joined = joined.merge(scope, on="issue_name", how="left")
    else:
        joined["national_weight"] = 0.0
        joined["local_weight"] = 1.0
    joined["national_weight"] = joined["national_weight"].fillna(0.0)
    joined["local_weight"] = joined["local_weight"].fillna(1.0)

    base = joined["issue_score"] * joined["candidate_link_score"]
    joined["issue_impact_national"] = base * joined["national_weight"]
    joined["issue_impact_local"] = base * joined["local_weight"] * joined["sensitivity_score"]
    joined["issue_impact"] = joined["issue_impact_national"] + joined["issue_impact_local"]
    return (
        joined.groupby(["candidate_id", "region_id"], as_index=False)[
            ["issue_impact", "issue_impact_national", "issue_impact_local"]
        ]
        .sum()
    )
