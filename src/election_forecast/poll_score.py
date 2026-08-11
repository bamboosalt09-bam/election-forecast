"""Poll score calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_rate(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return values.where(values.abs().le(1.0), values / 100.0)


def compute_poll_scores(
    polls: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    half_life_days: float,
) -> pd.DataFrame:
    """Compute candidate poll score using sqrt(sample) x time_decay x pollster_weight."""

    if polls.empty:
        return pd.DataFrame(columns=["candidate_id", "poll_score"])

    cutoff = pd.Timestamp(forecast_date)
    frame = polls.copy()
    frame["published_date"] = pd.to_datetime(frame["published_date"], errors="coerce")
    frame = frame.loc[frame["published_date"].le(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["candidate_id", "poll_score"])

    age_days = (cutoff - frame["published_date"]).dt.days.clip(lower=0)
    frame["support_rate"] = _as_rate(frame["support_rate"])
    frame["sample_size"] = pd.to_numeric(frame["sample_size"], errors="coerce").fillna(0.0)
    frame["pollster_weight"] = pd.to_numeric(frame["pollster_weight"], errors="coerce").fillna(1.0)
    frame["time_decay"] = np.exp(-age_days / half_life_days)
    frame["weight"] = np.sqrt(frame["sample_size"].clip(lower=1.0)) * frame["time_decay"] * frame["pollster_weight"]
    frame["weighted_support"] = frame["support_rate"] * frame["weight"]

    grouped = frame.groupby("candidate_id", as_index=False).agg(
        weighted_support=("weighted_support", "sum"), weight=("weight", "sum")
    )
    grouped["poll_score"] = grouped["weighted_support"] / grouped["weight"].replace(0, np.nan)
    grouped["poll_score"] = grouped["poll_score"].fillna(0.0)
    return grouped[["candidate_id", "poll_score"]]
