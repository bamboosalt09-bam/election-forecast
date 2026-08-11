"""Minimal backtest runner."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from election_forecast.config import ForecastConfig
from election_forecast.ensemble import run_forecast_ensemble


def _actual_national_vote_share(election_results: pd.DataFrame, election_id: str) -> pd.DataFrame:
    """Compute actual national vote shares for one target election."""

    actual = election_results.loc[election_results["election_id"] == election_id].copy()
    if actual.empty:
        raise ValueError(f"No actual results found for election_id={election_id}")

    actual["votes"] = pd.to_numeric(actual["votes"], errors="coerce").fillna(0.0)
    if actual["votes"].sum() > 0:
        grouped = actual.groupby("candidate_name", as_index=False)["votes"].sum()
        grouped["actual_vote_share"] = grouped["votes"] / grouped["votes"].sum()
        return grouped[["candidate_name", "actual_vote_share"]]

    actual["vote_share"] = pd.to_numeric(actual["vote_share"], errors="coerce").fillna(0.0)
    actual["vote_share"] = actual["vote_share"].where(actual["vote_share"].abs().le(1.0), actual["vote_share"] / 100.0)
    grouped = actual.groupby("candidate_name", as_index=False)["vote_share"].mean()
    return grouped.rename(columns={"vote_share": "actual_vote_share"})


def run_backtest(
    raw_data: Dict[str, pd.DataFrame],
    election_id: str,
    forecast_date: str | pd.Timestamp,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Predict from a past forecast date and compare with final results."""

    _, ensemble_results, _ = run_forecast_ensemble(
        raw_data,
        forecast_date,
        config,
        exclude_election_id=election_id,
        target_election_id=election_id,
    )
    predicted = ensemble_results.rename(columns={"mean_national_vote_share": "predicted_vote_share"})
    actual = _actual_national_vote_share(raw_data["election_results"], election_id)
    result = predicted.merge(actual, on="candidate_name", how="inner")
    result["error"] = result["predicted_vote_share"] - result["actual_vote_share"]
    result["abs_error"] = result["error"].abs()
    return result[
        ["candidate_id", "candidate_name", "predicted_vote_share", "actual_vote_share", "error", "abs_error"]
    ]
