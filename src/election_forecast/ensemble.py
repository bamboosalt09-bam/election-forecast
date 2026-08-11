"""Model ensemble orchestration."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from election_forecast.config import ForecastConfig
from election_forecast.utility import compute_utility_frame, prepare_forecast_inputs
from election_forecast.vote_share import apply_softmax_vote_share, compute_national_vote_share


def run_model_forecast(
    data: Dict[str, pd.DataFrame],
    forecast_date: str | pd.Timestamp,
    model_name: str,
    config: ForecastConfig,
    target_election_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one weighted model and return regional predictions plus party base."""

    weights = config.weights_config[model_name]
    utility_frame, party_base = compute_utility_frame(
        data,
        forecast_date,
        config,
        weights,
        target_election_id=target_election_id,
    )
    vote_frame = apply_softmax_vote_share(utility_frame, config.softmax_temperature)
    vote_frame = compute_national_vote_share(vote_frame)
    vote_frame["forecast_date"] = str(pd.Timestamp(forecast_date).date())
    vote_frame["model_name"] = model_name
    vote_frame["uncertainty_group"] = "pending"
    columns = [
        "forecast_date",
        "model_name",
        "candidate_id",
        "candidate_name",
        "region_id",
        "predicted_vote_share",
        "national_vote_share",
        "utility",
        "uncertainty_group",
    ]
    return vote_frame[columns], party_base


def aggregate_ensemble(model_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate national vote shares across model variants."""

    national_by_model = model_results[
        ["forecast_date", "model_name", "candidate_id", "candidate_name", "national_vote_share"]
    ].drop_duplicates()
    grouped = national_by_model.groupby(["forecast_date", "candidate_id", "candidate_name"], as_index=False).agg(
        mean_national_vote_share=("national_vote_share", "mean"),
        min_model_vote_share=("national_vote_share", "min"),
        max_model_vote_share=("national_vote_share", "max"),
        std_model_vote_share=("national_vote_share", lambda values: float(values.std(ddof=0))),
    )
    return grouped


def assign_uncertainty_groups(model_results: pd.DataFrame, ensemble_results: pd.DataFrame) -> pd.DataFrame:
    """Attach simple low/medium/high uncertainty labels based on model spread."""

    labels = ensemble_results[["forecast_date", "candidate_id", "std_model_vote_share"]].copy()
    labels["uncertainty_group"] = pd.cut(
        labels["std_model_vote_share"],
        bins=[-1.0, 0.01, 0.03, float("inf")],
        labels=["low", "medium", "high"],
    ).astype(str)
    frame = model_results.drop(columns=["uncertainty_group"], errors="ignore").merge(
        labels[["forecast_date", "candidate_id", "uncertainty_group"]],
        on=["forecast_date", "candidate_id"],
        how="left",
    )
    frame["uncertainty_group"] = frame["uncertainty_group"].fillna("unknown")
    return frame


def run_forecast_ensemble(
    raw_data: Dict[str, pd.DataFrame],
    forecast_date: str | pd.Timestamp,
    config: ForecastConfig,
    exclude_election_id: str | None = None,
    target_election_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all configured model variants and aggregate the ensemble."""

    data = prepare_forecast_inputs(
        raw_data,
        forecast_date,
        exclude_election_id=exclude_election_id,
        target_election_id=target_election_id,
    )
    results = []
    party_base = pd.DataFrame()
    for model_name in config.weights_config:
        model_result, party_base = run_model_forecast(
            data,
            forecast_date,
            model_name,
            config,
            target_election_id=target_election_id,
        )
        results.append(model_result)

    forecast_results = pd.concat(results, ignore_index=True)
    ensemble_results = aggregate_ensemble(forecast_results)
    forecast_results = assign_uncertainty_groups(forecast_results, ensemble_results)
    return forecast_results, ensemble_results, party_base
