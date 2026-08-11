"""Monte Carlo simulation for presidential variable models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from election_forecast.presidential.transfer import apply_transfer_adjustments, load_transfer_events
from election_forecast.presidential.utility_model import compute_utilities
from election_forecast.presidential.vote_share import utility_to_vote_share


def run_monte_carlo(
    variables: pd.DataFrame,
    weights: pd.DataFrame,
    uncertainty: pd.DataFrame,
    candidate_slots: pd.DataFrame,
    regions: pd.DataFrame,
    election_id: str,
    n_sim: int,
    temperature: float = 1.0,
    seed: int | None = None,
    transfer_events: pd.DataFrame | None = None,
    available_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run Monte Carlo simulations and return national slot results."""

    if n_sim <= 0:
        raise ValueError("n_sim must be greater than 0")
    rng = np.random.default_rng(seed)
    uncertainty_lookup = _uncertainty_lookup(uncertainty)
    normalized_events = (
        load_transfer_events(transfer_events, election_id, available_date)
        if transfer_events is not None
        else pd.DataFrame()
    )

    simulation_frames: list[pd.DataFrame] = []
    for simulation_id in range(n_sim):
        sampled_variables = _sample_variables(variables, uncertainty_lookup, rng)
        utilities, _ = compute_utilities(
            sampled_variables,
            weights,
            candidate_slots,
            regions,
            election_id,
            available_date,
        )
        if not normalized_events.empty:
            utilities, _ = apply_transfer_adjustments(utilities, normalized_events)
        predictions = utility_to_vote_share(utilities, temperature=temperature)
        national = _national_vote_share(predictions, regions)
        national["simulation_id"] = simulation_id
        national["election_id"] = election_id
        simulation_frames.append(national)

    results = pd.concat(simulation_frames, ignore_index=True)
    results["is_winner"] = False
    for _, group in results.groupby(["simulation_id", "model_name"]):
        active = group.loc[group["is_active_slot"].astype(bool)]
        if active.empty:
            continue
        winner_index = active["national_vote_share"].idxmax()
        results.loc[winner_index, "is_winner"] = True
    return results[
        ["simulation_id", "election_id", "model_name", "slot", "national_vote_share", "is_winner"]
    ].sort_values(["simulation_id", "model_name", "slot"]).reset_index(drop=True)


def summarize_monte_carlo(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize Monte Carlo national vote shares and win probabilities."""

    if results.empty:
        return pd.DataFrame(
            columns=[
                "election_id",
                "model_name",
                "slot",
                "mean_vote_share",
                "lower_95",
                "upper_95",
                "win_probability",
            ]
        )
    summary = results.groupby(["election_id", "model_name", "slot"], as_index=False).agg(
        mean_vote_share=("national_vote_share", "mean"),
        lower_95=("national_vote_share", lambda values: values.quantile(0.025)),
        upper_95=("national_vote_share", lambda values: values.quantile(0.975)),
        win_probability=("is_winner", "mean"),
    )
    return summary.sort_values(["model_name", "slot"]).reset_index(drop=True)


def _sample_variables(
    variables: pd.DataFrame,
    uncertainty_lookup: dict[str, dict[str, float | str]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    sampled = variables.copy()
    sampled["variable_value"] = pd.to_numeric(sampled["variable_value"], errors="coerce").fillna(0.0)
    for variable_name, indices in sampled.groupby("variable_name").groups.items():
        spec = uncertainty_lookup.get(variable_name)
        if spec is None:
            continue
        if spec["distribution"] != "normal":
            raise ValueError(f"Unsupported uncertainty distribution for {variable_name}: {spec['distribution']}")
        sigma = float(spec["sigma"])
        noise = rng.normal(0.0, sigma, size=len(indices))
        values = sampled.loc[indices, "variable_value"].to_numpy(dtype=float) + noise
        sampled.loc[indices, "variable_value"] = np.clip(
            values,
            float(spec["min_value"]),
            float(spec["max_value"]),
        )
    return sampled


def _uncertainty_lookup(uncertainty: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    frame = uncertainty.copy()
    if frame.empty:
        return {}
    for column in ["sigma", "min_value", "max_value"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["distribution"] = frame["distribution"].fillna("normal").astype(str).str.lower()
    return frame.set_index("variable_name")[["sigma", "distribution", "min_value", "max_value"]].to_dict("index")


def _national_vote_share(predictions: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "total_votes" in regions.columns:
        weights = regions[["region_id", "total_votes"]].drop_duplicates()
        weights["total_votes"] = pd.to_numeric(weights["total_votes"], errors="coerce").fillna(0.0)
        frame = frame.merge(weights, on="region_id", how="left")
        total_votes = frame[["region_id", "total_votes"]].drop_duplicates()["total_votes"].sum()
        if total_votes > 0:
            frame["weighted_share"] = frame["predicted_vote_share"] * frame["total_votes"].fillna(0.0)
            national = frame.groupby(["model_name", "slot", "is_active_slot"], as_index=False)["weighted_share"].sum()
            national["national_vote_share"] = national["weighted_share"] / total_votes
            return national[["model_name", "slot", "is_active_slot", "national_vote_share"]]

    national = frame.groupby(["model_name", "slot", "is_active_slot"], as_index=False)["predicted_vote_share"].mean()
    return national.rename(columns={"predicted_vote_share": "national_vote_share"})
