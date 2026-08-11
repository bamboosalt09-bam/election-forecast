"""Utility calculation for presidential A/B/C/alpha slot models."""

from __future__ import annotations

import pandas as pd

from election_forecast.presidential.schemas import normalize_slots
from election_forecast.presidential.variables import prepare_variables


def compute_utilities(
    variables: pd.DataFrame,
    weights: pd.DataFrame,
    candidate_slots: pd.DataFrame,
    regions: pd.DataFrame,
    election_id: str,
    available_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Utility and variable contribution rows.

    Utility is a weighted sum of CSV-defined political variables. It is not a
    vote share; downstream code converts active slot utilities with softmax.
    """

    slots = normalize_slots(candidate_slots)
    slots = slots.loc[slots["election_id"] == election_id].copy()
    if slots.empty:
        raise ValueError(f"No candidate slots found for election_id={election_id}")

    model_weights = weights.copy()
    model_weights["weight"] = pd.to_numeric(model_weights["weight"], errors="coerce").fillna(0.0)
    variables_for_election = prepare_variables(variables, election_id, available_date)

    contribution_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    region_lookup = regions[["region_id", "region_name", "province"]].drop_duplicates()

    for model_name, weight_rows in model_weights.groupby("model_name"):
        grid = (
            region_lookup[["region_id"]]
            .drop_duplicates()
            .merge(slots[["slot", "is_active_slot"]], how="cross")
        )
        grid["model_name"] = model_name

        joined = grid.merge(
            weight_rows[["variable_name", "weight"]],
            how="cross",
        ).merge(
            variables_for_election[
                ["election_id", "region_id", "slot", "variable_name", "variable_value"]
            ],
            on=["region_id", "slot", "variable_name"],
            how="left",
        )
        joined["election_id"] = election_id
        joined["variable_value"] = joined["variable_value"].fillna(0.0)
        joined["contribution"] = joined["variable_value"] * joined["weight"]
        contribution_rows.append(
            joined[
                [
                    "election_id",
                    "region_id",
                    "slot",
                    "model_name",
                    "variable_name",
                    "variable_value",
                    "weight",
                    "contribution",
                ]
            ]
        )

        utility = (
            joined.groupby(["election_id", "region_id", "slot", "is_active_slot", "model_name"], as_index=False)[
                "contribution"
            ]
            .sum()
            .rename(columns={"contribution": "utility"})
            .merge(region_lookup, on="region_id", how="left")
        )
        prediction_rows.append(utility)

    utilities = pd.concat(prediction_rows, ignore_index=True)
    contributions = pd.concat(contribution_rows, ignore_index=True)
    utilities = utilities[
        [
            "election_id",
            "region_id",
            "region_name",
            "province",
            "slot",
            "is_active_slot",
            "model_name",
            "utility",
        ]
    ].sort_values(["model_name", "region_id", "slot"])
    contributions = contributions.sort_values(["model_name", "region_id", "slot", "variable_name"])
    return utilities.reset_index(drop=True), contributions.reset_index(drop=True)

