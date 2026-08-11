import pandas as pd

from election_forecast.presidential.utility_model import compute_utilities


def test_variable_model_utility_is_weighted_sum() -> None:
    utilities, _ = compute_utilities(
        pd.read_csv("data/presidential/political_variables.csv"),
        pd.read_csv("data/presidential/model_weights.csv"),
        pd.read_csv("data/presidential/candidate_slots.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
        "pres_2022",
    )
    row = utilities.loc[
        (utilities["model_name"] == "balanced")
        & (utilities["region_id"] == "sgg_001")
        & (utilities["slot"] == "A")
    ].iloc[0]

    assert abs(row["utility"] - 0.2475) < 1e-12

