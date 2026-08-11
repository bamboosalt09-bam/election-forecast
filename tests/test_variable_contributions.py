import pandas as pd

from election_forecast.presidential.utility_model import compute_utilities


def test_variable_contribution_records_value_times_weight() -> None:
    _, contributions = compute_utilities(
        pd.read_csv("data/presidential/political_variables.csv"),
        pd.read_csv("data/presidential/model_weights.csv"),
        pd.read_csv("data/presidential/candidate_slots.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
        "pres_2022",
    )
    row = contributions.loc[
        (contributions["model_name"] == "balanced")
        & (contributions["region_id"] == "sgg_001")
        & (contributions["slot"] == "A")
        & (contributions["variable_name"] == "regional_base")
    ].iloc[0]

    assert row["variable_value"] == 0.45
    assert row["weight"] == 0.30
    assert abs(row["contribution"] - 0.135) < 1e-12

