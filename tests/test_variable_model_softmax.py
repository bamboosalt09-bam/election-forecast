import pandas as pd

from election_forecast.presidential.utility_model import compute_utilities
from election_forecast.presidential.vote_share import utility_to_vote_share


def test_softmax_uses_active_slots_only_and_inactive_gets_zero() -> None:
    variables = pd.DataFrame(
        {
            "election_id": ["pres_2012", "pres_2012", "pres_2012", "pres_2012"],
            "region_id": ["sgg_001", "sgg_001", "sgg_001", "sgg_001"],
            "slot": ["A", "B", "C", "alpha"],
            "variable_name": ["regional_base", "regional_base", "regional_base", "regional_base"],
            "variable_value": [1.0, 0.0, 10.0, -1.0],
            "available_date": ["2012-12-18"] * 4,
            "source_note": ["sample"] * 4,
        }
    )
    weights = pd.DataFrame(
        {"model_name": ["balanced"], "variable_name": ["regional_base"], "weight": [1.0], "notes": [""]}
    )
    utilities, _ = compute_utilities(
        variables,
        weights,
        pd.read_csv("data/presidential/candidate_slots.csv"),
        pd.read_csv("data/presidential/regions_master.csv").head(1),
        "pres_2012",
    )

    result = utility_to_vote_share(utilities)

    inactive_c = result.loc[result["slot"] == "C"].iloc[0]
    assert inactive_c["predicted_vote_share"] == 0
    assert abs(result["predicted_vote_share"].sum() - 1.0) < 1e-12


def test_variable_model_softmax_sums_to_one_by_region_model() -> None:
    utilities, _ = compute_utilities(
        pd.read_csv("data/presidential/political_variables.csv"),
        pd.read_csv("data/presidential/model_weights.csv"),
        pd.read_csv("data/presidential/candidate_slots.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
        "pres_2022",
    )
    result = utility_to_vote_share(utilities)

    sums = result.groupby(["region_id", "model_name"])["predicted_vote_share"].sum()
    assert all(abs(value - 1.0) < 1e-12 for value in sums)

