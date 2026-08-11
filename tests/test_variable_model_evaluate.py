import pandas as pd

from election_forecast.presidential.evaluate import evaluate_predictions
from election_forecast.presidential.standardize_results import standardize_presidential_results


def test_variable_model_evaluation_metrics() -> None:
    actual, _ = standardize_presidential_results(
        pd.read_csv("data/presidential/presidential_results_raw.csv"),
        pd.read_csv("data/presidential/candidate_slots.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
    )
    predictions = actual.rename(columns={"vote_share": "predicted_vote_share"})
    predictions["model_name"] = "perfect"
    predictions["utility"] = 0.0

    evaluation, regional_errors = evaluate_predictions(predictions, actual, "pres_2022")

    overall = evaluation.loc[evaluation["metric"] == "overall_mae"].iloc[0]
    margin = evaluation.loc[evaluation["metric"] == "ab_margin_mae"].iloc[0]
    winner = evaluation.loc[evaluation["metric"] == "winner_accuracy"].iloc[0]
    assert overall["value"] == 0
    assert margin["value"] == 0
    assert winner["value"] == 1
    assert regional_errors["abs_error"].sum() == 0

