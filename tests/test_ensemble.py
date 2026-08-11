import pandas as pd

from election_forecast.ensemble import aggregate_ensemble


def test_ensemble_aggregates_model_spread() -> None:
    frame = pd.DataFrame(
        {
            "forecast_date": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"],
            "model_name": ["m1", "m2", "m1", "m2"],
            "candidate_id": ["a", "a", "b", "b"],
            "candidate_name": ["A", "A", "B", "B"],
            "region_id": ["r1", "r1", "r1", "r1"],
            "national_vote_share": [0.4, 0.5, 0.6, 0.5],
        }
    )

    result = aggregate_ensemble(frame)
    row = result.loc[result["candidate_id"] == "a"].iloc[0]

    assert row["mean_national_vote_share"] == 0.45
    assert row["min_model_vote_share"] == 0.4
    assert row["max_model_vote_share"] == 0.5
    assert abs(row["std_model_vote_share"] - 0.05) < 1e-12
