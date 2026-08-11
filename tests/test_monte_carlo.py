import pandas as pd

from election_forecast.presidential.monte_carlo import run_monte_carlo, summarize_monte_carlo


def _run(seed: int) -> pd.DataFrame:
    return run_monte_carlo(
        pd.read_csv("data/presidential/political_variables.csv"),
        pd.read_csv("data/presidential/model_weights.csv"),
        pd.read_csv("data/presidential/variable_uncertainty.csv"),
        pd.read_csv("data/presidential/candidate_slots.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
        "pres_2022",
        n_sim=25,
        temperature=1.0,
        seed=seed,
        transfer_events=pd.read_csv("data/presidential/transfer_events.csv"),
    )


def test_monte_carlo_is_reproducible_with_same_seed() -> None:
    first = _run(42)
    second = _run(42)

    pd.testing.assert_frame_equal(first, second)


def test_monte_carlo_summary_uses_quantile_interval_and_win_probabilities() -> None:
    summary = summarize_monte_carlo(_run(42))

    assert (summary["lower_95"] <= summary["mean_vote_share"]).all()
    assert (summary["upper_95"] >= summary["mean_vote_share"]).all()
    win_sums = summary.groupby("model_name")["win_probability"].sum()
    assert all(abs(value - 1.0) < 1e-12 for value in win_sums)


def test_inactive_slot_is_not_counted_as_winner() -> None:
    variables = pd.DataFrame(
        {
            "election_id": ["pres_2012"] * 4,
            "region_id": ["sgg_001"] * 4,
            "slot": ["A", "B", "C", "alpha"],
            "variable_name": ["regional_base"] * 4,
            "variable_value": [0.0, 0.0, 1.0, 0.0],
            "available_date": ["2012-12-18"] * 4,
            "source_note": ["sample"] * 4,
        }
    )
    results = run_monte_carlo(
        variables,
        pd.DataFrame({"model_name": ["m"], "variable_name": ["regional_base"], "weight": [1.0], "notes": [""]}),
        pd.DataFrame({"variable_name": ["regional_base"], "sigma": [0.0], "distribution": ["normal"], "min_value": [-1], "max_value": [1], "notes": [""]}),
        pd.read_csv("data/presidential/candidate_slots.csv"),
        pd.read_csv("data/presidential/regions_master.csv").head(1),
        "pres_2012",
        n_sim=3,
        seed=1,
    )

    assert not results.loc[results["slot"] == "C", "is_winner"].any()

