import pandas as pd

from election_forecast.presidential.standardize_results import standardize_presidential_results


def _standardized() -> pd.DataFrame:
    standardized, _ = standardize_presidential_results(
        pd.read_csv("data/presidential/presidential_results_raw.csv"),
        pd.read_csv("data/presidential/candidate_slots.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
    )
    return standardized


def test_standardize_aggregates_unmapped_candidates_to_alpha() -> None:
    result = _standardized()
    alpha = result.loc[
        (result["election_id"] == "pres_2022")
        & (result["region_id"] == "sgg_001")
        & (result["slot"] == "alpha")
    ].iloc[0]

    assert alpha["votes"] == 2000
    assert alpha["vote_share"] == 0.02


def test_standardize_emits_zero_for_inactive_c_slot() -> None:
    result = _standardized()
    c_slot = result.loc[
        (result["election_id"] == "pres_2012")
        & (result["region_id"] == "sgg_001")
        & (result["slot"] == "C")
    ].iloc[0]

    assert bool(c_slot["is_active_slot"]) is False
    assert c_slot["votes"] == 0
    assert c_slot["vote_share"] == 0
