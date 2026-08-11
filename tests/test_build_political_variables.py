import pandas as pd

from election_forecast.presidential.feature_builder import build_political_variables


def test_party_controversy_becomes_national_mood_for_all_regions() -> None:
    result = build_political_variables(
        pd.read_csv("data/presidential/manual_political_variables.csv"),
        pd.read_csv("data/presidential/party_controversy_scores.csv"),
        pd.read_csv("data/presidential/candidate_tone_scores.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
    )
    rows = result.loc[(result["slot"] == "A") & (result["variable_name"] == "national_mood")]

    assert set(rows["region_id"]) == {"sgg_001", "sgg_002"}
    assert set(rows["variable_value"]) == {0.15}


def test_candidate_tone_becomes_candidate_strength_for_all_regions() -> None:
    result = build_political_variables(
        pd.DataFrame(columns=pd.read_csv("data/presidential/manual_political_variables.csv").columns),
        pd.read_csv("data/presidential/party_controversy_scores.csv"),
        pd.read_csv("data/presidential/candidate_tone_scores.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
    )
    rows = result.loc[(result["slot"] == "B") & (result["variable_name"] == "candidate_strength")]

    assert set(rows["region_id"]) == {"sgg_001", "sgg_002"}
    assert set(rows["variable_value"]) == {0.10}


def test_manual_values_override_generated_values() -> None:
    result = build_political_variables(
        pd.read_csv("data/presidential/manual_political_variables.csv"),
        pd.read_csv("data/presidential/party_controversy_scores.csv"),
        pd.read_csv("data/presidential/candidate_tone_scores.csv"),
        pd.read_csv("data/presidential/regions_master.csv"),
    )
    row = result.loc[
        (result["region_id"] == "sgg_002")
        & (result["slot"] == "B")
        & (result["variable_name"] == "candidate_strength")
    ].iloc[0]

    assert row["variable_value"] == 0.20
    assert row["source_note"].startswith("manual_political_variables")

