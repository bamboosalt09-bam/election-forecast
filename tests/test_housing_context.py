from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.issue_vote_engine import _housing_context_features, _housing_pressure_features


def test_housing_context_uses_latest_available_quarter() -> None:
    housing = pd.DataFrame(
        [
            {
                "region_id": "sido_11",
                "period": "2021-12-31",
                "value": 150.0,
                "yoy_change_pct": 10.0,
                "available_date": "2022-03-01",
            },
            {
                "region_id": "sido_11",
                "period": "2022-03-31",
                "value": 155.0,
                "yoy_change_pct": 8.0,
                "available_date": "2022-05-30",
            },
        ]
    )
    housing["period"] = pd.to_datetime(housing["period"])
    housing["available_date"] = pd.to_datetime(housing["available_date"])

    result = _housing_context_features(housing, {"pres_x": "2022-03-09"})

    assert result.loc[0, "housing_price_index"] == 150.0
    assert result.loc[0, "housing_price_period"] == "2021-12-31"


def test_housing_context_keeps_region_specific_latest_values() -> None:
    housing = pd.DataFrame(
        [
            {
                "region_id": "sido_11",
                "period": "2021-12-31",
                "value": 150.0,
                "yoy_change_pct": 10.0,
                "available_date": "2022-03-01",
            },
            {
                "region_id": "sido_26",
                "period": "2021-09-30",
                "value": 120.0,
                "yoy_change_pct": 5.0,
                "available_date": "2021-11-29",
            },
        ]
    )
    housing["period"] = pd.to_datetime(housing["period"])
    housing["available_date"] = pd.to_datetime(housing["available_date"])

    result = _housing_context_features(housing, {"pres_x": "2022-03-09"})
    values = dict(zip(result["region_id"], result["housing_price_index"]))

    assert values == {"sido_11": 150.0, "sido_26": 120.0}


def test_housing_pressure_uses_previous_election_baseline_and_available_current() -> None:
    housing = pd.DataFrame(
        [
            {
                "region_id": "sido_11",
                "period": "2017-03-31",
                "value": 100.0,
                "yoy_change_pct": 0.0,
                "available_date": "2017-04-30",
            },
            {
                "region_id": "sido_11",
                "period": "2021-12-31",
                "value": 180.0,
                "yoy_change_pct": 20.0,
                "available_date": "2022-03-01",
            },
            {
                "region_id": "sido_11",
                "period": "2022-03-31",
                "value": 190.0,
                "yoy_change_pct": 25.0,
                "available_date": "2022-05-30",
            },
        ]
    )
    housing["period"] = pd.to_datetime(housing["period"])
    housing["available_date"] = pd.to_datetime(housing["available_date"])
    alignment = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "slot": "B",
                "housing_responsibility_score": 1.0,
                "available_date": "2022-02-01",
            },
            {
                "election_id": "pres_2022",
                "slot": "A",
                "housing_responsibility_score": -1.0,
                "available_date": "2022-02-01",
            },
        ]
    )
    alignment["available_date"] = pd.to_datetime(alignment["available_date"])

    result = _housing_pressure_features(
        housing,
        alignment,
        {"pres_2017": "2017-05-09", "pres_2022": "2022-03-09"},
        ["pres_2017", "pres_2022"],
    )
    effects = dict(zip(result["slot"], result["housing_pressure_effect"]))

    assert result["housing_current_period"].iloc[0] == "2021-12-31"
    assert result["housing_cumulative_change_pct"].iloc[0] == 80.0
    assert effects["B"] < 0
    assert effects["A"] > 0


def test_housing_pressure_uses_sgg_dispersion_without_post_cutoff_rows() -> None:
    housing = pd.DataFrame(
        [
            {"region_id": "sido_11", "period": "2017-03-31", "value": 100.0, "yoy_change_pct": 0.0, "available_date": "2017-04-30"},
            {"region_id": "sido_11", "period": "2021-12-31", "value": 130.0, "yoy_change_pct": 8.0, "available_date": "2022-03-01"},
        ]
    )
    sgg = pd.DataFrame(
        [
            {"region_id": "sido_11", "sgg_name": "A", "period": "2017-03-31", "value": 100.0, "available_date": "2017-04-30"},
            {"region_id": "sido_11", "sgg_name": "B", "period": "2017-03-31", "value": 100.0, "available_date": "2017-04-30"},
            {"region_id": "sido_11", "sgg_name": "A", "period": "2021-12-31", "value": 180.0, "available_date": "2022-03-01"},
            {"region_id": "sido_11", "sgg_name": "B", "period": "2021-12-31", "value": 110.0, "available_date": "2022-03-01"},
            {"region_id": "sido_11", "sgg_name": "A", "period": "2022-03-31", "value": 500.0, "available_date": "2022-05-30"},
        ]
    )
    for frame in (housing, sgg):
        frame["period"] = pd.to_datetime(frame["period"])
        frame["available_date"] = pd.to_datetime(frame["available_date"])
    alignment = pd.DataFrame(
        [{"election_id": "pres_2022", "slot": "A", "housing_responsibility_score": 1.0}]
    )

    out = _housing_pressure_features(
        housing,
        alignment,
        {"pres_2017": "2017-05-09", "pres_2022": "2022-03-09"},
        ["pres_2017", "pres_2022"],
        sgg,
    )

    assert out.loc[0, "housing_sgg_median_change_pct"] == pytest.approx(45.0)
    assert out.loc[0, "housing_sgg_dispersion"] > 0.0
    assert out.loc[0, "housing_sgg_count"] == 2
    assert out.loc[0, "housing_pressure_intensity"] > 0.0
