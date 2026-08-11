from __future__ import annotations

import pandas as pd

from presidential_issue_engine.issue_vote_engine import _economic_context_features


def test_economic_context_uses_only_available_indicators() -> None:
    indicators = pd.DataFrame(
        [
            {
                "period": "2019-12-31",
                "indicator_name": "real_gdp_growth_yoy",
                "value": 2.0,
                "available_date": "2020-01-31",
            },
            {
                "period": "2020-12-31",
                "indicator_name": "real_gdp_growth_yoy",
                "value": 1.0,
                "available_date": "2021-01-31",
            },
            {
                "period": "2021-12-31",
                "indicator_name": "real_gdp_growth_yoy",
                "value": 9.0,
                "available_date": "2021-12-31",
            },
        ]
    )
    indicators["period"] = pd.to_datetime(indicators["period"])
    indicators["available_date"] = pd.to_datetime(indicators["available_date"])
    alignment = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "slot": "A",
                "economic_responsibility_score": 1.0,
                "available_date": "2021-02-01",
            }
        ]
    )
    alignment["available_date"] = pd.to_datetime(alignment["available_date"])

    result = _economic_context_features(indicators, alignment, {"pres_x": "2021-03-01"})

    assert result.loc[0, "real_gdp_growth_yoy"] == 1.0


def test_economic_stress_penalizes_responsible_slot() -> None:
    indicators = pd.DataFrame(
        [
            {
                "period": f"{year}-12-31",
                "indicator_name": "real_gdp_growth_yoy",
                "value": value,
                "available_date": f"{year + 1}-01-31",
            }
            for year, value in [(2017, 4.0), (2018, 3.0), (2019, 2.0), (2020, -1.0)]
        ]
    )
    indicators["period"] = pd.to_datetime(indicators["period"])
    indicators["available_date"] = pd.to_datetime(indicators["available_date"])
    alignment = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "slot": "A",
                "economic_responsibility_score": 1.0,
                "available_date": "2021-02-01",
            },
            {
                "election_id": "pres_x",
                "slot": "B",
                "economic_responsibility_score": -1.0,
                "available_date": "2021-02-01",
            },
        ]
    )
    alignment["available_date"] = pd.to_datetime(alignment["available_date"])

    result = _economic_context_features(indicators, alignment, {"pres_x": "2021-03-01"})
    effects = dict(zip(result["slot"], result["economic_context_effect"]))

    assert effects["A"] < 0
    assert effects["B"] > 0


def test_trade_stress_is_separate_from_growth_stress() -> None:
    periods = pd.date_range("2019-01-31", periods=24, freq="ME")
    values = [100.0] * 18 + [-100.0] * 6
    indicators = pd.DataFrame(
        [
            {
                "period": period,
                "indicator_name": "current_account_balance",
                "value": value,
                "available_date": period,
            }
            for period, value in zip(periods, values)
        ]
    )
    indicators["period"] = pd.to_datetime(indicators["period"])
    indicators["available_date"] = pd.to_datetime(indicators["available_date"])
    alignment = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "slot": "A",
                "economic_responsibility_score": 1.0,
                "available_date": "2021-01-01",
            }
        ]
    )
    alignment["available_date"] = pd.to_datetime(alignment["available_date"])

    result = _economic_context_features(indicators, alignment, {"pres_x": "2021-01-02"})

    assert result.loc[0, "economic_context_effect"] == 0.0
    assert result.loc[0, "trade_context_effect"] < 0.0
