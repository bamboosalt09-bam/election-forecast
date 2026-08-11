from __future__ import annotations

import pandas as pd

from news_collector.sources.contextual_issue_weights import housing_issue_boosts_from_index


def test_housing_issue_boost_uses_prior_election_snapshot(tmp_path) -> None:
    path = tmp_path / "housing.csv"
    frame = pd.DataFrame(
        [
            {
                "region_id": "sido_11",
                "period": "2017-03-31",
                "value": 100.0,
                "available_date": "2017-04-30",
            },
            {
                "region_id": "sido_11",
                "period": "2021-12-31",
                "value": 180.0,
                "available_date": "2022-03-01",
            },
            {
                "region_id": "sido_11",
                "period": "2022-03-31",
                "value": 250.0,
                "available_date": "2022-05-30",
            },
        ]
    )
    frame.to_csv(path, index=False)

    boosts, diagnostics = housing_issue_boosts_from_index(
        path,
        {"pres_2017": "2017-05-09", "pres_2022": "2022-03-09"},
        ["pres_2017", "pres_2022"],
        max_boost=1.25,
    )

    assert boosts["pres_2022"]["housing"] == 1.2
    assert diagnostics.loc[0, "mean_housing_change_pct"] == 80.0
    assert diagnostics.loc[0, "current_period_max"] == "2021-12-31"
