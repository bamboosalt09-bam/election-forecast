from __future__ import annotations

import pandas as pd

from presidential_issue_engine.issue_vote_engine import _interest_rate_context_features
from scripts.fetch_bok_base_rate import normalize_bok_base_rate_rows


def test_bok_rate_normalizer_uses_month_end_as_conservative_availability() -> None:
    out = normalize_bok_base_rate_rows(
        [
            {"TIME": "202201", "DATA_VALUE": "1.00", "UNIT_NAME": "연%"},
            {"TIME": "202202", "DATA_VALUE": "1.25", "UNIT_NAME": "연%"},
        ]
    )

    assert out["period"].dt.strftime("%Y-%m-%d").tolist() == ["2022-01-31", "2022-02-28"]
    assert out["available_date"].equals(out["period"])
    assert out["indicator_name"].eq("bok_base_rate").all()


def test_interest_rate_context_excludes_post_cutoff_rate_and_penalizes_responsible_slot() -> None:
    periods = pd.date_range("2019-01-31", periods=27, freq="ME")
    rates = pd.DataFrame(
        {
            "period": periods,
            "available_date": periods,
            "indicator_name": "bok_base_rate",
            "value": [1.0] * 12 + [1.5] * 12 + [9.0, 9.0, 9.0],
        }
    )
    alignment = pd.DataFrame(
        [
            {"election_id": "pres_x", "slot": "A", "economic_responsibility_score": 1.0},
            {"election_id": "pres_x", "slot": "B", "economic_responsibility_score": -1.0},
        ]
    )

    out = _interest_rate_context_features(rates, alignment, {"pres_x": "2021-01-02"})
    effects = dict(zip(out["slot"], out["interest_rate_context_effect"]))

    assert set(out["interest_rate_latest_period"]) == {"2020-12-31"}
    assert out["bok_base_rate"].iloc[0] == 1.5
    assert effects["A"] <= 0.0
    assert effects["B"] >= 0.0
