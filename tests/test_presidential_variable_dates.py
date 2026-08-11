from __future__ import annotations

import pandas as pd
import pytest

from election_forecast.presidential.variables import prepare_variables


def test_presidential_variables_reject_missing_availability() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "region_id": "ALL",
                "slot": "A",
                "variable_name": "x",
                "variable_value": 0.5,
                "available_date": None,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing or invalid"):
        prepare_variables(frame, "pres_2022", "2022-03-08")


def test_presidential_variables_exclude_future_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "region_id": "ALL",
                "slot": "A",
                "variable_name": "past",
                "variable_value": 0.5,
                "available_date": "2022-03-08",
            },
            {
                "election_id": "pres_2022",
                "region_id": "ALL",
                "slot": "A",
                "variable_name": "future",
                "variable_value": 0.5,
                "available_date": "2022-03-09",
            },
        ]
    )

    out = prepare_variables(frame, "pres_2022", "2022-03-08")

    assert out["variable_name"].tolist() == ["past"]
