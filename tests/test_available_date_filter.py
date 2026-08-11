import pandas as pd
import pytest

from election_forecast.filters import filter_available_date


def test_filter_available_date_removes_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "value": [1, 2],
            "available_date": ["2026-01-01", "2026-02-01"],
        }
    )

    filtered = filter_available_date(frame, "2026-01-15")

    assert filtered["value"].tolist() == [1]


def test_filter_available_date_rejects_invalid_dates() -> None:
    frame = pd.DataFrame({"value": [1], "available_date": [None]})

    with pytest.raises(ValueError, match="missing or invalid"):
        filter_available_date(frame, "2026-01-15")
