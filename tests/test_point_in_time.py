from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.point_in_time import (
    filter_available_by_election,
    filter_observed_by_election,
    forecast_cutoff,
    parse_observed_dates,
)


ELECTION_DATES = {
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}


def test_forecast_cutoff_is_day_before_election() -> None:
    assert forecast_cutoff("pres_2022", ELECTION_DATES) == pd.Timestamp("2022-03-08")


def test_filter_available_rejects_missing_time_metadata() -> None:
    frame = pd.DataFrame([{"election_id": "pres_2022", "value": 1.0}])

    with pytest.raises(ValueError, match="missing point-in-time columns"):
        filter_available_by_election(frame, ELECTION_DATES, source_name="fixture")


def test_filter_available_rejects_unknown_election() -> None:
    frame = pd.DataFrame(
        [{"election_id": "pres_unknown", "available_date": "2022-03-01"}]
    )

    with pytest.raises(ValueError, match="without auditable cutoff metadata"):
        filter_available_by_election(frame, ELECTION_DATES, source_name="fixture")


def test_filter_available_excludes_election_day_and_future_rows() -> None:
    frame = pd.DataFrame(
        [
            {"election_id": "pres_2022", "available_date": "2022-03-08", "value": "keep"},
            {"election_id": "pres_2022", "available_date": "2022-03-09", "value": "drop"},
            {"election_id": "pres_2022", "available_date": "2022-03-10", "value": "drop"},
        ]
    )

    out = filter_available_by_election(frame, ELECTION_DATES, source_name="fixture")

    assert out["value"].tolist() == ["keep"]


def test_observed_date_filter_parses_korean_assembly_dates() -> None:
    parsed = parse_observed_dates(pd.Series(["1996년7월18일(목)", "2022-03-08"]))
    assert parsed.tolist() == [pd.Timestamp("1996-07-18"), pd.Timestamp("2022-03-08")]

    frame = pd.DataFrame(
        [
            {"election_id": "pres_2022", "meeting_date": "2022년3월8일(화)", "value": "keep"},
            {"election_id": "pres_2022", "meeting_date": "2022년3월9일(수)", "value": "drop"},
        ]
    )
    out = filter_observed_by_election(
        frame,
        ELECTION_DATES,
        source_name="assembly fixture",
        date_column="meeting_date",
    )
    assert out["value"].tolist() == ["keep"]
