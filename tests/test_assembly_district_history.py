from __future__ import annotations

import pytest

from presidential_issue_engine.assembly_district_history import (
    build_assembly_district_history,
)


def _item(sd_name: str, wiw_name: str) -> dict[str, str]:
    return {
        "sgId": "20160413",
        "sgTypecode": "2",
        "sggName": "종로구",
        "sdName": sd_name,
        "wiwName": wiw_name,
        "jd01": "새누리당",
        "hbj01": "후보1",
        "dugsu01": "40",
        "jd02": "더불어민주당",
        "hbj02": "후보2",
        "dugsu02": "60",
    }


def test_constituency_parser_keeps_one_province_total_per_district() -> None:
    history = build_assembly_district_history(
        [
            _item("합계", "합계"),
            _item("서울특별시", "합계"),
            _item("서울특별시", "종로구"),
            _item("서울특별시", "합계"),
        ],
        election_id="assembly_2016_district",
        election_date="2016-04-13",
    )
    assert len(history) == 2
    assert set(history["region_id"]) == {"sido_11"}
    assert history["district_valid_votes"].unique().tolist() == [100.0]
    shares = dict(zip(history["candidate_name"], history["candidate_vote_share"]))
    assert shares == {"후보1": pytest.approx(0.4), "후보2": pytest.approx(0.6)}
    winner = history.loc[history["candidate_won"], "candidate_name"].tolist()
    assert winner == ["후보2"]
    assert set(history["available_date"]) == {"2016-04-14"}
