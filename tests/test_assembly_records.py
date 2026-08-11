"""국회 회의록 importer — date/ speaker parsing + issue signals (offline)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from news_collector.sources.assembly_records import (
    clean_speaker,
    parse_meeting_date,
    records_from_rows,
    speeches_to_member_issue,
    speeches_to_salience,
)

KW = {"housing": ["부동산", "집값"], "education": ["교육", "입시"], "security_nk": ["대북", "북핵"]}


def test_parse_meeting_date_korean_format() -> None:
    assert parse_meeting_date("2000年6月22日(木)") == date(2000, 6, 22)
    assert parse_meeting_date("2012-11-05") == date(2012, 11, 5)
    assert parse_meeting_date("없음") is None


def test_clean_speaker_strips_role() -> None:
    assert clean_speaker("위원장 이규택") == "이규택"
    assert clean_speaker("이규택 위원") == "이규택"
    assert clean_speaker("부위원장 홍길동") == "홍길동"


def _rows() -> list[dict]:
    return [
        {"회의일자": "2012年11月05日", "위원회": "교육위원회", "발언자": "위원 김갑", "의원ID": "100",
         "발언내용1": "교육 입시 제도 개편을", "발언내용2": "부동산 가격도 언급"},
        {"회의일자": "2012年11月06日", "위원회": "국방위원회", "발언자": "위원장 이을", "의원ID": "200",
         "발언내용1": "대북 정책과 북핵 대응"},
        {"회의일자": "bad", "위원회": "x", "발언자": "y", "의원ID": "1", "발언내용1": "교육"},  # dropped
    ]


def test_records_from_rows_parses_and_drops_bad_dates() -> None:
    df = records_from_rows(_rows())
    assert len(df) == 2  # bad-date row dropped
    assert df.iloc[0]["speaker"] == "김갑"
    assert "교육" in df.iloc[0]["text"]


def test_speeches_to_salience_counts_issues() -> None:
    df = records_from_rows(_rows())
    sal = speeches_to_salience(df, KW, "pres_2012")
    assert set(sal["issue_name"]) >= {"education", "housing", "security_nk"}
    assert (sal["instrument"] == "assembly_speech").all()


def test_speeches_to_member_issue_links_slot() -> None:
    df = records_from_rows(_rows())
    out = speeches_to_member_issue(df, KW, {"김갑": "A", "이을": "B"})
    assert ((out["slot"] == "A") & (out["issue_name"] == "education")).any()
    assert ((out["slot"] == "B") & (out["issue_name"] == "security_nk")).any()
