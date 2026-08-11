"""Offline tests for assembly-batch election-window counting."""

from __future__ import annotations

from collections import Counter
from datetime import date

import pytest

from news_collector.sources.assembly_batch import ELECTION_WINDOWS, accumulate, which_election

KW = {"housing": ["부동산"], "education": ["교육", "교육 개혁"]}


def test_which_election_maps_date_to_window() -> None:
    assert which_election(date(2002, 11, 1), ELECTION_WINDOWS) == "pres_2002"
    assert which_election(date(2022, 1, 15), ELECTION_WINDOWS) == "pres_2022"
    assert which_election(date(2010, 6, 1), ELECTION_WINDOWS) is None


def test_accumulate_filters_to_window_and_counts_weighted_matches() -> None:
    rows = [
        {"회의일자": "2002년 11월 5일", "발언자": "위원 김갑", "의원ID": "1", "발언내용1": "교육 개혁"},
        {"회의일자": "2002년 11월 5일", "발언자": "위원 이을", "의원ID": "2", "발언내용1": "부동산 대책"},
        {"회의일자": "2010년 6월 1일", "발언자": "위원 박병", "의원ID": "3", "발언내용1": "교육"},
    ]
    sal: Counter = Counter()
    mem: Counter = Counter()
    accumulate(rows, KW, ELECTION_WINDOWS, {"김갑": "A", "이을": "B"}, sal, mem)

    assert sal[("pres_2002", "education", "2002-11-04")] == pytest.approx(1.0)
    assert sal[("pres_2002", "housing", "2002-11-04")] == pytest.approx(0.35)
    assert all(k[0] == "pres_2002" for k in sal)
    assert mem[("pres_2002", "A", "education", "2002-11-04")] == pytest.approx(1.0)
    assert mem[("pres_2002", "B", "housing", "2002-11-04")] == pytest.approx(0.35)
