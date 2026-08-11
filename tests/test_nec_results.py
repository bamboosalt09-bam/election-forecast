"""선관위 개표현황 파서 — 슬롯 배정 + 득표율 (구성한 xlsx로 end-to-end)."""

from __future__ import annotations

import openpyxl

from news_collector.sources.nec_results import _num, _sido_key, parse_results


def test_num_and_sido_helpers() -> None:
    assert _num("1,234  ") == 1234.0
    assert _num("·") is None and _num("nan") is None
    assert _sido_key("서울  ") == "서울"
    assert _sido_key("합계") is None


def _build_sheet(path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["중앙선거관리위원회"])  # r1 junk
    ws.append(["개표현황"])            # r2
    ws.append(["[제16대][대통령선거]"])  # r3
    # 후보 헤더 행 (r4): 시도명,선거인수,투표수, 3 candidates, 계, 무효, 기권
    ws.append(["시도명", "선거인수", "투표수", "P가\n갑", "P나\n을", "P다\n병", "계", "무효", "기권"])
    # 합계(전국): 갑 100, 을 300, 병 50 → 을>갑>병 → A=을, B=갑, C=병
    ws.append(["합계", "1000", "500", "100", "300", "50", "450", "10", "490"])
    ws.append(["", "", "", "22.2", "66.7", "11.1", "", "", ""])
    # 서울: 갑40 을50 병10 (유효100)
    ws.append(["서울", "200", "110", "40", "50", "10", "100", "5", "90"])
    ws.append(["", "", "", "40.0", "50.0", "10.0", "", "", ""])
    wb.save(path)


def test_parse_results_assigns_slots_by_national_rank(tmp_path) -> None:
    p = tmp_path / "개표현황.xlsx"
    _build_sheet(p)
    df = parse_results(p, "pres_test", c_active_threshold=3.0)

    abc = df[df.slot.isin(["A", "B", "C"])][["slot", "candidate_name"]].drop_duplicates()
    mapping = dict(zip(abc["slot"], abc["candidate_name"]))
    assert mapping == {"A": "을", "B": "갑", "C": "병"}  # national rank 을>갑>병

    seoul = df[df.region_id == "sido_11"].set_index("slot")
    assert abs(seoul.loc["A", "vote_share"] - 0.50) < 1e-6  # 을 50/100
    assert abs(seoul.loc["B", "vote_share"] - 0.40) < 1e-6  # 갑 40/100
    assert abs(seoul.loc["C", "vote_share"] - 0.10) < 1e-6  # 병 10/100
    assert abs(seoul["vote_share"].sum() - 1.0) < 1e-6      # sums to 1
