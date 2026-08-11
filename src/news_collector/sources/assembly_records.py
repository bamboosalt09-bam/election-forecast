"""국회 회의록(상임위/특위/본회의) → (의원 × 이슈 × 시점) 발언 신호.

공공기록(완전 합법)이며 제16대~제22대(약 2000~2024+)를 덮어 **6개 대선 시기와
정렬**된다. 뉴스 salience(노출량)와 달리, 회의록은 *정치인이 어떤 이슈를 얼마나
다루는가* = 이슈 소유/연결(candidate_link)과 강조의 신호를 준다.

입력 스키마(국회 회의록 데이터셋):
``회의번호, 회의록구분, 대수, 회의구분, 위원회, 회수, 차수, 기타정보, 회의일자,
안건, 발언자, 의원ID, 발언순번, 발언내용1..7``

- ``회의일자``: "2000年6月22日(木)" 형식 → 숫자만 뽑아 파싱.
- ``발언자``: "위원장 이규택"처럼 역할어가 앞에 붙음 → 정리.
- ``발언내용1..7``: 길어서 여러 칸으로 분할 → 합쳐서 키워드 매칭.

본문(발언)을 *저장*하지 않는다 — 이슈 매칭 결과(카운트)만 남긴다.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Iterable

import pandas as pd

from election_forecast.features.issue_matcher import matched_issues
from news_collector.sources.salience_base import normalize_within_election

_DATE = re.compile(r"(\d{4})\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})")
_ROLES = (
    "위원장직무대행", "의장직무대행", "위원장대리", "직무대행", "권한대행",
    "부위원장", "소위원장", "위원장", "부의장", "의장",
    "수석전문위원", "전문위원", "사무총장", "사무차장", "간사", "위원", "의원",
)
_SPEECH_COLS = [f"발언내용{i}" for i in range(1, 8)]


def parse_meeting_date(value: Any) -> date | None:
    """'2000年6月22日(木)' 등에서 날짜 추출."""

    if value is None:
        return None
    m = _DATE.search(str(value))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except (ValueError, TypeError):
        return None


def clean_speaker(value: Any) -> str:
    """'위원장 이규택' → '이규택' (역할어 제거)."""

    text = str(value or "").strip()
    for role in _ROLES:
        text = text.replace(role, " ")
    return " ".join(text.split())


def records_from_rows(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """헤더-매핑된 dict 행들 → 정규화 레코드(date, committee, speaker, member_id, text)."""

    out: list[dict[str, Any]] = []
    for r in rows:
        d = parse_meeting_date(r.get("회의일자"))
        if d is None:
            continue
        text = " ".join(str(r.get(c) or "").strip() for c in _SPEECH_COLS if r.get(c))
        out.append(
            {
                "date": d,
                "committee": str(r.get("위원회") or "").strip(),
                "speaker": clean_speaker(r.get("발언자")),
                "member_id": str(r.get("의원ID") or "").strip(),
                "text": text,
            }
        )
    return pd.DataFrame(out, columns=["date", "committee", "speaker", "member_id", "text"])


def _week(d: date) -> str:
    ts = pd.Timestamp(d)
    return (ts - pd.Timedelta(days=int(ts.weekday()))).date().isoformat()


def _matched_issues(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    return matched_issues(text, keyword_map)


def speeches_to_salience(records: pd.DataFrame, keyword_map: dict[str, list[str]], election_id: str) -> pd.DataFrame:
    """이슈별·주별 발언 빈도 → salience [0,1] (instrument=assembly_speech)."""

    rows: list[dict[str, Any]] = []
    for rec in records.itertuples(index=False):
        wk = _week(rec.date)
        for issue in _matched_issues(rec.text, keyword_map):
            rows.append({"issue_name": issue, "period": wk})
    if not rows:
        return normalize_within_election(pd.DataFrame(columns=["issue_name", "period", "count"]), "count", election_id, "assembly_speech")
    counts = pd.DataFrame(rows).value_counts(["issue_name", "period"]).reset_index(name="count")
    return normalize_within_election(counts, "count", election_id, instrument="assembly_speech")


def speeches_to_member_issue(
    records: pd.DataFrame, keyword_map: dict[str, list[str]], member_to_slot: dict[str, str]
) -> pd.DataFrame:
    """(슬롯 × 이슈 × 주) 발언 빈도 = 후보의 이슈 소유/강조 신호.

    ``member_to_slot`` maps 발언자명 (or 의원ID) → slot. Returns
    ``slot, issue_name, period, mentions``.
    """

    rows: list[dict[str, Any]] = []
    for rec in records.itertuples(index=False):
        slot = member_to_slot.get(rec.speaker) or member_to_slot.get(rec.member_id)
        if not slot:
            continue
        wk = _week(rec.date)
        for issue in _matched_issues(rec.text, keyword_map):
            rows.append({"slot": slot, "issue_name": issue, "period": wk})
    if not rows:
        return pd.DataFrame(columns=["slot", "issue_name", "period", "mentions"])
    return pd.DataFrame(rows).value_counts(["slot", "issue_name", "period"]).reset_index(name="mentions")
