"""BIGKinds 뉴스 메타데이터 export → 이슈 salience + 인물(후보)–이슈 연결.

BIGKinds(한국언론진흥재단)는 무료 자동 API가 없어 사용자가 "뉴스분석 → 데이터
다운로드"로 **메타데이터**를 export한다(본문 아님 → 저작권 안전, 1990~ 과거 커버).
이 모듈은 그 export를 읽어:

1. 제목·키워드·특성추출에서 이슈 키워드를 매칭 → (이슈 × 주) 기사수 → **salience**
   (instrument="bigkinds_meta", 연합 제목 salience와 같은 계약).
2. ``인물`` 컬럼으로 (후보 × 이슈 × 주) 동시언급 → **candidate_link** 신호.

**본문은 읽지도 저장하지도 않는다.** BIGKinds export의 메타데이터 컬럼만 사용한다.

기대 입력 (BIGKinds 표준 export 컬럼명을 기본 인식; 다르면 column_map로 매핑):
- ``일자`` (YYYYMMDD 또는 YYYY-MM-DD)
- ``제목``
- ``키워드`` (쉼표 구분)
- ``인물`` (쉼표 구분, 선택)
- ``언론사`` / ``통합 분류1`` (선택, 참고용)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from news_collector.sources.salience_base import normalize_within_election

_DEFAULT_MAP = {
    "date": "일자",
    "title": "제목",
    "keywords": "키워드",
    "persons": "인물",
    "press": "언론사",
    "category": "통합 분류1",
}


def load_bigkinds_metadata(path: str | Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a BIGKinds metadata export (CSV/Excel) into normalized columns.

    Returns columns: ``date (datetime), title, keywords, persons``.
    """

    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(p)
    else:
        # BIGKinds 공개 메타데이터는 CP949, '오늘의 이슈' 등 일부는 UTF-8 → 순차 시도.
        for enc in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                raw = pd.read_csv(p, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError(f"could not decode {p.name} as utf-8/cp949")
    cmap = {**_DEFAULT_MAP, **(column_map or {})}
    if cmap["date"] not in raw.columns or cmap["title"] not in raw.columns:
        raise ValueError(
            f"BIGKinds export must contain date/title columns "
            f"('{cmap['date']}', '{cmap['title']}'). Found: {list(raw.columns)[:12]}"
        )
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[cmap["date"]].astype(str).str.replace(r"[^0-9]", "", regex=True), format="%Y%m%d", errors="coerce"),
            "title": raw[cmap["title"]].astype(str).fillna(""),
            "keywords": raw.get(cmap["keywords"], "").astype(str).fillna("") if cmap["keywords"] in raw.columns else "",
            "persons": raw.get(cmap["persons"], "").astype(str).fillna("") if cmap["persons"] in raw.columns else "",
        }
    )
    return df.dropna(subset=["date"]).reset_index(drop=True)


def _week(d: pd.Timestamp) -> str:
    """ISO week start (Monday) as YYYY-MM-DD string."""

    return (d - pd.Timedelta(days=int(d.weekday()))).date().isoformat()


def _matched_issues(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    return [issue for issue, kws in keyword_map.items() if any(kw and kw in text for kw in kws)]


def metadata_to_salience(
    meta: pd.DataFrame, keyword_map: dict[str, list[str]], election_id: str
) -> pd.DataFrame:
    """Count articles per (issue, week) → salience [0,1] (instrument=bigkinds_meta)."""

    rows: list[dict[str, Any]] = []
    for rec in meta.itertuples(index=False):
        text = f"{rec.title} {rec.keywords}"
        week = _week(pd.Timestamp(rec.date))
        for issue in _matched_issues(text, keyword_map):
            rows.append({"issue_name": issue, "period": week})
    if not rows:
        return normalize_within_election(pd.DataFrame(columns=["issue_name", "period", "count"]), "count", election_id, "bigkinds_meta")
    counts = pd.DataFrame(rows).value_counts(["issue_name", "period"]).reset_index(name="count")
    return normalize_within_election(counts, "count", election_id, instrument="bigkinds_meta")


def metadata_to_candidate_issue(
    meta: pd.DataFrame, keyword_map: dict[str, list[str]], candidate_names: dict[str, str]
) -> pd.DataFrame:
    """(후보 × 이슈 × 주) 동시언급 카운트 → candidate_link 신호.

    ``candidate_names`` maps a display name found in the ``인물`` column to a
    ``slot`` (A/B/C/alpha). Returns ``slot, issue_name, period, cooccurrence``.
    """

    rows: list[dict[str, Any]] = []
    for rec in meta.itertuples(index=False):
        persons = str(rec.persons)
        slots = {slot for name, slot in candidate_names.items() if name and name in persons}
        if not slots:
            continue
        issues = _matched_issues(f"{rec.title} {rec.keywords}", keyword_map)
        week = _week(pd.Timestamp(rec.date))
        for slot in slots:
            for issue in issues:
                rows.append({"slot": slot, "issue_name": issue, "period": week})
    if not rows:
        return pd.DataFrame(columns=["slot", "issue_name", "period", "cooccurrence"])
    return pd.DataFrame(rows).value_counts(["slot", "issue_name", "period"]).reset_index(name="cooccurrence")
