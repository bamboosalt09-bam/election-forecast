"""선관위 개표현황 xlsx → presidential_results_standardized (A/B/C/alpha 슬롯).

선관위 선거통계시스템 개표현황은 두 포맷이 섞여 있어 둘 다 처리한다:

- **요약형** (예: 16·17대): ``시도명 | 선거인수 | 투표수 | [후보별 득표수×N] | 계 | 무효 | 기권``.
  시도당 2행(득표수/득표율%). 시도명이 약칭(서울) 또는 정식(서울특별시)일 수 있음.
- **상세형** (예: 18·19·20대): ``시도 | 구시군 | 읍면동 | 투표구 | 선거인수 | 투표수 |
  [후보별 득표수×N] | 후보자별 득표수\n계 | 무효투표수``. 시도 합계행은 ``구시군=='합계'``.

공통 규칙으로 흡수: 후보 컬럼 = ``정당\n이름`` 셀, 유효투표(계) = 마지막 후보 컬럼 +1.
슬롯 = 전국 합계 득표 1·2·3위 → A·B·C, 나머지 → alpha. vote_share = 슬롯득표/유효투표.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# 시도 약칭·정식명 모두 → (region_id, 정식명)
_SIDO_ALIASES: dict[str, tuple[str, str]] = {}
for _aliases, _rid, _name in [
    (("서울", "서울특별시"), "sido_11", "서울특별시"),
    (("부산", "부산광역시"), "sido_26", "부산광역시"),
    (("대구", "대구광역시"), "sido_27", "대구광역시"),
    (("인천", "인천광역시"), "sido_28", "인천광역시"),
    (("광주", "광주광역시"), "sido_29", "광주광역시"),
    (("대전", "대전광역시"), "sido_30", "대전광역시"),
    (("울산", "울산광역시"), "sido_31", "울산광역시"),
    (("세종", "세종특별자치시", "세종시"), "sido_36", "세종특별자치시"),
    (("경기", "경기도"), "sido_41", "경기도"),
    (("강원", "강원도", "강원특별자치도"), "sido_42", "강원특별자치도"),
    (("충북", "충청북도"), "sido_43", "충청북도"),
    (("충남", "충청남도"), "sido_44", "충청남도"),
    (("전북", "전라북도", "전북특별자치도"), "sido_45", "전북특별자치도"),
    (("전남", "전라남도"), "sido_46", "전라남도"),
    (("경북", "경상북도"), "sido_47", "경상북도"),
    (("경남", "경상남도"), "sido_48", "경상남도"),
    (("제주", "제주도", "제주특별자치도"), "sido_50", "제주특별자치도"),
]:
    for _a in _aliases:
        _SIDO_ALIASES[_a] = (_rid, _name)

_ALIAS_BY_LEN = sorted(_SIDO_ALIASES, key=len, reverse=True)  # 긴 별칭 우선 매칭
_SLOTS = ("A", "B", "C", "alpha")


def _num(x: Any) -> float | None:
    s = str(x).replace(",", "").strip()
    if s in {"", "nan", "·", "None"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _sido_key(raw: Any) -> str | None:
    """헬퍼(테스트 호환): 시도 약칭 반환."""
    name = str(raw or "").strip()
    for short in ("서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원",
                  "충북", "충남", "전북", "전남", "경북", "경남", "제주"):
        if name.startswith(short):
            return short
    return None


def _region(raw: Any) -> tuple[str, str] | None:
    name = str(raw or "").strip()
    if not name or name in {"전국", "합계", "계"}:
        return None
    for alias in _ALIAS_BY_LEN:
        if name.startswith(alias):
            return _SIDO_ALIASES[alias]
    return None


def _is_candidate_cell(s: Any) -> bool:
    t = str(s)
    return "\n" in t and "계" not in t and "득표수" not in t and "투표수" not in t and t.strip() != "nan"


def parse_results(path: str | Path, election_id: str, c_active_threshold: float = 3.0) -> pd.DataFrame:
    """개표현황 xlsx → 표준화 결과 DataFrame."""

    df = pd.read_excel(path, header=None, dtype=str)
    nrow, ncol = df.shape

    # 후보 헤더 행 = 후보 셀(정당\n이름)이 2개 이상인 첫 행
    cand_row = None
    for i in range(min(nrow, 12)):
        cols = [j for j in range(ncol) if _is_candidate_cell(df.iat[i, j])]
        if len(cols) >= 2:
            cand_row, cand_cols = i, cols
            break
    if cand_row is None:
        raise ValueError(f"{Path(path).name}: 후보 헤더 행을 찾지 못함")
    gye_col = max(cand_cols) + 1  # 유효투표(계)는 마지막 후보 컬럼 바로 다음

    candidates = []  # (col, party, name)
    for j in cand_cols:
        parts = str(df.iat[cand_row, j]).split("\n")
        candidates.append((j, parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")))

    # 포맷 판별: 헤더 블록에 '구시군'이 있으면 상세형
    header_cells = {str(df.iat[r, c]).strip() for r in range(cand_row + 1) for c in range(ncol)}
    detailed = "구시군" in header_cells
    gugun_col = 1 if detailed else None

    def is_sido_total(i: int) -> tuple[str, str] | None:
        reg = _region(df.iat[i, 0])
        if reg is None:
            return None
        if detailed and str(df.iat[i, gugun_col]).strip() != "합계":
            return None
        if not detailed and _num(df.iat[i, candidates[0][0]]) is None:
            return None  # 요약형의 % 행/빈 행 제외
        return reg

    # 전국 합계 행 → 슬롯 랭킹
    nat_row = None
    for i in range(cand_row + 1, nrow):
        c0 = str(df.iat[i, 0]).strip()
        if (detailed and c0 == "전국") or (not detailed and c0.startswith("합계")):
            nat_row = i
            break
    if nat_row is None:
        raise ValueError(f"{Path(path).name}: 전국 합계 행을 찾지 못함")

    nat = {(p, n): _num(df.iat[nat_row, c]) or 0.0 for c, p, n in candidates}
    ranked = sorted(candidates, key=lambda cpn: nat[(cpn[1], cpn[2])], reverse=True)
    slot_of = {c: "alpha" for c, _, _ in candidates}
    abc = ranked[:3]
    for slot, (c, p, n) in zip(("A", "B", "C"), abc):
        slot_of[c] = slot
    slot_pn = {slot: (p, n) for slot, (c, p, n) in zip(("A", "B", "C"), abc)}
    total_valid = sum(nat.values())
    c_share = (nat[(abc[2][1], abc[2][2])] / total_valid) if len(abc) >= 3 and total_valid else 0.0
    c_active = c_share * 100 >= c_active_threshold

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in range(cand_row + 1, nrow):
        reg = is_sido_total(i)
        if reg is None or reg[0] in seen:
            continue
        seen.add(reg[0])
        region_id, region_name = reg
        valid = _num(df.iat[i, gye_col]) or 0.0
        sv = {s: 0.0 for s in _SLOTS}
        for c, p, n in candidates:
            sv[slot_of[c]] += _num(df.iat[i, c]) or 0.0
        for slot in _SLOTS:
            party, name = slot_pn.get(slot, ("", "기타후보 합산" if slot == "alpha" else ""))
            rows.append(
                {
                    "election_id": election_id, "region_id": region_id, "region_name": region_name,
                    "province": region_name, "slot": slot, "candidate_name": name, "party_name": party,
                    "is_active_slot": (slot in {"A", "B", "alpha"}) or (slot == "C" and c_active),
                    "votes": sv[slot], "vote_share": round(sv[slot] / valid, 6) if valid else 0.0,
                }
            )
    return pd.DataFrame(rows)
