"""역대 국회의원 인적사항 → {의원명: 정당} 매핑 (대수별).

회의록엔 정당이 없으므로 이 명부로 발언자→정당을 붙인다. 명부 컬럼:
``대(제N대), 이름, 정당명`` (+ 한자명/선거구 등). 매칭 키는 (대수, 이름).

정당 → 슬롯(A/B/C/alpha) 변환은 선거별 후보 정당(NEC API)로 별도 수행한다.
위성정당·당명 변경(미래통합당↔국민의힘 등)은 ``party_bloc``으로 진영 단위 흡수.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def load_member_party(path: str | Path, term: str | None = None) -> dict[str, str]:
    """{의원명: 정당명}. ``term`` 예 '21' 이면 제21대만."""

    for enc in ("cp949", "utf-8-sig", "euc-kr"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(f"cannot decode {Path(path).name}")
    name_col = "이름" if "이름" in df.columns else df.columns[2]
    party_col = "정당명" if "정당명" in df.columns else next(c for c in df.columns if "정당" in c)
    if term is not None and "대" in df.columns:
        df = df[df["대"].astype(str).str.contains(str(term))]
    df = df.dropna(subset=[name_col, party_col])
    return {str(n).strip(): str(p).strip() for n, p in zip(df[name_col], df[party_col])}


# 진영(블록) 정규화: 당명 변경·위성정당을 대표 정당으로 묶음
_BLOC = {
    "미래통합당": "국민의힘", "미래한국당": "국민의힘", "자유한국당": "국민의힘", "새누리당": "국민의힘",
    "한나라당": "국민의힘", "국민의힘": "국민의힘",
    "더불어시민당": "더불어민주당", "열린민주당": "더불어민주당", "민주통합당": "더불어민주당",
    "새정치민주연합": "더불어민주당", "열린우리당": "더불어민주당", "새천년민주당": "더불어민주당",
    "더불어민주당": "더불어민주당",
}


_OPEN_ASSEMBLY = "https://open.assembly.go.kr/portal/openapi/nprlapfmaufmqytet"


def party_for_daesu(dae_text: str, daesu: int | str) -> str | None:
    """DAE 다선 이력 텍스트에서 해당 대수의 정당 추출.

    예: '제16대 국회의원(전북 군산) 새천년민주당' (DAESU=16) → '새천년민주당'.
    """

    pat = re.compile(rf"제\s*{int(daesu)}\s*대\s*(?:국회의원)?\s*(?:\([^)]*\))?\s*(.+)")
    for line in str(dae_text or "").splitlines():
        m = pat.match(line.strip())
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def fetch_roster(daesu: int | str, key: str, page_size: int = 1000) -> list[dict[str, str]]:
    """열린국회 API로 한 대수 명부 조회 → [{daesu, name, party}]."""

    import httpx

    out: list[dict[str, str]] = []
    page = 1
    while True:
        r = httpx.get(
            _OPEN_ASSEMBLY,
            params={"KEY": key, "Type": "json", "pIndex": page, "pSize": page_size, "DAESU": str(daesu)},
            timeout=30,
        )
        blocks = r.json().get("nprlapfmaufmqytet")
        if not blocks:
            break
        row_blk = next((b for b in blocks if "row" in b), None)
        head_blk = next((b for b in blocks if "head" in b), None)
        if not row_blk:
            break
        for row in row_blk["row"]:
            out.append(
                {
                    "daesu": str(daesu),
                    "name": str(row.get("NAME") or "").strip(),
                    "party": party_for_daesu(row.get("DAE"), daesu) or "",
                }
            )
        total = head_blk["head"][0]["list_total_count"] if head_blk else len(out)
        if len(out) >= total or not row_blk["row"]:
            break
        page += 1
    return out


def party_bloc(party: str) -> str:
    """당명 → 대표 진영(블록). 미등록은 원래 당명 유지."""

    p = re.sub(r"\s+", "", str(party or ""))
    return _BLOC.get(p, p)
