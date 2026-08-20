"""Continuous split strength for a third candidate's vehicle.

The binary ``major_split_lineage`` flag separates the V24 scored panel cleanly,
but it cannot rank vehicles inside a class: 개혁신당 and 국민의당 both normalise
to 제3지대 although one was formed by four defecting incumbents and the other by
a large 새정치민주연합 split.

This module expresses split strength continuously as the number of sitting
members of the National Assembly who left a governing or main opposition party
for the new vehicle before the presidential contest, normalised by the chamber
size of that Assembly:

    defection_scale = defection_seats / assembly_size

Data status
-----------
Two derivations are available. Cross-term name matching in
``data/assembly_roster.csv`` counts winners under the new vehicle who held a
major-party seat in the preceding Assembly, which covers 민주노동당, 정의당, and
국민의당. ``data/raw/assembly_member_history.csv`` records mid-term affiliation
for the twenty-first Assembly, which covers 개혁신당. Every party-backed third
candidate from 2002 onward therefore has a derived count.

Roster coverage begins at the sixteenth Assembly, so 통일국민당 1992 and
국민신당 1997 remain ``needs_source`` and are returned as missing rather than
guessed; both are warmup-only and do not enter scoring. Independents carry
``no_party`` because a seat count cannot express personal lineage — 이회창 2007
has zero party-level carry-in alongside a strong 한나라당 leadership lineage.

The module reports coverage and scale; the ceiling in
``third_candidate_lineage_constraint`` consumes the same table directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINEAGE_TABLE = (
    ROOT / "presidential_issue_engine" / "fixed_dataset" / "v24" / "third_candidate_lineage.csv"
)
MEMBER_HISTORY = ROOT / "data" / "raw" / "assembly_member_history.csv"

ASSEMBLY_SIZE = {
    "pres_1992": 299,
    "pres_1997": 299,
    "pres_2002": 273,
    "pres_2007": 299,
    "pres_2012": 299,
    "pres_2017": 300,
    "pres_2022": 300,
    "pres_2025": 300,
}

MAJOR_KEYS = (
    "한나라당", "새누리당", "국민의힘", "미래통합당", "신한국당", "민자당",
    "새천년민주당", "열린우리당", "대통합민주신당", "민주통합당",
    "새정치민주연합", "더불어민주당",
)


def derive_defection_seats(party_name: str) -> int | None:
    """Count 21st-Assembly incumbents holding this party's affiliation mid-term.

    Returns ``None`` when the repository cannot support the count, which is the
    case for every Assembly before the twenty-first.
    """

    if not MEMBER_HISTORY.exists():
        return None
    frame = pd.read_csv(MEMBER_HISTORY, encoding="utf-8-sig", dtype=str)
    if "party" not in frame.columns:
        return None
    matched = frame.loc[frame["party"].astype(str).str.strip().eq(str(party_name))]
    if matched.empty:
        return None
    return int(len(matched))


def load_scale(path: Path | str | None = None) -> pd.DataFrame:
    """Return the lineage table with a normalised defection scale where available."""

    source = Path(path) if path is not None else LINEAGE_TABLE
    frame = pd.read_csv(source, encoding="utf-8-sig")
    frame["defection_seats"] = pd.to_numeric(frame.get("defection_seats"), errors="coerce")
    frame["assembly_size"] = frame["election_id"].map(ASSEMBLY_SIZE)
    frame["defection_scale"] = frame["defection_seats"] / frame["assembly_size"]
    frame["scale_available"] = frame["defection_scale"].notna()
    return frame


def coverage_report(path: Path | str | None = None) -> pd.DataFrame:
    """Summarise which rows carry a usable scale and which still need a source."""

    frame = load_scale(path)
    columns = [
        "election_id",
        "candidate_name",
        "party_name",
        "major_split_lineage",
        "defection_seats",
        "defection_seats_source",
        "defection_scale",
        "scale_available",
    ]
    return frame[columns]
