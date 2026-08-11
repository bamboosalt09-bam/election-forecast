"""Reusable regional party/bloc priors from repeated election results.

The open-source forecast uses this as a general party-terrain feature. It is
adapted from the statistics-competition regional prior: historical presidential,
assembly proportional, local council, and related party results are combined
with time decay and election-type weights.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


CONSERVATIVE_BLOC = "\uad6d\ubbfc\uc758\ud798"
LIBERAL_BLOC = "\ub354\ubd88\uc5b4\ubbfc\uc8fc\ub2f9"
PROGRESSIVE_BLOC = "\uc9c4\ubcf4\uc815\ub2f9\uacc4"
THIRD_BLOC = "\uc81c3\uc9c0\ub300"
INDEPENDENT_BLOC = "\ubb34\uc18c\uc18d"

BLOC_ALIASES = {
    "\ubbfc\uc8fc\uc790\uc720\ub2f9": CONSERVATIVE_BLOC,
    "\uc2e0\ud55c\uad6d\ub2f9": CONSERVATIVE_BLOC,
    "\ud55c\ub098\ub77c\ub2f9": CONSERVATIVE_BLOC,
    "\uc0c8\ub204\ub9ac\ub2f9": CONSERVATIVE_BLOC,
    "\uc790\uc720\ud55c\uad6d\ub2f9": CONSERVATIVE_BLOC,
    "\ubbf8\ub798\ud1b5\ud569\ub2f9": CONSERVATIVE_BLOC,
    "\ubbf8\ub798\ud55c\uad6d\ub2f9": CONSERVATIVE_BLOC,
    "\uad6d\ubbfc\uc758\ud798": CONSERVATIVE_BLOC,
    "\uad6d\ubbfc\uc758\ubbf8\ub798": CONSERVATIVE_BLOC,
    "\uce5c\ubc15\uc5f0\ub300": CONSERVATIVE_BLOC,
    "\uce5c\ubc15\uc5f0\ud569": CONSERVATIVE_BLOC,
    "\ub300\ud55c\uc560\uad6d\ub2f9": CONSERVATIVE_BLOC,
    "\uc6b0\ub9ac\uacf5\ud654\ub2f9": CONSERVATIVE_BLOC,
    "\uad6d\uac00\uc7ac\uac74\uce5c\ubc15\uc5f0\ud569": CONSERVATIVE_BLOC,
    "\uc790\uc720\ud1b5\uc77c\ub2f9": CONSERVATIVE_BLOC,
    "\ubbfc\uc8fc\ub2f9": LIBERAL_BLOC,
    "\uc0c8\uc815\uce58\uad6d\ubbfc\ud68c\uc758": LIBERAL_BLOC,
    "\uc0c8\ucc9c\ub144\ubbfc\uc8fc\ub2f9": LIBERAL_BLOC,
    "\ub300\ud1b5\ud569\ubbfc\uc8fc\uc2e0\ub2f9": LIBERAL_BLOC,
    "\ud1b5\ud569\ubbfc\uc8fc\ub2f9": LIBERAL_BLOC,
    "\ubbfc\uc8fc\ud1b5\ud569\ub2f9": LIBERAL_BLOC,
    "\uc0c8\uc815\uce58\ubbfc\uc8fc\uc5f0\ud569": LIBERAL_BLOC,
    "\ub354\ubd88\uc5b4\ubbfc\uc8fc\ub2f9": LIBERAL_BLOC,
    "\ub354\ubd88\uc5b4\ubbfc\uc8fc\uc5f0\ud569": LIBERAL_BLOC,
    "\ub354\ubd88\uc5b4\uc2dc\ubbfc\ub2f9": LIBERAL_BLOC,
    "\ub354\ubd88\uc5b4\uc8fc\ubbfc\uc5f0\ud569": LIBERAL_BLOC,
    "\uc5f4\ub9b0\uc6b0\ub9ac\ub2f9": LIBERAL_BLOC,
    "\uc5f4\ub9b0\ubbfc\uc8fc\ub2f9": LIBERAL_BLOC,
    "\uc870\uad6d\ud601\uc2e0\ub2f9": LIBERAL_BLOC,
    "\uc815\uc758\ub2f9": PROGRESSIVE_BLOC,
    "\ubbfc\uc8fc\ub178\ub3d9\ub2f9": PROGRESSIVE_BLOC,
    "\ud1b5\ud569\uc9c4\ubcf4\ub2f9": PROGRESSIVE_BLOC,
    "\uc9c4\ubcf4\uc2e0\ub2f9": PROGRESSIVE_BLOC,
    "\uc9c4\ubcf4\ub2f9": PROGRESSIVE_BLOC,
    "\ub179\uc0c9\uc815\uc758\ub2f9": PROGRESSIVE_BLOC,
    "\ubbfc\uc911\ub2f9": PROGRESSIVE_BLOC,
    "\ub178\ub3d9\ub2f9": PROGRESSIVE_BLOC,
    "\ub179\uc0c9\ub2f9": PROGRESSIVE_BLOC,
    "\uad6d\uc81c\ub179\uc0c9\ub2f9": PROGRESSIVE_BLOC,
    "\uae30\ubcf8\uc18c\ub4dd\ub2f9": PROGRESSIVE_BLOC,
    "\uc9c4\ubcf4\ub2f9\u00b7\uc815\uc758\ub2f9\uacc4": PROGRESSIVE_BLOC,
    "\uad6d\ubbfc\uc758\ub2f9": THIRD_BLOC,
    "\ud1b5\uc77c\uad6d\ubbfc\ub2f9": THIRD_BLOC,
    "\uc2e0\uc815\uce58\uac1c\ud601\ub2f9": THIRD_BLOC,
    "\uad6d\ubbfc\uc2e0\ub2f9": THIRD_BLOC,
    "\uc790\uc720\ubbfc\uc8fc\uc5f0\ud569": THIRD_BLOC,
    "\uc790\uc720\ubbfc\uc8fc\uc5f0\ud569\uc815\ub2f9": THIRD_BLOC,
    "\uad6d\ubbfc\uc0dd\uac01": THIRD_BLOC,
    "\ubc14\ub978\ubbf8\ub798\ub2f9": THIRD_BLOC,
    "\ubbfc\uc0dd\ub2f9": THIRD_BLOC,
    "\ubbfc\uc8fc\ud3c9\ud654\ub2f9": THIRD_BLOC,
    "\uc790\uc720\uc120\uc9c4\ub2f9": THIRD_BLOC,
    "\uad6d\ubbfc\ucc38\uc5ec\ub2f9": THIRD_BLOC,
    "\uc0c8\ub85c\uc6b4\ubbf8\ub798": THIRD_BLOC,
    "\uac1c\ud601\uc2e0\ub2f9": THIRD_BLOC,
    "\ubbf8\ub798\uc5f0\ud569": THIRD_BLOC,
    "\uc81c3\uc9c0\ub300": THIRD_BLOC,
    "\uc81c3\uc9c0\ub300 \ubc0f \uae30\ud0c0": THIRD_BLOC,
    "\ubb34\uc18c\uc18d": INDEPENDENT_BLOC,
}

DEFAULT_ELECTION_TYPE_WEIGHTS = {
    "presidential": 0.8,
    "national_assembly_pr": 1.45,
    "assembly_pr": 1.45,
    "assembly_district": 0.18,
    "metro_council_pr": 0.55,
    "local_council_pr": 0.11,
    "metro_council_district": 0.18,
    "local_council_district": 0.028,
    "metro_governor": 0.004,
    "local_governor": 0.003,
    "education_superintendent": 0.0,
    "education_council": 0.0,
}

PRESIDENTIAL_ELECTION_DATES = {
    1992: "1992-12-18",
    1997: "1997-12-18",
    2002: "2002-12-19",
    2007: "2007-12-19",
    2012: "2012-12-19",
    2017: "2017-05-09",
    2022: "2022-03-09",
    2025: "2025-06-03",
}
ASSEMBLY_ELECTION_DATES = {
    1992: "1992-03-24",
    1996: "1996-04-11",
    2000: "2000-04-13",
    2004: "2004-04-15",
    2008: "2008-04-09",
    2012: "2012-04-11",
    2016: "2016-04-13",
    2020: "2020-04-15",
    2024: "2024-04-10",
}
LOCAL_ELECTION_DATES = {
    1995: "1995-06-27",
    1998: "1998-06-04",
    2002: "2002-06-13",
    2006: "2006-05-31",
    2010: "2010-06-02",
    2014: "2014-06-04",
    2018: "2018-06-13",
    2022: "2022-06-01",
}
LOCAL_ELECTION_MARKERS = (
    "local_governor",
    "metro_governor",
    "local_council",
    "metro_council",
    "education_superintendent",
)


def normalize_bloc(value: object) -> str:
    """Map historical party or bloc labels into stable analytic blocs."""

    text = "" if pd.isna(value) else str(value).strip()
    collapsed = re.sub(r"\s+", "", text)
    return BLOC_ALIASES.get(text, BLOC_ALIASES.get(collapsed, collapsed or text))


def election_year(election_id: object) -> float | None:
    """Extract a year from an election identifier."""

    match = re.search(r"(19|20)\d{2}", str(election_id or ""))
    return float(match.group(0)) if match else None


def election_date(election_id: object) -> pd.Timestamp | None:
    """Infer official election date for known Korean election identifiers."""

    text = str(election_id or "")
    year = election_year(text)
    if year is None:
        return None
    year_int = int(year)
    date_text: str | None = None
    if text.startswith("pres_"):
        date_text = PRESIDENTIAL_ELECTION_DATES.get(year_int)
    elif text.startswith("assembly_"):
        date_text = ASSEMBLY_ELECTION_DATES.get(year_int)
    elif any(marker in text for marker in LOCAL_ELECTION_MARKERS):
        date_text = LOCAL_ELECTION_DATES.get(year_int)
    return pd.Timestamp(date_text) if date_text else None


def load_bloc_history(path: str | Path) -> pd.DataFrame:
    """Load normalized bloc-history rows from a CSV path."""

    csv_path = Path(path)
    if not csv_path.exists():
        return _empty_history()
    frame = pd.read_csv(csv_path)
    required = {"election_id", "election_type", "region_id", "bloc", "vote_share"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"bloc history is missing required columns: {missing}")
    out = frame.copy()
    out["bloc"] = out["bloc"].map(normalize_bloc)
    out["vote_share"] = pd.to_numeric(out["vote_share"], errors="coerce").fillna(0.0)
    if "data_quality_weight" not in out.columns:
        out["data_quality_weight"] = 1.0
    out["data_quality_weight"] = pd.to_numeric(out["data_quality_weight"], errors="coerce").fillna(1.0)
    return out[["election_id", "election_type", "region_id", "bloc", "vote_share", "data_quality_weight"]]


def compute_bloc_base(
    history: pd.DataFrame,
    target_election_id: str,
    half_life_elections: float = 2.0,
    election_type_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate regional bloc vote base from elections before target."""

    columns = ["region_id", "bloc", "bloc_base", "effective_election_count"]
    if history.empty:
        return pd.DataFrame(columns=columns)

    target_date = election_date(target_election_id)
    target_year = election_year(target_election_id)
    frame = history.copy()
    frame["source_date"] = pd.to_datetime(frame["election_id"].map(election_date), errors="coerce")
    frame["source_year"] = frame["election_id"].map(election_year)
    if target_date is not None:
        frame = frame.loc[frame["source_date"].notna() & (frame["source_date"] < target_date)].copy()
    elif target_year is not None:
        frame = frame.loc[frame["source_year"].notna() & (frame["source_year"] < target_year)].copy()
    else:
        return pd.DataFrame(columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["bloc"] = frame["bloc"].map(normalize_bloc)
    if target_date is not None:
        age = (target_date - frame["source_date"]).dt.days / (365.25 * 5.0)
    else:
        age = (target_year - frame["source_year"].astype(float)) / 5.0
    type_weights = election_type_weights or DEFAULT_ELECTION_TYPE_WEIGHTS
    frame["time_weight"] = age.map(lambda value: math.exp(-float(value) / half_life_elections))
    frame["type_weight"] = frame["election_type"].map(type_weights).fillna(0.35)
    frame["weight"] = frame["time_weight"] * frame["type_weight"] * frame["data_quality_weight"]
    frame["weighted_share"] = frame["vote_share"] * frame["weight"]

    grouped = frame.groupby(["region_id", "bloc"], as_index=False).agg(
        weighted_share=("weighted_share", "sum"),
        weight=("weight", "sum"),
        effective_election_count=("election_id", "nunique"),
    )
    grouped["bloc_base"] = grouped["weighted_share"] / grouped["weight"].replace(0, pd.NA)
    grouped["bloc_base"] = grouped["bloc_base"].fillna(0.0).clip(lower=0.0)
    region_sum = grouped.groupby("region_id")["bloc_base"].transform("sum")
    nonzero = region_sum.gt(0)
    grouped.loc[nonzero, "bloc_base"] = grouped.loc[nonzero, "bloc_base"] / region_sum.loc[nonzero]
    return grouped[columns]


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["election_id", "election_type", "region_id", "bloc", "vote_share", "data_quality_weight"]
    )
