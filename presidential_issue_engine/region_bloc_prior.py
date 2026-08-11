"""Region bloc prior from repeated party-list and presidential results.

This module estimates regional partisan structure from historical vote shares
instead of hand-coding regions such as Honam or TK. It is intended to consume a
combined history table containing presidential, assembly proportional, and local
council proportional results. When that table is not available yet, callers can
fall back to standardized presidential results so the engine remains runnable.
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
STRENGTH_BLOCS = {CONSERVATIVE_BLOC, LIBERAL_BLOC, PROGRESSIVE_BLOC, THIRD_BLOC}

BLOC_ALIASES = {
    "\ubbfc\uc8fc\uc790\uc720\ub2f9": CONSERVATIVE_BLOC,
    "\uc2e0\ud55c\uad6d\ub2f9": CONSERVATIVE_BLOC,
    "\ud55c\ub098\ub77c\ub2f9": CONSERVATIVE_BLOC,
    "\uc0c8\ub204\ub9ac\ub2f9": CONSERVATIVE_BLOC,
    "\uc790\uc720\ud55c\uad6d\ub2f9": CONSERVATIVE_BLOC,
    "\ubbf8\ub798\ud1b5\ud569\ub2f9": CONSERVATIVE_BLOC,
    "\ubbf8\ub798\ud55c\uad6d\ub2f9": CONSERVATIVE_BLOC,
    "\uad6d\ubbfc\uc758\ud798": CONSERVATIVE_BLOC,
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
    "\uc5f4\ub9b0\ubbfc\uc8fc\ub2f9": LIBERAL_BLOC,
    "\uc5f4\ub9b0\uc6b0\ub9ac\ub2f9": LIBERAL_BLOC,
    "\uc870\uad6d\ud601\uc2e0\ub2f9": LIBERAL_BLOC,
    "\uad6d\ubbfc\uc758\ubbf8\ub798": CONSERVATIVE_BLOC,
    "\uce5c\ubc15\uc5f0\ub300": CONSERVATIVE_BLOC,
    "\uce5c\ubc15\uc5f0\ud569": CONSERVATIVE_BLOC,
    "\ub300\ud55c\uc560\uad6d\ub2f9": CONSERVATIVE_BLOC,
    "\uc6b0\ub9ac\uacf5\ud654\ub2f9": CONSERVATIVE_BLOC,
    "\uad6d\uac00\uc7ac\uac74\uce5c\ubc15\uc5f0\ud569": CONSERVATIVE_BLOC,
    "\uc790\uc720\ud1b5\uc77c\ub2f9": CONSERVATIVE_BLOC,
    "\uc815\uc758\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\ub179\uc0c9\uc815\uc758\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\ud1b5\ud569\uc9c4\ubcf4\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\ubbfc\uc8fc\ub178\ub3d9\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\uc9c4\ubcf4\uc2e0\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\ubbfc\uc911\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\ub178\ub3d9\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\ub179\uc0c9\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\uad6d\uc81c\ub179\uc0c9\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\uae30\ubcf8\uc18c\ub4dd\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\uc9c4\ubcf4\ub2f9": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\uc9c4\ubcf4\ub2f9\u00b7\uc815\uc758\ub2f9\uacc4": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
    "\ubbfc\uc0dd\ub2f9": "\uc81c3\uc9c0\ub300",
    "\ud1b5\uc77c\uad6d\ubbfc\ub2f9": "\uc81c3\uc9c0\ub300",
    "\uc2e0\uc815\uce58\uac1c\ud601\ub2f9": "\uc81c3\uc9c0\ub300",
    "\uad6d\ubbfc\uc2e0\ub2f9": "\uc81c3\uc9c0\ub300",
    "\uc790\uc720\ubbfc\uc8fc\uc5f0\ud569": "\uc81c3\uc9c0\ub300",
    "\uc790\uc720\ubbfc\uc8fc\uc5f0\ud569\uc815\ub2f9": "\uc81c3\uc9c0\ub300",
    "\uad6d\ubbfc\uc758\ub2f9": "\uc81c3\uc9c0\ub300",
    "\uad6d\ubbfc\uc0dd\uac01": "\uc81c3\uc9c0\ub300",
    "\ubc14\ub978\ubbf8\ub798\ub2f9": "\uc81c3\uc9c0\ub300",
    "\ubbfc\uc8fc\ud3c9\ud654\ub2f9": "\uc81c3\uc9c0\ub300",
    "\uc790\uc720\uc120\uc9c4\ub2f9": "\uc81c3\uc9c0\ub300",
    "\uad6d\ubbfc\ucc38\uc5ec\ub2f9": "\uc81c3\uc9c0\ub300",
    "\uc0c8\ub85c\uc6b4\ubbf8\ub798": "\uc81c3\uc9c0\ub300",
    "\uac1c\ud601\uc2e0\ub2f9": "\uc81c3\uc9c0\ub300",
    "\ubbf8\ub798\uc5f0\ud569": "\uc81c3\uc9c0\ub300",
    "\uc81c3\uc9c0\ub300": "\uc81c3\uc9c0\ub300",
    "\uc81c3\uc9c0\ub300 \ubc0f \uae30\ud0c0": "\uc81c3\uc9c0\ub300",
}

ELECTION_TYPE_WEIGHTS = {
    "presidential": 0.8,
    "assembly_pr": 1.45,
    "assembly_district": 0.18,
    "metro_council_pr": 0.55,
    "local_council_pr": 0.11,
    "metro_council_district": 0.18,
    "local_council_district": 0.028,
    "metro_governor": 0.004,
    "local_governor": 0.003,
    "education_superintendent": 0.001,
    "education_council": 0.001,
}

# Candidate-centered constituency elections are kept separate from direct
# party ballots. This profile describes regional organization and concentration,
# not national party support.
DISTRICT_TERRAIN_TYPE_WEIGHTS = {
    "presidential": 0.0,
    "assembly_pr": 0.0,
    "assembly_district": 1.0,
    "metro_council_pr": 0.0,
    "local_council_pr": 0.0,
    "metro_council_district": 0.80,
    "local_council_district": 0.50,
    "metro_governor": 0.15,
    "local_governor": 0.10,
    "education_superintendent": 0.0,
    "education_council": 0.0,
}

DEFAULT_HISTORY_PATH = Path("presidential_issue_engine/fixed_dataset/bloc_history_results.csv")

PRESIDENTIAL_ELECTION_DATES = {
    1992: "1992-12-18",
    1997: "1997-12-18",
    2002: "2002-12-19",
    2007: "2007-12-19",
    2012: "2012-12-19",
    2017: "2017-05-09",
    2022: "2022-03-09",
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
    "education_council",
)


def normalize_bloc(value: object) -> str:
    """Map historical party or bloc names into stable analytic blocs."""

    text = "" if pd.isna(value) else str(value).strip()
    collapsed = re.sub(r"\s+", "", text)
    return BLOC_ALIASES.get(text, BLOC_ALIASES.get(collapsed, collapsed or text))


def election_year(election_id: object) -> float | None:
    """Extract the first four-digit year from an election identifier."""

    match = re.search(r"(19|20)\d{2}", str(election_id or ""))
    return float(match.group(0)) if match else None


def election_date(election_id: object) -> pd.Timestamp | None:
    """Infer an official election date for known Korean election identifiers."""

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
    if date_text is None:
        return None
    return pd.Timestamp(date_text)


def presidential_results_to_history(results: pd.DataFrame) -> pd.DataFrame:
    """Convert standardized presidential results into bloc history rows."""

    frame = results.copy()
    frame = frame.loc[frame["slot"] != "alpha"].copy()
    frame["bloc"] = frame["party_name"].map(normalize_bloc)
    frame["vote_share"] = pd.to_numeric(frame["vote_share"], errors="coerce").fillna(0.0)
    history = (
        frame.groupby(["election_id", "region_id", "bloc"], as_index=False)["vote_share"]
        .sum()
        .assign(election_type="presidential", data_quality_weight=1.0)
    )
    return history[
        ["election_id", "election_type", "region_id", "bloc", "vote_share", "data_quality_weight"]
    ]


def load_bloc_history(
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    presidential_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load combined bloc history, falling back to presidential results."""

    path = Path(history_path)
    if path.exists():
        history = pd.read_csv(path)
        if presidential_results is not None:
            presidential_history = presidential_results_to_history(presidential_results)
            if "election_type" in history.columns:
                existing_presidential_ids = set(
                    history.loc[
                        history["election_type"] == "presidential", "election_id"
                    ].astype(str)
                )
                presidential_history = presidential_history.loc[
                    ~presidential_history["election_id"].astype(str).isin(existing_presidential_ids)
                ].copy()
            history = pd.concat(
                [history, presidential_history],
                ignore_index=True,
            )
    elif presidential_results is not None:
        history = presidential_results_to_history(presidential_results)
    else:
        return _empty_history()

    required = {"election_id", "election_type", "region_id", "bloc", "vote_share"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"bloc history is missing required columns: {missing}")

    frame = history.copy()
    frame["bloc"] = frame["bloc"].map(normalize_bloc)
    frame["vote_share"] = pd.to_numeric(frame["vote_share"], errors="coerce").fillna(0.0)
    if "data_quality_weight" not in frame.columns:
        frame["data_quality_weight"] = 1.0
    frame["data_quality_weight"] = pd.to_numeric(
        frame["data_quality_weight"], errors="coerce"
    ).fillna(1.0)
    if "baseline_share" not in frame.columns:
        frame["baseline_share"] = pd.NA
    frame["baseline_share"] = pd.to_numeric(frame["baseline_share"], errors="coerce")
    frame = (
        frame.groupby(["election_id", "election_type", "region_id", "bloc"], as_index=False)
        .agg(
            vote_share=("vote_share", "sum"),
            data_quality_weight=("data_quality_weight", "max"),
            baseline_share=("baseline_share", "mean"),
        )
    )
    return frame[
        [
            "election_id",
            "election_type",
            "region_id",
            "bloc",
            "vote_share",
            "data_quality_weight",
            "baseline_share",
        ]
    ]


def compute_bloc_prior(
    history: pd.DataFrame,
    target_election_id: str,
    election_order: list[str],
    half_life_elections: float = 2.0,
    election_type_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate bloc loyalty and region strength using only prior elections."""

    if history.empty:
        return _empty_prior()

    type_weights = election_type_weights or ELECTION_TYPE_WEIGHTS
    if target_election_id in election_order:
        target_index = election_order.index(target_election_id)
    else:
        target_index = None
    target_year = election_year(target_election_id)
    target_date = election_date(target_election_id)
    order_lookup = {election_id: idx for idx, election_id in enumerate(election_order)}

    frame = history.copy()
    if "baseline_share" not in frame.columns:
        frame["baseline_share"] = pd.NA
    frame["baseline_share"] = pd.to_numeric(frame["baseline_share"], errors="coerce")
    frame["order_index"] = frame["election_id"].map(order_lookup)
    frame["source_year"] = frame["election_id"].map(election_year)
    frame["source_date"] = pd.to_datetime(frame["election_id"].map(election_date), errors="coerce")
    if target_date is not None:
        frame = frame.loc[frame["source_date"].notna() & (frame["source_date"] < target_date)].copy()
    elif target_year is not None:
        frame = frame.loc[frame["source_year"].notna() & (frame["source_year"] < target_year)].copy()
    elif target_index is not None:
        frame = frame.loc[frame["order_index"].notna() & (frame["order_index"] < target_index)].copy()
    else:
        return _empty_prior()
    if frame.empty:
        return _empty_prior()

    region_counts = (
        frame.groupby("election_id", as_index=False)["region_id"]
        .nunique()
        .rename(columns={"region_id": "region_count"})
    )
    frame = frame.merge(region_counts, on="election_id", how="left")
    full_region = (frame["region_count"] >= 10) | (frame["election_type"] == "presidential")
    national = (
        frame.loc[full_region]
        .groupby(["election_id", "bloc"], as_index=False)["vote_share"]
        .mean()
        .rename(columns={"vote_share": "national_share"})
    )
    frame = frame.merge(national, on=["election_id", "bloc"], how="left")
    partial_baseline = frame["baseline_share"].fillna(0.25)
    frame["national_share"] = frame["national_share"].where(full_region, partial_baseline)
    frame["bloc_lean"] = frame["vote_share"] - frame["national_share"]
    if target_date is not None:
        age = (target_date - frame["source_date"]).dt.days / (365.25 * 5.0)
    elif target_year is not None:
        age = (target_year - frame["source_year"].astype(float)) / 5.0
    else:
        age = target_index - frame["order_index"].astype(float)
    frame["time_weight"] = age.map(lambda value: math.exp(-float(value) / half_life_elections))
    frame["type_weight"] = frame["election_type"].map(type_weights).fillna(0.5)
    frame = frame.loc[frame["type_weight"] > 0.0].copy()
    if frame.empty:
        return _empty_prior()
    frame["weight"] = frame["time_weight"] * frame["type_weight"] * frame["data_quality_weight"]
    frame["weighted_lean"] = frame["bloc_lean"] * frame["weight"]

    loyalty = frame.groupby(["region_id", "bloc"], as_index=False).agg(
        weighted_lean=("weighted_lean", "sum"),
        weight=("weight", "sum"),
        effective_election_count=("election_id", "nunique"),
    )
    loyalty["bloc_loyalty"] = loyalty["weighted_lean"] / loyalty["weight"].replace(0, pd.NA)
    loyalty["bloc_loyalty"] = loyalty["bloc_loyalty"].fillna(0.0)
    strength_source = loyalty.loc[loyalty["bloc"].isin(STRENGTH_BLOCS)].copy()
    if strength_source.empty:
        strength_source = loyalty
    strength = (
        strength_source.groupby("region_id", as_index=False)["bloc_loyalty"]
        .apply(lambda values: values.abs().max())
        .rename(columns={"bloc_loyalty": "bloc_strength"})
    )
    out = loyalty.merge(strength, on="region_id", how="left")
    out["partisan_prior"] = out["bloc_loyalty"] * (1.0 + out["bloc_strength"].fillna(0.0))
    out["target_election_id"] = target_election_id
    return out[
        [
            "target_election_id",
            "region_id",
            "bloc",
            "bloc_loyalty",
            "bloc_strength",
            "partisan_prior",
            "effective_election_count",
        ]
    ]


def attach_bloc_prior(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    election_order: list[str],
) -> pd.DataFrame:
    """Attach a prior row to each election-region-slot observation."""

    if frame.empty:
        return frame.copy()

    slot_blocs = frame[["election_id", "slot", "bloc"]].drop_duplicates().copy()
    pieces: list[pd.DataFrame] = []
    for election_id in frame["election_id"].drop_duplicates():
        prior = compute_bloc_prior(history, str(election_id), election_order)
        if prior.empty:
            continue
        bloc_rows = slot_blocs.loc[slot_blocs["election_id"] == election_id, ["slot", "bloc"]]
        piece = prior.merge(bloc_rows, on="bloc", how="inner")
        piece["election_id"] = election_id
        pieces.append(piece)

    result = frame.copy()
    if not pieces:
        result["bloc_loyalty"] = 0.0
        result["bloc_strength"] = 0.0
        result["partisan_prior"] = 0.0
        result["effective_election_count"] = 0.0
        return result

    prior_frame = pd.concat(pieces, ignore_index=True)
    result = result.merge(
        prior_frame[
            [
                "election_id",
                "region_id",
                "slot",
                "bloc_loyalty",
                "bloc_strength",
                "partisan_prior",
                "effective_election_count",
            ]
        ],
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in ["bloc_loyalty", "bloc_strength", "partisan_prior", "effective_election_count"]:
        result[column] = result[column].fillna(0.0)
    return result


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "election_id",
            "election_type",
            "region_id",
            "bloc",
            "vote_share",
            "data_quality_weight",
            "baseline_share",
        ]
    )


def _empty_prior() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "target_election_id",
            "region_id",
            "bloc",
            "bloc_loyalty",
            "bloc_strength",
            "partisan_prior",
            "effective_election_count",
        ]
    )
