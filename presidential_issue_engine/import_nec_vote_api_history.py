"""Build bloc history from NEC VoteXmntckInfoInqireService2.

This importer uses the public data.go.kr counting-result API. It does not store
or print service keys; set DATA_GO_KR_SERVICE_KEY in the environment.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from news_collector.sources.nec_vote_api import count_item_candidate_rows, fetch_all_count_items

try:
    from presidential_issue_engine.build_bloc_history_from_nec import REGION_IDS
    from presidential_issue_engine.region_bloc_prior import normalize_bloc
except ImportError:  # pragma: no cover - supports direct script execution
    from build_bloc_history_from_nec import REGION_IDS  # type: ignore
    from region_bloc_prior import normalize_bloc  # type: ignore


ASSEMBLY_ELECTIONS = {
    1992: "19920324",
    1996: "19960411",
    2000: "20000413",
    2004: "20040415",
    2008: "20080409",
    2012: "20120411",
    2016: "20160413",
    2020: "20200415",
    2024: "20240410",
}

PRESIDENTIAL_ELECTIONS = {
    1992: "19921218",
    1997: "19971218",
    2002: "20021219",
    2007: "20071219",
    2012: "20121219",
    2017: "20170509",
    2022: "20220309",
}

LOCAL_ELECTIONS = {
    1995: "19950627",
    1998: "19980604",
    2002: "20020613",
    2006: "20060531",
    2010: "20100602",
    2014: "20140604",
    2018: "20180613",
    2022: "20220601",
}

LOCAL_RESULT_TYPES = {
    "3": ("metro_governor", "metro_governor_{year}", 0.75, False),
    "4": ("local_governor", "local_governor_{year}", 0.7, False),
    "5": ("metro_council_district", "metro_council_{year}_district", 0.8, False),
    "6": ("local_council_district", "local_council_{year}_district", 0.75, False),
    "8": ("metro_council_pr", "metro_council_{year}_pr", 1.0, False),
    "9": ("local_council_pr", "local_council_{year}_pr", 1.0, False),
    # Education races are non-partisan in the API. They are fetched only if a
    # candidate-to-bloc mapping is added later; otherwise they would not attach
    # to presidential party blocs and could distort region strength.
    "10": ("education_council", "education_council_{year}", 0.25, True),
    "11": ("education_superintendent", "education_superintendent_{year}", 0.25, True),
}

PRESIDENTIAL_TYPE = "1"
ASSEMBLY_DISTRICT_TYPE = "2"
ASSEMBLY_PR_TYPE = "7"
INDEPENDENT_LABELS = {"무소속", "무소속후보자", "계"}


def _candidate_rows(items: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.extend(count_item_candidate_rows(item))
    return pd.DataFrame(rows)


def _items_to_history(
    items: list[dict[str, object]],
    *,
    election_id: str,
    election_type: str,
    data_quality_weight: float,
    include_independent: bool = False,
    include_nonparty: bool = False,
    aggregate_only: bool = True,
) -> pd.DataFrame:
    rows = _candidate_rows(items)
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "election_id",
                "election_type",
                "region_id",
                "bloc",
                "vote_share",
                "data_quality_weight",
            ]
        )

    rows["region_id"] = rows["sdName"].map(REGION_IDS)
    rows = rows.loc[rows["region_id"].notna()].copy()
    if aggregate_only and "wiwName" in rows.columns:
        rows = rows.loc[rows["wiwName"].fillna("").astype(str).str.strip() == "합계"].copy()
    rows["party"] = rows["party"].fillna("").astype(str).str.strip()
    if not include_independent:
        rows = rows.loc[~rows["party"].isin(INDEPENDENT_LABELS)].copy()
    if include_nonparty:
        empty_party = rows["party"] == ""
        rows.loc[empty_party, "party"] = rows.loc[empty_party, "candidate"].fillna("").astype(str)
    rows = rows.loc[rows["party"] != ""].copy()
    rows["bloc"] = rows["party"].map(normalize_bloc)
    rows["votes"] = pd.to_numeric(rows["votes"], errors="coerce").fillna(0.0)
    grouped = rows.groupby(["region_id", "bloc"], as_index=False)["votes"].sum()
    grouped["total_votes"] = grouped.groupby("region_id")["votes"].transform("sum")
    grouped = grouped.loc[grouped["total_votes"] > 0].copy()
    grouped["vote_share"] = grouped["votes"] / grouped["total_votes"]
    grouped["election_id"] = election_id
    grouped["election_type"] = election_type
    grouped["data_quality_weight"] = data_quality_weight
    return grouped[
        [
            "election_id",
            "election_type",
            "region_id",
            "bloc",
            "vote_share",
            "data_quality_weight",
        ]
    ]


def _safe_fetch_count_items(
    sg_id: str,
    sg_typecode: str,
    aggregate_only: bool = True,
) -> list[dict[str, object]]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return fetch_all_count_items(
                sg_id=sg_id,
                sg_typecode=sg_typecode,
                wiw_name="합계" if aggregate_only else None,
            )
        except RuntimeError as exc:
            if "INFO-03" in str(exc):
                return []
            last_error = exc
        except Exception as exc:  # network resets are common on old bulk API pages
            last_error = exc
        time.sleep(1.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return []


def fetch_assembly_history(years: list[int] | None = None, include_district: bool = True) -> pd.DataFrame:
    selected_years = years or sorted(ASSEMBLY_ELECTIONS)
    pieces: list[pd.DataFrame] = []

    for year in selected_years:
        sg_id = ASSEMBLY_ELECTIONS[year]
        pr_items = _safe_fetch_count_items(sg_id, ASSEMBLY_PR_TYPE)
        pieces.append(
            _items_to_history(
                pr_items,
                election_id=f"assembly_{year}_pr",
                election_type="assembly_pr",
                data_quality_weight=1.0,
            )
        )

        if include_district:
            district_items = _safe_fetch_count_items(sg_id, ASSEMBLY_DISTRICT_TYPE)
            pieces.append(
                _items_to_history(
                    district_items,
                    election_id=f"assembly_{year}_district",
                    election_type="assembly_district",
                    data_quality_weight=0.65,
                    include_independent=False,
                )
            )

    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if combined.empty:
        return combined
    return (
        combined.groupby(["election_id", "election_type", "region_id", "bloc"], as_index=False)
        .agg(vote_share=("vote_share", "sum"), data_quality_weight=("data_quality_weight", "max"))
        .sort_values(["election_id", "region_id", "bloc"])
    )


def fetch_presidential_history(years: list[int] | None = None) -> pd.DataFrame:
    selected_years = years or sorted(PRESIDENTIAL_ELECTIONS)
    pieces: list[pd.DataFrame] = []
    for year in selected_years:
        sg_id = PRESIDENTIAL_ELECTIONS[year]
        items = _safe_fetch_count_items(sg_id, PRESIDENTIAL_TYPE)
        pieces.append(
            _items_to_history(
                items,
                election_id=f"pres_{year}",
                election_type="presidential",
                data_quality_weight=1.0,
                include_independent=False,
            )
        )
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if combined.empty:
        return combined
    return (
        combined.groupby(["election_id", "election_type", "region_id", "bloc"], as_index=False)
        .agg(vote_share=("vote_share", "sum"), data_quality_weight=("data_quality_weight", "max"))
        .sort_values(["election_id", "region_id", "bloc"])
    )


def fetch_local_history(
    years: list[int] | None = None,
    include_nonparty_education: bool = True,
) -> pd.DataFrame:
    selected_years = years or sorted(LOCAL_ELECTIONS)
    pieces: list[pd.DataFrame] = []
    for year in selected_years:
        sg_id = LOCAL_ELECTIONS[year]
        for sg_typecode, (
            election_type,
            election_id_template,
            data_quality_weight,
            is_nonparty,
        ) in LOCAL_RESULT_TYPES.items():
            if is_nonparty and not include_nonparty_education:
                continue
            items = _safe_fetch_count_items(sg_id, sg_typecode)
            if not items:
                continue
            pieces.append(
                _items_to_history(
                    items,
                    election_id=election_id_template.format(year=year),
                    election_type=election_type,
                    data_quality_weight=data_quality_weight,
                    include_independent=False,
                    include_nonparty=is_nonparty,
                )
            )
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if combined.empty:
        return combined
    return (
        combined.groupby(["election_id", "election_type", "region_id", "bloc"], as_index=False)
        .agg(vote_share=("vote_share", "sum"), data_quality_weight=("data_quality_weight", "max"))
        .sort_values(["election_id", "region_id", "bloc"])
    )


def fetch_all_history(
    *,
    assembly_years: list[int] | None = None,
    local_years: list[int] | None = None,
    presidential_years: list[int] | None = None,
    include_assembly_district: bool = True,
    include_local: bool = True,
    include_presidential: bool = True,
    include_nonparty_education: bool = True,
) -> pd.DataFrame:
    pieces = [
        fetch_assembly_history(assembly_years, include_district=include_assembly_district)
    ]
    if include_local:
        pieces.append(fetch_local_history(local_years, include_nonparty_education))
    if include_presidential:
        pieces.append(fetch_presidential_history(presidential_years))
    pieces = [piece for piece in pieces if not piece.empty]
    if not pieces:
        return pd.DataFrame()
    combined = pd.concat(pieces, ignore_index=True)
    return (
        combined.groupby(["election_id", "election_type", "region_id", "bloc"], as_index=False)
        .agg(vote_share=("vote_share", "sum"), data_quality_weight=("data_quality_weight", "max"))
        .sort_values(["election_id", "region_id", "bloc"])
    )


def merge_into_history(existing_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    if not existing.empty and not new_rows.empty:
        existing = existing.loc[~existing["election_id"].isin(new_rows["election_id"].unique())].copy()
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = (
        combined.groupby(["election_id", "election_type", "region_id", "bloc"], as_index=False)
        .agg(vote_share=("vote_share", "sum"), data_quality_weight=("data_quality_weight", "max"))
        .sort_values(["election_id", "region_id", "bloc"])
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="presidential_issue_engine/fixed_dataset/bloc_history_results.csv",
        help="history CSV to update",
    )
    parser.add_argument("--assembly-years", nargs="*", type=int, default=sorted(ASSEMBLY_ELECTIONS))
    parser.add_argument("--local-years", nargs="*", type=int, default=sorted(LOCAL_ELECTIONS))
    parser.add_argument(
        "--presidential-years",
        nargs="*",
        type=int,
        default=sorted(PRESIDENTIAL_ELECTIONS),
    )
    parser.add_argument("--years", nargs="*", type=int, help="backward-compatible assembly years")
    parser.add_argument(
        "--no-district",
        action="store_true",
        help="fetch proportional results only",
    )
    parser.add_argument("--no-local", action="store_true", help="skip local election results")
    parser.add_argument("--no-presidential", action="store_true", help="skip presidential results")
    parser.add_argument(
        "--no-education",
        action="store_true",
        help="skip non-party education races",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="write only API-derived rows instead of merging into an existing history CSV",
    )
    args = parser.parse_args()

    assembly_years = args.years if args.years is not None else args.assembly_years
    new_rows = fetch_all_history(
        assembly_years=assembly_years,
        local_years=args.local_years,
        presidential_years=args.presidential_years,
        include_assembly_district=not args.no_district,
        include_local=not args.no_local,
        include_presidential=not args.no_presidential,
        include_nonparty_education=not args.no_education,
    )
    output = Path(args.out)
    result = new_rows if args.replace else merge_into_history(output, new_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"saved: {output}")
    print(f"api rows: {len(new_rows)}")
    print(
        result.groupby(["election_id", "election_type"])["region_id"]
        .nunique()
        .reset_index(name="regions")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
