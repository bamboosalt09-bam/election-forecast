"""Build bloc_history_results.csv from NEC proportional election result files.

The input is the NEC zip containing National Assembly proportional CSV rows and
local-council proportional Excel workbooks. Output rows are province-level
region bloc vote shares that can feed region_bloc_prior.py.
"""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

try:
    from presidential_issue_engine.region_bloc_prior import normalize_bloc
except ImportError:  # pragma: no cover - supports direct script execution
    from region_bloc_prior import normalize_bloc  # type: ignore


REGION_IDS = {
    "\uc11c\uc6b8\ud2b9\ubcc4\uc2dc": "sido_11",
    "\ubd80\uc0b0\uad11\uc5ed\uc2dc": "sido_26",
    "\ub300\uad6c\uad11\uc5ed\uc2dc": "sido_27",
    "\uc778\ucc9c\uad11\uc5ed\uc2dc": "sido_28",
    "\uad11\uc8fc\uad11\uc5ed\uc2dc": "sido_29",
    "\ub300\uc804\uad11\uc5ed\uc2dc": "sido_30",
    "\uc6b8\uc0b0\uad11\uc5ed\uc2dc": "sido_31",
    "\uc138\uc885\ud2b9\ubcc4\uc790\uce58\uc2dc": "sido_36",
    "\uacbd\uae30\ub3c4": "sido_41",
    "\uac15\uc6d0\ub3c4": "sido_42",
    "\uac15\uc6d0\ud2b9\ubcc4\uc790\uce58\ub3c4": "sido_42",
    "\ucda9\uccad\ubd81\ub3c4": "sido_43",
    "\ucda9\uccad\ub0a8\ub3c4": "sido_44",
    "\uc804\ub77c\ubd81\ub3c4": "sido_45",
    "\uc804\ubd81\ud2b9\ubcc4\uc790\uce58\ub3c4": "sido_45",
    "\uc804\ub77c\ub0a8\ub3c4": "sido_46",
    "\uacbd\uc0c1\ubd81\ub3c4": "sido_47",
    "\uacbd\uc0c1\ub0a8\ub3c4": "sido_48",
    "\uc81c\uc8fc\ub3c4": "sido_50",
    "\uc81c\uc8fc\ud2b9\ubcc4\uc790\uce58\ub3c4": "sido_50",
}

ADMIN_LABELS = {
    "\uc120\uac70\uc778\uc218",
    "\ud22c\ud45c\uc218",
    "\uae30\uad8c\uc790\uc218",
    "\ubb34\ud6a8 \ud22c\ud45c\uc218",
    "\ubb34\ud6a8\ud22c\ud45c\uc218",
}
SUM_LABELS = {"\ud569\uacc4", "\uacc4"}
PROVINCE_HEADER_NAMES = {"\uc2dc\ub3c4", "\uc2dc\ub3c4\uba85"}
VOTE_MARKER = "\ub4dd\ud45c\uc218"
PARTY_PREFIX = "\uc815\ub2f9"


def _number(value: object) -> float:
    """Parse NEC numeric cells containing comma separators."""

    parsed = pd.to_numeric(str(value).replace(",", ""), errors="coerce")
    return 0.0 if pd.isna(parsed) else float(parsed)


def _extract_year(name: str) -> str:
    """Return the first four-digit election year in a filename."""

    match = re.search(r"(20\d{2}|19\d{2})", name)
    if not match:
        raise ValueError(f"cannot infer election year from {name}")
    return match.group(1)


def _find_header_col(raw: pd.DataFrame, names: set[str]) -> int | None:
    """Find a column whose first-row header is one of names."""

    for col, value in enumerate(raw.iloc[0].fillna("").astype(str)):
        if value.strip() in names:
            return col
    return None


def _shares_from_party_votes(
    frame: pd.DataFrame,
    election_id: str,
    election_type: str,
) -> pd.DataFrame:
    """Normalize party votes into region-bloc vote shares."""

    data = frame.copy()
    data["region_id"] = data["province"].map(REGION_IDS)
    data = data.loc[data["region_id"].notna()].copy()
    data["bloc"] = data["party"].map(normalize_bloc)
    data["votes"] = pd.to_numeric(data["votes"], errors="coerce").fillna(0.0)
    grouped = data.groupby(["region_id", "bloc"], as_index=False)["votes"].sum()
    grouped["total_votes"] = grouped.groupby("region_id")["votes"].transform("sum")
    grouped = grouped.loc[grouped["total_votes"] > 0].copy()
    grouped["vote_share"] = grouped["votes"] / grouped["total_votes"]
    grouped["election_id"] = election_id
    grouped["election_type"] = election_type
    grouped["data_quality_weight"] = 1.0
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


def parse_assembly_pr_csv(zip_file: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Parse the 2024 National Assembly proportional CSV."""

    year = _extract_year(name)
    with zip_file.open(name) as handle:
        raw = pd.read_csv(handle, encoding="cp949")

    frame = raw.iloc[:, [0, 4, 5]].copy()
    frame.columns = ["province", "party", "votes"]
    frame = frame.loc[~frame["party"].isin(ADMIN_LABELS)].copy()
    return _shares_from_party_votes(frame, f"assembly_{year}_pr", "assembly_pr")


def _local_party_columns(raw: pd.DataFrame) -> tuple[int, int, list[tuple[int, str]]]:
    """Find province, party-header row, and party vote columns in a local workbook."""

    marker_col = None
    for col, value in enumerate(raw.iloc[0].fillna("").astype(str)):
        if VOTE_MARKER in value:
            marker_col = col
            break
    if marker_col is None:
        raise ValueError("cannot find party vote column marker")

    party_row = 2 if str(raw.iat[1, marker_col]).strip().startswith(PARTY_PREFIX) else 1
    province_col = _find_header_col(raw, PROVINCE_HEADER_NAMES)
    if province_col is None:
        province_col = 0

    parties: list[tuple[int, str]] = []
    for col in range(marker_col, raw.shape[1]):
        top = "" if pd.isna(raw.iat[0, col]) else str(raw.iat[0, col]).strip()
        party = "" if pd.isna(raw.iat[party_row, col]) else str(raw.iat[party_row, col]).strip()
        party = party.replace("_x000D_", "").strip()
        if (
            not party
            or party in SUM_LABELS
            or party in ADMIN_LABELS
            or top in SUM_LABELS
            or "\ubb34\ud6a8" in top
            or "\uae30\uad8c" in top
        ):
            break
        parties.append((col, party))

    if not parties:
        raise ValueError("cannot find local council party columns")
    return province_col, party_row, parties


def parse_local_council_pr_xlsx(zip_file: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Parse a local-election workbook's metropolitan council PR sheet."""

    year = _extract_year(name)
    temp_path = Path(tempfile.gettempdir()) / f"nec_local_pr_{abs(hash(name))}.xlsx"
    temp_path.write_bytes(zip_file.read(name))
    raw = pd.read_excel(temp_path, sheet_name=4, header=None, dtype=object)
    province_col, party_row, parties = _local_party_columns(raw)

    body = raw.iloc[party_row + 1 :].copy()
    geo_cols = list(range(min(parties[0][0], body.shape[1])))
    total_mask = body[geo_cols].apply(
        lambda row: any(str(value).strip() in SUM_LABELS for value in row),
        axis=1,
    )
    total_rows = body.loc[total_mask].copy()
    if total_rows.empty:
        raise ValueError(f"cannot find local council total rows in {name}")

    rows: list[dict[str, object]] = []
    for province, province_rows in total_rows.groupby(total_rows.iloc[:, province_col]):
        province_text = str(province).strip()
        if not province_text:
            continue
        for col, party in parties:
            votes = province_rows.iloc[:, col].map(_number).sum()
            rows.append({"province": province_text, "party": party, "votes": votes})

    return _shares_from_party_votes(
        pd.DataFrame(rows),
        f"local_council_{year}_pr",
        "local_council_pr",
    )


def build_bloc_history(zip_path: str | Path) -> pd.DataFrame:
    """Build the combined bloc history table from a NEC zip file."""

    pieces: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zip_file:
        seen_local_years: set[str] = set()
        for name in zip_file.namelist():
            lowered = name.lower()
            if lowered.endswith(".csv"):
                pieces.append(parse_assembly_pr_csv(zip_file, name))
            elif lowered.endswith(".xlsx"):
                year = _extract_year(name)
                if year in seen_local_years:
                    continue
                seen_local_years.add(year)
                pieces.append(parse_local_council_pr_xlsx(zip_file, name))

    if not pieces:
        raise ValueError(f"no supported NEC result files found in {zip_path}")

    combined = pd.concat(pieces, ignore_index=True)
    combined = (
        combined.groupby(["election_id", "election_type", "region_id", "bloc"], as_index=False)
        .agg(vote_share=("vote_share", "sum"), data_quality_weight=("data_quality_weight", "max"))
        .sort_values(["election_id", "region_id", "bloc"])
    )
    return combined[
        [
            "election_id",
            "election_type",
            "region_id",
            "bloc",
            "vote_share",
            "data_quality_weight",
        ]
    ]


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", required=True, help="NEC proportional results zip")
    parser.add_argument(
        "--out",
        default="presidential_issue_engine/fixed_dataset/bloc_history_results.csv",
        help="output CSV path",
    )
    args = parser.parse_args()

    result = build_bloc_history(args.zip_path)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"saved: {output_path}")
    print(f"rows: {len(result)}")
    print(
        result.groupby(["election_id", "election_type"])["region_id"]
        .nunique()
        .reset_index(name="regions")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
