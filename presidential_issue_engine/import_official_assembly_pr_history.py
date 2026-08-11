"""Import official National Assembly election XLSX files into bloc history.

The input workbooks are NEC election-statistics exports named like:
```
개표현황[제19대][국회의원선거][비례대표국회의원선거].xlsx
```
When available, the 20th Assembly district-vote workbook is also imported as a
low-weight directional prior.

The output schema matches fixed_dataset/bloc_history_results.csv.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

try:
    from presidential_issue_engine.build_bloc_history_from_nec import REGION_IDS, _shares_from_party_votes
except ImportError:  # pragma: no cover - supports direct script execution
    from build_bloc_history_from_nec import REGION_IDS, _shares_from_party_votes  # type: ignore


ASSEMBLY_YEAR_BY_GENERATION = {
    18: 2008,
    19: 2012,
    20: 2016,
    21: 2020,
    22: 2024,
}


def _parse_int(value: object) -> float:
    match = re.search(r"\d[\d,]*", "" if value is None else str(value))
    if not match:
        return 0.0
    return float(match.group(0).replace(",", ""))


def _generation_from_path(path: Path) -> int:
    match = re.search(r"\[(?:\D*)(\d+)(?:\D*)\]", path.name)
    if not match:
        raise ValueError(f"cannot infer assembly generation from {path.name}")
    generation = int(match.group(1))
    if generation not in ASSEMBLY_YEAR_BY_GENERATION:
        raise ValueError(f"unsupported assembly generation {generation}: {path.name}")
    return generation


def official_workbooks(downloads: Path) -> list[Path]:
    files = []
    for path in downloads.glob("*.xlsx"):
        name = path.name
        if (
            "\uac1c\ud45c\ud604\ud669" in name
            and "\uad6d\ud68c\uc758\uc6d0\uc120\uac70" in name
            and "\ube44\ub840\ub300\ud45c\uad6d\ud68c\uc758\uc6d0\uc120\uac70" in name
        ):
            try:
                generation = _generation_from_path(path)
            except ValueError:
                continue
            if generation in ASSEMBLY_YEAR_BY_GENERATION:
                files.append(path)
    return sorted(files, key=_generation_from_path)


def district_2016_workbooks(downloads: Path) -> list[Path]:
    return sorted(
        downloads.glob(
            "\uc81c20\ub300 \uad6d\uc120 \uc9c0\uc5ed\uad6c \ubc0f \ube44\ub840\ub300\ud45c \uc815\ub2f9\ubcc4 \ub4dd\ud45c\uc218 \ud604\ud669*.xlsx"
        )
    )


def parse_official_xlsx(path: Path) -> pd.DataFrame:
    generation = _generation_from_path(path)
    year = ASSEMBLY_YEAR_BY_GENERATION[generation]
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object)

    party_row = 6
    parties: list[tuple[int, str]] = []
    for col in range(3, raw.shape[1]):
        value = raw.iat[party_row, col]
        if pd.isna(value):
            break
        party = str(value).replace("\n", "").strip()
        if not party or party == "\uacc4":
            break
        parties.append((col, party))
    if not parties:
        raise ValueError(f"cannot find party columns in {path}")

    rows: list[dict[str, object]] = []
    for row in range(party_row + 1, raw.shape[0]):
        province = raw.iat[row, 0]
        if pd.isna(province):
            continue
        province_text = str(province).strip()
        if province_text == "\ud569\uacc4" or province_text not in REGION_IDS:
            continue
        for col, party in parties:
            rows.append(
                {
                    "province": province_text,
                    "party": party,
                    "votes": _parse_int(raw.iat[row, col]),
                }
            )

    frame = _shares_from_party_votes(pd.DataFrame(rows), f"assembly_{year}_pr", "assembly_pr")
    frame["data_quality_weight"] = 1.0
    return frame


def parse_2016_district_xlsx(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet_name = next(
        (sheet for sheet in xls.sheet_names if "\uc9c0\uc5ed\uad6c" in sheet),
        xls.sheet_names[0],
    )
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object)

    party_row = 4
    parties: list[tuple[int, str]] = []
    for col in range(4, raw.shape[1]):
        value = raw.iat[party_row, col]
        if pd.isna(value):
            break
        party = str(value).replace("\n", "").strip()
        if not party or party == "\uacc4" or party == "\ubb34\uc18c\uc18d":
            break
        parties.append((col, party))
    if not parties:
        raise ValueError(f"cannot find district party columns in {path}")

    short_region_names = {
        "\uc11c\uc6b8": "\uc11c\uc6b8\ud2b9\ubcc4\uc2dc",
        "\ubd80\uc0b0": "\ubd80\uc0b0\uad11\uc5ed\uc2dc",
        "\ub300\uad6c": "\ub300\uad6c\uad11\uc5ed\uc2dc",
        "\uc778\ucc9c": "\uc778\ucc9c\uad11\uc5ed\uc2dc",
        "\uad11\uc8fc": "\uad11\uc8fc\uad11\uc5ed\uc2dc",
        "\ub300\uc804": "\ub300\uc804\uad11\uc5ed\uc2dc",
        "\uc6b8\uc0b0": "\uc6b8\uc0b0\uad11\uc5ed\uc2dc",
        "\uc138\uc885": "\uc138\uc885\ud2b9\ubcc4\uc790\uce58\uc2dc",
        "\uacbd\uae30": "\uacbd\uae30\ub3c4",
        "\uac15\uc6d0": "\uac15\uc6d0\ub3c4",
        "\ucda9\ubd81": "\ucda9\uccad\ubd81\ub3c4",
        "\ucda9\ub0a8": "\ucda9\uccad\ub0a8\ub3c4",
        "\uc804\ubd81": "\uc804\ub77c\ubd81\ub3c4",
        "\uc804\ub0a8": "\uc804\ub77c\ub0a8\ub3c4",
        "\uacbd\ubd81": "\uacbd\uc0c1\ubd81\ub3c4",
        "\uacbd\ub0a8": "\uacbd\uc0c1\ub0a8\ub3c4",
        "\uc81c\uc8fc": "\uc81c\uc8fc\ud2b9\ubcc4\uc790\uce58\ub3c4",
    }

    rows: list[dict[str, object]] = []
    for row in range(party_row + 1, raw.shape[0]):
        province = raw.iat[row, 0]
        if pd.isna(province):
            continue
        province_text = str(province).strip()
        if province_text in {"\uc804\uad6d", "\ub4dd\ud45c\uc728"}:
            continue
        province_text = short_region_names.get(province_text, province_text)
        if province_text not in REGION_IDS:
            continue
        for col, party in parties:
            rows.append(
                {
                    "province": province_text,
                    "party": party,
                    "votes": _parse_int(raw.iat[row, col]),
                }
            )

    frame = _shares_from_party_votes(
        pd.DataFrame(rows),
        "assembly_2016_district",
        "assembly_district",
    )
    frame["data_quality_weight"] = 0.65
    return frame


def build_official_history(downloads: Path) -> pd.DataFrame:
    files = official_workbooks(downloads)
    if not files:
        raise FileNotFoundError("cannot find official assembly PR XLSX files in Downloads")
    pieces = [parse_official_xlsx(path) for path in files]
    pieces.extend(parse_2016_district_xlsx(path) for path in district_2016_workbooks(downloads)[:1])
    combined = pd.concat(pieces, ignore_index=True)
    combined = (
        combined.groupby(["election_id", "election_type", "region_id", "bloc"], as_index=False)
        .agg(vote_share=("vote_share", "sum"), data_quality_weight=("data_quality_weight", "max"))
        .sort_values(["election_id", "region_id", "bloc"])
    )
    return combined


def merge_into_history(existing_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()
    if not existing.empty:
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
    parser.add_argument("--downloads", default=str(Path(os.environ["USERPROFILE"]) / "Downloads"))
    args = parser.parse_args()

    new_rows = build_official_history(Path(args.downloads))
    output = Path(args.out)
    merged = merge_into_history(output, new_rows)
    merged.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"saved: {output}")
    print(f"official assembly rows: {len(new_rows)}")
    print(
        merged.groupby(["election_id", "election_type"])["region_id"]
        .nunique()
        .reset_index(name="regions")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
