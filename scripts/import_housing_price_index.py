"""Import KOSIS apartment transaction price index workbook.

The workbook is a wide, quarter-by-quarter KOSIS export. This script preserves
the city/county/district rows and also writes a province-level average aligned
with the presidential model's current ``sido_*`` region IDs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROVINCE_MAP = {
    "서울": ("sido_11", "서울특별시"),
    "부산": ("sido_26", "부산광역시"),
    "대구": ("sido_27", "대구광역시"),
    "인천": ("sido_28", "인천광역시"),
    "광주": ("sido_29", "광주광역시"),
    "대전": ("sido_30", "대전광역시"),
    "울산": ("sido_31", "울산광역시"),
    "세종": ("sido_36", "세종특별자치시"),
    "경기": ("sido_41", "경기도"),
    "강원": ("sido_42", "강원특별자치도"),
    "충북": ("sido_43", "충청북도"),
    "충남": ("sido_44", "충청남도"),
    "전북": ("sido_45", "전북특별자치도"),
    "전남": ("sido_46", "전라남도"),
    "경북": ("sido_47", "경상북도"),
    "경남": ("sido_48", "경상남도"),
    "제주": ("sido_50", "제주특별자치도"),
}


def import_housing_price_index(workbook_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Normalize a KOSIS housing-price workbook into SGG and SIDO CSVs."""

    raw = pd.read_excel(workbook_path, sheet_name=0)
    province_col, sgg_col = raw.columns[:2]
    raw[province_col] = raw[province_col].ffill()

    period_columns = [column for column in raw.columns[2:] if _parse_quarter(str(column)) is not None]
    long = raw.melt(
        id_vars=[province_col, sgg_col],
        value_vars=period_columns,
        var_name="quarter_label",
        value_name="value",
    )
    long = long.rename(columns={province_col: "province_short", sgg_col: "sgg_name"})
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value", "province_short", "sgg_name"])
    long["region_id"] = long["province_short"].map(lambda value: PROVINCE_MAP.get(value, ("", ""))[0])
    long["province"] = long["province_short"].map(lambda value: PROVINCE_MAP.get(value, ("", ""))[1])
    long = long.loc[long["region_id"] != ""].copy()

    quarters = long["quarter_label"].map(lambda value: _parse_quarter(str(value)))
    long["period"] = quarters.map(lambda item: item[0])
    long["year"] = quarters.map(lambda item: item[1])
    long["quarter"] = quarters.map(lambda item: item[2])
    long["available_date"] = (pd.to_datetime(long["period"]) + pd.Timedelta(days=60)).dt.date.astype(str)
    long["indicator_name"] = "apartment_transaction_price_index"
    long["unit"] = "index_2017q4_100"
    long["source"] = "Korea Real Estate Board / KOSIS"
    long["source_note"] = "KOSIS DT_KAB_11672_S5; apartment transaction price index by SGG"

    sgg = long[
        [
            "region_id",
            "province",
            "province_short",
            "sgg_name",
            "period",
            "year",
            "quarter",
            "indicator_name",
            "value",
            "unit",
            "source",
            "available_date",
            "source_note",
        ]
    ].sort_values(["region_id", "sgg_name", "period"])
    sgg = _attach_yoy(sgg, ["region_id", "sgg_name"])

    sido = (
        sgg.groupby(["region_id", "province", "period", "year", "quarter"], as_index=False)["value"]
        .mean()
        .sort_values(["region_id", "period"])
    )
    sido["indicator_name"] = "apartment_transaction_price_index"
    sido["unit"] = "index_2017q4_100"
    sido["source"] = "Korea Real Estate Board / KOSIS"
    sido["available_date"] = (pd.to_datetime(sido["period"]) + pd.Timedelta(days=60)).dt.date.astype(str)
    sido["source_note"] = "KOSIS DT_KAB_11672_S5; SIDO mean of available SGG rows"
    sido = _attach_yoy(sido, ["region_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    sgg_path = output_dir / "housing_price_index_sgg.csv"
    sido_path = output_dir / "housing_price_index_sido.csv"
    sgg.to_csv(sgg_path, index=False, encoding="utf-8-sig")
    sido.to_csv(sido_path, index=False, encoding="utf-8-sig")
    return sgg_path, sido_path


def _attach_yoy(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Attach same-quarter year-over-year percentage change."""

    out = frame.copy()
    previous = out[keys + ["year", "quarter", "value"]].copy()
    previous["year"] = previous["year"] + 1
    previous = previous.rename(columns={"value": "previous_year_value"})
    out = out.merge(previous, on=keys + ["year", "quarter"], how="left")
    out["yoy_change_pct"] = ((out["value"] / out["previous_year_value"]) - 1.0) * 100.0
    out["yoy_change_pct"] = out["yoy_change_pct"].where(out["previous_year_value"].notna(), pd.NA)
    return out.drop(columns=["previous_year_value"])


def _parse_quarter(label: str) -> tuple[str, int, int] | None:
    match = re.fullmatch(r"(\d{4})\.(\d)/4", label)
    if not match:
        return None
    year = int(match.group(1))
    quarter = int(match.group(2))
    month = quarter * 3
    period = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return period.date().isoformat(), year, quarter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("presidential_issue_engine/fixed_dataset"),
    )
    args = parser.parse_args()
    sgg_path, sido_path = import_housing_price_index(args.workbook, args.output_dir)
    print(f"wrote {sgg_path}")
    print(f"wrote {sido_path}")


if __name__ == "__main__":
    main()
