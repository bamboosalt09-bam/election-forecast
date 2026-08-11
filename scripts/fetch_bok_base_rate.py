"""Fetch the Bank of Korea monthly policy rate into a PIT-safe CSV.

The ECOS monthly series is intentionally assigned a month-end availability
date. Policy decisions may be public earlier within the month, but this
conservative convention never assumes knowledge of an intra-month update when
the dated publication record is not retained in the source export.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "presidential_issue_engine/fixed_dataset/interest_rate_indicators.csv"
STAT_CODE = "722Y001"
ITEM_CODE = "0101000"


def _ecos_url(service_key: str, start: int, end: int, start_period: str, end_period: str) -> str:
    parts = [
        "https://ecos.bok.or.kr/api/StatisticSearch",
        quote(service_key, safe=""),
        "json",
        "kr",
        str(start),
        str(end),
        STAT_CODE,
        "M",
        start_period,
        end_period,
        ITEM_CODE,
    ]
    return "/".join(parts)


def _read_page(service_key: str, start: int, end: int, start_period: str, end_period: str) -> dict:
    with urlopen(_ecos_url(service_key, start, end, start_period, end_period), timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "StatisticSearch" not in payload:
        message = payload.get("RESULT", {}).get("MESSAGE", "Unknown ECOS response")
        raise RuntimeError(f"ECOS base-rate request failed: {message}")
    return payload["StatisticSearch"]


def normalize_bok_base_rate_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Normalize ECOS rows and apply the conservative month-end availability rule."""

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "frequency",
                "indicator_name",
                "value",
                "unit",
                "source",
                "available_date",
                "source_note",
            ]
        )
    period = pd.to_datetime(frame["TIME"].astype(str), format="%Y%m", errors="coerce")
    out = pd.DataFrame(
        {
            "period": period + pd.offsets.MonthEnd(0),
            "frequency": "monthly",
            "indicator_name": "bok_base_rate",
            "value": pd.to_numeric(frame["DATA_VALUE"], errors="coerce"),
            "unit": frame["UNIT_NAME"].fillna("annual_percent"),
            "source": "Bank of Korea ECOS",
            "source_note": (
                "ECOS 722Y001/0101000; month-end conservative availability "
                "for point-in-time forecasting"
            ),
        }
    )
    out["available_date"] = out["period"]
    out = out.dropna(subset=["period", "value"]).drop_duplicates("period", keep="last")
    return out.sort_values("period").reset_index(drop=True)


def fetch_bok_base_rate(service_key: str, start_period: str, end_period: str) -> pd.DataFrame:
    """Fetch all pages of the official monthly policy-rate series."""

    first = _read_page(service_key, 1, 10, start_period, end_period)
    total = int(first.get("list_total_count", 0))
    rows = list(first.get("row", []))
    for start in range(11, total + 1, 10):
        page = _read_page(service_key, start, min(start + 9, total), start_period, end_period)
        rows.extend(page.get("row", []))
    if len(rows) != total:
        raise RuntimeError(f"ECOS returned {len(rows)} rows; expected {total}")
    return normalize_bok_base_rate_rows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-key", default=os.getenv("BOK_ECOS_API_KEY", "sample"))
    parser.add_argument("--start-period", default="199901")
    parser.add_argument("--end-period", default=pd.Timestamp.today().strftime("%Y%m"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = fetch_bok_base_rate(args.service_key, args.start_period, args.end_period)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(f"saved {len(frame)} BOK base-rate rows: {args.output}")
    if not frame.empty:
        print(f"period: {frame['period'].min().date()}..{frame['period'].max().date()}")


if __name__ == "__main__":
    main()
