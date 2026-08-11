"""Convert the user-provided KOSPI daily text export to an auditable CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "presidential_issue_engine/fixed_dataset/kospi_daily.csv"
DATE_PATTERN = re.compile(
    r"^(?P<year>\d{4})\uB144\s*(?P<month>\d{2})\uC6D4\s*(?P<day>\d{2})\uC77C\t(?P<values>.+)$"
)


def _number(value: str) -> float:
    return float(value.replace(",", "").replace("%", "").replace("+", "").strip())


def _volume(value: str) -> float:
    text = value.strip().upper().replace(",", "")
    multiplier = 1.0
    if text.endswith("B"):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("K"):
        multiplier = 1_000.0
        text = text[:-1]
    return float(text) * multiplier


def parse_kospi_text(text: str, source_name: str) -> pd.DataFrame:
    """Parse daily OHLCV rows while ignoring export summary lines."""

    rows: list[dict[str, object]] = []
    for raw_line in text.splitlines():
        match = DATE_PATTERN.match(raw_line.strip())
        if not match:
            continue
        values = match.group("values").split("\t")
        if len(values) != 6:
            raise ValueError(f"Unexpected KOSPI column count: {raw_line[:120]}")
        date = pd.Timestamp(
            year=int(match.group("year")),
            month=int(match.group("month")),
            day=int(match.group("day")),
        )
        close, open_value, high, low = [_number(value) for value in values[:4]]
        rows.append(
            {
                "date": date,
                "close": close,
                "open": open_value,
                "high": high,
                "low": low,
                "volume": _volume(values[4]),
                "daily_change_pct": _number(values[5]),
                "available_date": date,
                "source": "user_provided_kospi_history",
                "source_note": source_name,
            }
        )
    if not rows:
        raise ValueError("No KOSPI daily rows were parsed")

    frame = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    numeric = ["close", "open", "high", "low", "volume", "daily_change_pct"]
    if frame[numeric].isna().any().any():
        raise ValueError("KOSPI data contains invalid numeric values")
    if (frame[["close", "open", "high", "low", "volume"]] < 0).any().any():
        raise ValueError("KOSPI data contains negative price or volume values")
    invalid_ohlc = (
        frame["high"].lt(frame[["open", "close", "low"]].max(axis=1))
        | frame["low"].gt(frame[["open", "close", "high"]].min(axis=1))
    )
    frame["ohlc_quality_flag"] = "ok"
    frame.loc[invalid_ohlc, "ohlc_quality_flag"] = "source_range_inconsistent"
    return frame.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.source.read_text(encoding="utf-8-sig")
    frame = parse_kospi_text(text, args.source.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(
        f"saved {len(frame)} KOSPI rows: {args.output} "
        f"({frame['date'].min().date()}..{frame['date'].max().date()})"
    )
    inconsistent = int(frame["ohlc_quality_flag"].ne("ok").sum())
    if inconsistent:
        print(f"warning: preserved {inconsistent} source rows with inconsistent OHLC ranges")


if __name__ == "__main__":
    main()
