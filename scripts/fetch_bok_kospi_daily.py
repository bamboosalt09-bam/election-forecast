"""Fetch the official daily KOSPI closing index from Bank of Korea ECOS.

ECOS table 802Y001 is the Bank of Korea's daily stock-market table.  Item
0001000 is the KOSPI closing index and identifies the Korea Exchange as the
source organization.  Only the closing index is populated because the model
does not use unofficially reconstructed OHLCV fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/raw/official_sources/cache/bok_ecos_kospi_daily.csv"
DEFAULT_MANIFEST = ROOT / "data/raw/official_sources/bok_ecos_kospi_manifest.json"
API_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"
STAT_CODE = "802Y001"
ITEM_CODE = "0001000"
FREQUENCY = "D"
SOURCE_NAME = "Bank of Korea ECOS"
SOURCE_NOTE = (
    "ECOS 802Y001/0001000 daily KOSPI closing index; "
    "source organization: Korea Exchange"
)


def _ecos_url(
    service_key: str,
    start: int,
    end: int,
    start_period: str,
    end_period: str,
) -> str:
    parts = [
        API_BASE,
        quote(service_key, safe=""),
        "json",
        "kr",
        str(start),
        str(end),
        STAT_CODE,
        FREQUENCY,
        start_period,
        end_period,
        ITEM_CODE,
    ]
    return "/".join(parts)


def _read_page(
    service_key: str,
    start: int,
    end: int,
    start_period: str,
    end_period: str,
    *,
    retries: int = 4,
) -> dict[str, object]:
    url = _ecos_url(service_key, start, end, start_period, end_period)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "StatisticSearch" not in payload:
                message = payload.get("RESULT", {}).get("MESSAGE", "Unknown ECOS response")
                raise RuntimeError(f"ECOS KOSPI request failed: {message}")
            return payload["StatisticSearch"]
        except Exception as exc:  # pragma: no cover - exercised only on network failure
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"ECOS KOSPI request failed after {retries} attempts: {last_error}")


def normalize_bok_kospi_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Normalize ECOS daily KOSPI rows into the engine's PIT-safe schema."""

    columns = [
        "date",
        "close",
        "open",
        "high",
        "low",
        "volume",
        "daily_change_pct",
        "available_date",
        "source",
        "source_note",
        "ohlc_quality_flag",
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"TIME", "DATA_VALUE"}
    if not required.issubset(frame.columns):
        raise ValueError(f"ECOS KOSPI rows are missing fields: {sorted(required - set(frame.columns))}")

    date = pd.to_datetime(frame["TIME"].astype(str), format="%Y%m%d", errors="coerce")
    close = pd.to_numeric(frame["DATA_VALUE"], errors="coerce")
    out = pd.DataFrame({"date": date, "close": close})
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates("date", keep="last").reset_index(drop=True)
    if out.empty:
        raise ValueError("ECOS KOSPI response contained no valid observations")
    if out["close"].le(0.0).any():
        raise ValueError("ECOS KOSPI response contained a non-positive closing index")

    out["open"] = pd.NA
    out["high"] = pd.NA
    out["low"] = pd.NA
    out["volume"] = pd.NA
    out["daily_change_pct"] = out["close"].pct_change().mul(100.0)
    out["available_date"] = out["date"]
    out["source"] = SOURCE_NAME
    out["source_note"] = SOURCE_NOTE
    out["ohlc_quality_flag"] = "official_close_only"
    return out[columns]


def fetch_bok_kospi_daily(
    service_key: str,
    start_period: str,
    end_period: str,
    *,
    throttle_seconds: float = 0.0,
) -> pd.DataFrame:
    """Fetch all official daily KOSPI observations in the requested period."""

    page_size = 10 if service_key.lower() == "sample" else 1000
    first = _read_page(service_key, 1, page_size, start_period, end_period)
    total = int(first.get("list_total_count", 0))
    rows = list(first.get("row", []))
    for start in range(page_size + 1, total + 1, page_size):
        end = min(start + page_size - 1, total)
        page = _read_page(service_key, start, end, start_period, end_period)
        rows.extend(page.get("row", []))
        if throttle_seconds > 0.0:
            time.sleep(throttle_seconds)
    if len(rows) != total:
        raise RuntimeError(f"ECOS returned {len(rows)} KOSPI rows; expected {total}")
    return normalize_bok_kospi_rows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_manifest(
    path: Path,
    csv_path: Path,
    frame: pd.DataFrame,
    start_period: str,
    end_period: str,
) -> None:
    """Write machine-readable provenance without storing an API credential."""

    payload = {
        "dataset": "KOSPI daily closing index",
        "provider": "Bank of Korea ECOS",
        "original_source_organization": "Korea Exchange",
        "api_documentation": "https://ecos.bok.or.kr/api/",
        "api_service": "StatisticSearch",
        "stat_code": STAT_CODE,
        "item_code": ITEM_CODE,
        "frequency": FREQUENCY,
        "requested_start": start_period,
        "requested_end": end_period,
        "first_observation": frame["date"].min().date().isoformat(),
        "last_observation": frame["date"].max().date().isoformat(),
        "row_count": int(len(frame)),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": csv_path.resolve().relative_to(ROOT.resolve()).as_posix()
        if csv_path.resolve().is_relative_to(ROOT.resolve())
        else str(csv_path.resolve()),
        "csv_sha256": _sha256(csv_path),
        "availability_rule": "same trading date; election forecasts use the engine cutoff",
        "fields_used": ["date", "close"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-key", default=os.getenv("BOK_ECOS_API_KEY", "sample"))
    parser.add_argument("--start-period", default="19950103")
    parser.add_argument("--end-period", default="20221231")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--throttle-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = fetch_bok_kospi_daily(
        args.service_key,
        args.start_period,
        args.end_period,
        throttle_seconds=max(args.throttle_seconds, 0.0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    write_manifest(args.manifest, args.output, frame, args.start_period, args.end_period)
    print(
        f"saved {len(frame)} official KOSPI rows: {args.output} "
        f"({frame['date'].min().date()}..{frame['date'].max().date()})"
    )
    print(f"provenance manifest: {args.manifest}")


if __name__ == "__main__":
    main()
