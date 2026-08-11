"""Analyze housing-price index changes by presidential administration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REGIMES = [
    {
        "regime_code": "roh_partial",
        "regime_name": "노무현 정부(자료구간 한정)",
        "start_date": "2003-02-25",
        "end_date": "2008-02-24",
    },
    {
        "regime_code": "lee_mb",
        "regime_name": "이명박 정부",
        "start_date": "2008-02-25",
        "end_date": "2013-02-24",
    },
    {
        "regime_code": "park_gh",
        "regime_name": "박근혜 정부",
        "start_date": "2013-02-25",
        "end_date": "2017-03-10",
    },
    {
        "regime_code": "moon_ji",
        "regime_name": "문재인 정부",
        "start_date": "2017-05-10",
        "end_date": "2022-05-09",
    },
    {
        "regime_code": "yoon_sy",
        "regime_name": "윤석열 정부",
        "start_date": "2022-05-10",
        "end_date": "2025-04-04",
    },
    {
        "regime_code": "lee_jm_initial",
        "regime_name": "이재명 정부 초기",
        "start_date": "2025-06-04",
        "end_date": "2026-06-29",
    },
]


def analyze(input_path: Path, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary and detail tables for the configured administrations."""

    frame = pd.read_csv(input_path)
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["period", "value"])

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for regime in REGIMES:
        start = pd.Timestamp(regime["start_date"])
        end = pd.Timestamp(regime["end_date"])
        eligible = frame.loc[(frame["period"] >= start) & (frame["period"] <= end)].copy()
        if eligible.empty:
            summary_rows.append(_empty_summary(regime, "no quarter-end observation"))
            continue

        eligible = eligible.sort_values("period")
        first = eligible.groupby(keys, as_index=False).first()
        last = eligible.groupby(keys, as_index=False).last()
        detail = first[keys + ["period", "value"]].merge(
            last[keys + ["period", "value"]],
            on=keys,
            suffixes=("_start", "_end"),
        )
        detail["change_pct"] = (detail["value_end"] / detail["value_start"] - 1.0) * 100.0
        detail["regime_code"] = regime["regime_code"]
        detail["regime_name"] = regime["regime_name"]
        detail["start_date"] = regime["start_date"]
        detail["end_date"] = regime["end_date"]
        detail_frames.append(detail)

        periods = sorted(eligible["period"].dt.date.unique())
        enough_periods = len(periods) >= 2
        summary_rows.append(
            {
                "regime_code": regime["regime_code"],
                "regime_name": regime["regime_name"],
                "start_date": regime["start_date"],
                "end_date": regime["end_date"],
                "start_period": str(periods[0]),
                "end_period": str(periods[-1]),
                "n_regions": len(detail),
                "mean_change_pct": detail["change_pct"].mean() if enough_periods else pd.NA,
                "median_change_pct": detail["change_pct"].median() if enough_periods else pd.NA,
                "min_change_pct": detail["change_pct"].min() if enough_periods else pd.NA,
                "max_change_pct": detail["change_pct"].max() if enough_periods else pd.NA,
                "note": "" if enough_periods else "only one quarter",
            }
        )

    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), details


def _empty_summary(regime: dict[str, str], note: str) -> dict[str, object]:
    return {
        "regime_code": regime["regime_code"],
        "regime_name": regime["regime_name"],
        "start_date": regime["start_date"],
        "end_date": regime["end_date"],
        "start_period": "",
        "end_period": "",
        "n_regions": 0,
        "mean_change_pct": pd.NA,
        "median_change_pct": pd.NA,
        "min_change_pct": pd.NA,
        "max_change_pct": pd.NA,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sido", type=Path, default=Path("presidential_issue_engine/fixed_dataset/housing_price_index_sido.csv"))
    parser.add_argument("--sgg", type=Path, default=Path("presidential_issue_engine/fixed_dataset/housing_price_index_sgg.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("presidential_issue_engine/report/tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sido_summary, sido_detail = analyze(args.sido, ["region_id", "province"])
    sgg_summary, sgg_detail = analyze(args.sgg, ["region_id", "province", "sgg_name"])

    outputs = {
        "housing_regime_sido_summary.csv": sido_summary,
        "housing_regime_sido_detail.csv": sido_detail,
        "housing_regime_sgg_summary.csv": sgg_summary,
        "housing_regime_sgg_detail.csv": sgg_detail,
    }
    for filename, frame in outputs.items():
        path = args.output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"wrote {path}")

    print(sido_summary[["regime_code", "start_period", "end_period", "mean_change_pct", "median_change_pct", "note"]])


if __name__ == "__main__":
    main()
