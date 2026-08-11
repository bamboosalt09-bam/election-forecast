"""Evaluate held-out PR elections at party-label granularity."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

from election_forecast.config import DEFAULT_CONFIG
from election_forecast.features.region_bloc_prior import election_date, election_year


def normalize_party_label(value: object) -> str:
    """Keep party labels distinct while removing purely typographic variation."""

    text = "" if pd.isna(value) else str(value).strip()
    return re.sub(r"\s+", "", text)


def load_history(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"election_id", "election_type", "region_id", "bloc", "vote_share"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"history is missing required columns: {missing}")
    out = frame.copy()
    out["party_label"] = out["bloc"].map(normalize_party_label)
    out["vote_share"] = pd.to_numeric(out["vote_share"], errors="coerce").fillna(0.0)
    if "data_quality_weight" not in out.columns:
        out["data_quality_weight"] = 1.0
    out["data_quality_weight"] = pd.to_numeric(out["data_quality_weight"], errors="coerce").fillna(1.0)
    return out


def compute_party_base(history: pd.DataFrame, target_election_id: str) -> pd.DataFrame:
    """Predict regional party-label shares from elections before the target."""

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
        return pd.DataFrame(columns=["region_id", "party_label", "pred_share"])
    if frame.empty:
        return pd.DataFrame(columns=["region_id", "party_label", "pred_share"])

    if target_date is not None:
        age = (target_date - frame["source_date"]).dt.days / (365.25 * 5.0)
    else:
        age = (target_year - frame["source_year"].astype(float)) / 5.0
    frame["time_weight"] = age.map(lambda value: math.exp(-float(value) / 2.0))
    frame["type_weight"] = frame["election_type"].map(DEFAULT_CONFIG.election_type_weights).fillna(0.35)
    frame["weight"] = frame["time_weight"] * frame["type_weight"] * frame["data_quality_weight"]
    frame["weighted_share"] = frame["vote_share"] * frame["weight"]

    grouped = frame.groupby(["region_id", "party_label"], as_index=False).agg(
        weighted_share=("weighted_share", "sum"),
        weight=("weight", "sum"),
    )
    grouped["pred_share"] = grouped["weighted_share"] / grouped["weight"].replace(0, pd.NA)
    grouped["pred_share"] = grouped["pred_share"].fillna(0.0).clip(lower=0.0)
    totals = grouped.groupby("region_id")["pred_share"].transform("sum")
    mask = totals.gt(0)
    grouped.loc[mask, "pred_share"] = grouped.loc[mask, "pred_share"] / totals.loc[mask]
    return grouped[["region_id", "party_label", "pred_share"]]


def build_actual(history: pd.DataFrame, target_election_id: str) -> pd.DataFrame:
    actual = history.loc[history["election_id"].astype(str) == str(target_election_id)].copy()
    actual = (
        actual.groupby(["region_id", "party_label"], as_index=False)["vote_share"]
        .sum()
        .rename(columns={"vote_share": "actual_share"})
    )
    totals = actual.groupby("region_id")["actual_share"].transform("sum")
    actual = actual.loc[totals.gt(0)].copy()
    actual["actual_share"] = actual["actual_share"] / totals.loc[actual.index]
    return actual


def evaluate_target(history: pd.DataFrame, target_election_id: str, party_universe: str) -> pd.DataFrame:
    pred = compute_party_base(history, target_election_id)
    actual = build_actual(history, target_election_id)
    if pred.empty or actual.empty:
        return pd.DataFrame()
    regions = actual[["region_id"]].drop_duplicates()
    if party_universe == "union":
        parties = pd.concat([pred[["party_label"]], actual[["party_label"]]], ignore_index=True).drop_duplicates()
    elif party_universe == "target_ballot":
        parties = actual[["party_label"]].drop_duplicates()
    else:
        raise ValueError(f"Unsupported party universe: {party_universe}")
    grid = regions.merge(parties, how="cross")
    out = (
        grid.merge(pred, on=["region_id", "party_label"], how="left")
        .merge(actual, on=["region_id", "party_label"], how="left")
        .fillna({"pred_share": 0.0, "actual_share": 0.0})
    )
    for col in ["pred_share", "actual_share"]:
        totals = out.groupby("region_id")[col].transform("sum")
        mask = totals.gt(0)
        out.loc[mask, col] = out.loc[mask, col] / totals.loc[mask]

    pred_top = (
        out.sort_values(["region_id", "pred_share"], ascending=[True, False])
        .drop_duplicates("region_id")[["region_id", "party_label"]]
        .rename(columns={"party_label": "pred_top_party"})
    )
    actual_top = (
        out.sort_values(["region_id", "actual_share"], ascending=[True, False])
        .drop_duplicates("region_id")[["region_id", "party_label"]]
        .rename(columns={"party_label": "actual_top_party"})
    )
    out = out.merge(pred_top, on="region_id", how="left").merge(actual_top, on="region_id", how="left")
    out["top_party_correct"] = out["pred_top_party"].eq(out["actual_top_party"])
    out["election_id"] = target_election_id
    out["election_type"] = history.loc[
        history["election_id"].astype(str) == str(target_election_id), "election_type"
    ].iloc[0]
    out["target_date"] = election_date(target_election_id)
    out["error"] = out["pred_share"] - out["actual_share"]
    out["abs_error"] = out["error"].abs()
    return out[
        [
            "election_id",
            "election_type",
            "target_date",
            "region_id",
            "party_label",
            "pred_share",
            "actual_share",
            "error",
            "abs_error",
            "pred_top_party",
            "actual_top_party",
            "top_party_correct",
        ]
    ]


def summarize(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_election = (
        details.groupby(["election_id", "election_type", "target_date"], as_index=False)
        .agg(
            rows=("abs_error", "size"),
            parties=("party_label", "nunique"),
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            p90_ae=("abs_error", lambda s: s.quantile(0.90)),
            max_ae=("abs_error", "max"),
            top_party_accuracy=("top_party_correct", "mean"),
        )
        .sort_values(["target_date", "election_id"])
    )
    by_type = (
        details.groupby("election_type", as_index=False)
        .agg(
            elections=("election_id", "nunique"),
            rows=("abs_error", "size"),
            parties=("party_label", "nunique"),
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            p90_ae=("abs_error", lambda s: s.quantile(0.90)),
            max_ae=("abs_error", "max"),
            top_party_accuracy=("top_party_correct", "mean"),
        )
        .sort_values("mae")
    )
    national = (
        details.groupby(["election_id", "election_type", "party_label"], as_index=False)
        .agg(pred_share=("pred_share", "mean"), actual_share=("actual_share", "mean"))
    )
    national["error"] = national["pred_share"] - national["actual_share"]
    national["abs_error"] = national["error"].abs()
    return by_election, by_type, national


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="data/raw/bloc_history_results.csv")
    parser.add_argument("--output-dir", default="outputs/party_forecast_eval_pr_only")
    parser.add_argument("--min-year", type=int, default=2006)
    parser.add_argument(
        "--party-universe",
        choices=["target_ballot", "union"],
        default="target_ballot",
        help="target_ballot treats the registered party list as known before election day.",
    )
    parser.add_argument(
        "--election-types",
        nargs="*",
        default=["assembly_pr", "metro_council_pr", "local_council_pr"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_history(args.history)
    target_ids = []
    election_types = set(args.election_types)
    for election_id, group in history.groupby("election_id"):
        election_type = str(group["election_type"].iloc[0])
        date = election_date(election_id)
        if election_type not in election_types or date is None or date.year < args.min_year:
            continue
        target_ids.append(str(election_id))
    target_ids = sorted(target_ids, key=lambda value: election_date(value) or pd.Timestamp.max)
    frames = []
    for target_id in target_ids:
        frame = evaluate_target(history, target_id, args.party_universe)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise SystemExit("No evaluable party-level target elections found.")
    details = pd.concat(frames, ignore_index=True)
    by_election, by_type, national = summarize(details)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "party_forecast_details.csv", index=False)
    by_election.to_csv(output_dir / "party_forecast_by_election.csv", index=False)
    by_type.to_csv(output_dir / "party_forecast_by_type.csv", index=False)
    national.to_csv(output_dir / "party_forecast_national_points.csv", index=False)

    display = by_type.copy()
    for col in ["mae", "median_ae", "p90_ae", "max_ae"]:
        display[col] = (display[col] * 100).round(2)
    display["top_party_accuracy"] = (display["top_party_accuracy"] * 100).round(1)
    print(display.to_string(index=False))
    print(f"overall_national_point_mae_pp={(national['abs_error'].mean() * 100):.3f}")


if __name__ == "__main__":
    main()
