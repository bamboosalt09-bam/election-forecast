"""Evaluate regional bloc-prior forecasts against held-out election results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from election_forecast.config import DEFAULT_CONFIG
from election_forecast.features.region_bloc_prior import (
    CONSERVATIVE_BLOC,
    INDEPENDENT_BLOC,
    LIBERAL_BLOC,
    PROGRESSIVE_BLOC,
    THIRD_BLOC,
    compute_bloc_base,
    election_date,
    load_bloc_history,
    normalize_bloc,
)


EVALUATED_BLOCS = [
    CONSERVATIVE_BLOC,
    LIBERAL_BLOC,
    PROGRESSIVE_BLOC,
    THIRD_BLOC,
    INDEPENDENT_BLOC,
]


def build_actual_frame(history: pd.DataFrame, target_election_id: str) -> pd.DataFrame:
    """Return target-election actual shares normalized to evaluated blocs."""

    actual = history.loc[history["election_id"].astype(str) == str(target_election_id)].copy()
    if actual.empty:
        return pd.DataFrame(columns=["region_id", "bloc", "actual_share"])
    actual["bloc"] = actual["bloc"].map(normalize_bloc)
    actual = actual.loc[actual["bloc"].isin(EVALUATED_BLOCS)].copy()
    actual = (
        actual.groupby(["region_id", "bloc"], as_index=False)["vote_share"]
        .sum()
        .rename(columns={"vote_share": "actual_share"})
    )
    region_sum = actual.groupby("region_id")["actual_share"].transform("sum")
    actual = actual.loc[region_sum.gt(0)].copy()
    actual["actual_share"] = actual["actual_share"] / region_sum.loc[actual.index]
    return actual


def evaluate_target(history: pd.DataFrame, target_election_id: str) -> pd.DataFrame:
    """Evaluate one target election using only elections before its date."""

    predictions = compute_bloc_base(
        history,
        target_election_id,
        election_type_weights=DEFAULT_CONFIG.election_type_weights,
    ).rename(columns={"bloc_base": "pred_share"})
    predictions = predictions.loc[predictions["bloc"].isin(EVALUATED_BLOCS)].copy()
    actual = build_actual_frame(history, target_election_id)
    if predictions.empty or actual.empty:
        return pd.DataFrame()

    grid = actual[["region_id"]].drop_duplicates().merge(
        pd.DataFrame({"bloc": EVALUATED_BLOCS}),
        how="cross",
    )
    out = (
        grid.merge(predictions[["region_id", "bloc", "pred_share"]], on=["region_id", "bloc"], how="left")
        .merge(actual, on=["region_id", "bloc"], how="left")
        .fillna({"pred_share": 0.0, "actual_share": 0.0})
    )
    for col in ["pred_share", "actual_share"]:
        totals = out.groupby("region_id")[col].transform("sum")
        mask = totals.gt(0)
        out.loc[mask, col] = out.loc[mask, col] / totals.loc[mask]

    pred_winner = (
        out.sort_values(["region_id", "pred_share"], ascending=[True, False])
        .drop_duplicates("region_id")
        [["region_id", "bloc"]]
        .rename(columns={"bloc": "pred_top_bloc"})
    )
    actual_winner = (
        out.sort_values(["region_id", "actual_share"], ascending=[True, False])
        .drop_duplicates("region_id")
        [["region_id", "bloc"]]
        .rename(columns={"bloc": "actual_top_bloc"})
    )
    out = out.merge(pred_winner, on="region_id", how="left").merge(actual_winner, on="region_id", how="left")
    out["top_bloc_correct"] = out["pred_top_bloc"].eq(out["actual_top_bloc"])

    target_type = history.loc[
        history["election_id"].astype(str) == str(target_election_id), "election_type"
    ].iloc[0]
    out["election_id"] = target_election_id
    out["election_type"] = target_type
    out["target_date"] = election_date(target_election_id)
    out["error"] = out["pred_share"] - out["actual_share"]
    out["abs_error"] = out["error"].abs()
    return out[
        [
            "election_id",
            "election_type",
            "target_date",
            "region_id",
            "bloc",
            "pred_share",
            "actual_share",
            "error",
            "abs_error",
            "pred_top_bloc",
            "actual_top_bloc",
            "top_bloc_correct",
        ]
    ]


def summarize(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build election-level and type-level summaries."""

    by_election = (
        details.groupby(["election_id", "election_type", "target_date"], as_index=False)
        .agg(
            rows=("abs_error", "size"),
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            p90_ae=("abs_error", lambda s: s.quantile(0.90)),
            max_ae=("abs_error", "max"),
            top_bloc_accuracy=("top_bloc_correct", "mean"),
        )
        .sort_values(["target_date", "election_id"])
    )
    by_type = (
        details.groupby("election_type", as_index=False)
        .agg(
            elections=("election_id", "nunique"),
            rows=("abs_error", "size"),
            mae=("abs_error", "mean"),
            median_ae=("abs_error", "median"),
            p90_ae=("abs_error", lambda s: s.quantile(0.90)),
            max_ae=("abs_error", "max"),
            top_bloc_accuracy=("top_bloc_correct", "mean"),
        )
        .sort_values("mae")
    )
    return by_election, by_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="data/raw/bloc_history_results.csv")
    parser.add_argument("--output-dir", default="outputs/bloc_forecast_eval")
    parser.add_argument(
        "--election-types",
        nargs="*",
        default=[
            "assembly_pr",
            "metro_council_pr",
            "local_council_pr",
        ],
    )
    parser.add_argument("--min-year", type=int, default=2006)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_bloc_history(args.history)
    target_ids = []
    for election_id, group in history.groupby("election_id"):
        election_type = str(group["election_type"].iloc[0])
        date = election_date(election_id)
        if election_type not in set(args.election_types) or date is None or date.year < args.min_year:
            continue
        target_ids.append(str(election_id))
    target_ids = sorted(target_ids, key=lambda value: election_date(value) or pd.Timestamp.max)

    frames = []
    for target_id in target_ids:
        frame = evaluate_target(history, target_id)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise SystemExit("No evaluable target elections found.")
    details = pd.concat(frames, ignore_index=True)
    by_election, by_type = summarize(details)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "bloc_forecast_details.csv", index=False)
    by_election.to_csv(output_dir / "bloc_forecast_by_election.csv", index=False)
    by_type.to_csv(output_dir / "bloc_forecast_by_type.csv", index=False)

    print(f"Wrote {output_dir / 'bloc_forecast_details.csv'}")
    print(f"Wrote {output_dir / 'bloc_forecast_by_election.csv'}")
    print(f"Wrote {output_dir / 'bloc_forecast_by_type.csv'}")
    print("\nBy election type, MAE in percentage points:")
    display = by_type.copy()
    for col in ["mae", "median_ae", "p90_ae", "max_ae"]:
        display[col] = (display[col] * 100).round(2)
    display["top_bloc_accuracy"] = (display["top_bloc_accuracy"] * 100).round(1)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
