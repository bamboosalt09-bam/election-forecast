"""Evaluate presidential variable-model predictions against standardized results."""

from __future__ import annotations

import pandas as pd


def evaluate_predictions(
    predictions: pd.DataFrame,
    actual: pd.DataFrame,
    target_election: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute model metrics and regional error rows."""

    pred = predictions.loc[predictions["election_id"] == target_election].copy()
    act = actual.loc[actual["election_id"] == target_election].copy()
    if pred.empty:
        raise ValueError(f"No predictions found for target_election={target_election}")
    if act.empty:
        raise ValueError(f"No actual results found for target_election={target_election}")

    merged = pred.merge(
        act[
            [
                "election_id",
                "region_id",
                "slot",
                "vote_share",
            ]
        ].rename(columns={"vote_share": "actual_vote_share"}),
        on=["election_id", "region_id", "slot"],
        how="inner",
    )
    merged["error"] = merged["predicted_vote_share"] - merged["actual_vote_share"]
    merged["abs_error"] = merged["error"].abs()
    regional_errors = merged[
        [
            "election_id",
            "region_id",
            "region_name",
            "province",
            "slot",
            "is_active_slot",
            "model_name",
            "predicted_vote_share",
            "actual_vote_share",
            "error",
            "abs_error",
        ]
    ].rename(columns={"election_id": "target_election_id"})

    metric_rows: list[dict[str, object]] = []
    for model_name, model_frame in regional_errors.groupby("model_name"):
        _add_metric(metric_rows, target_election, model_name, "overall_mae", None, model_frame["abs_error"].mean())
        for slot, slot_frame in model_frame.groupby("slot"):
            if slot == "C" and not slot_frame["is_active_slot"].any():
                continue
            _add_metric(metric_rows, target_election, model_name, "slot_mae", slot, slot_frame["abs_error"].mean())
        _add_metric(
            metric_rows,
            target_election,
            model_name,
            "region_mean_mae",
            None,
            model_frame.groupby("region_id")["abs_error"].mean().mean(),
        )
        _add_metric(
            metric_rows,
            target_election,
            model_name,
            "ab_margin_mae",
            None,
            _ab_margin_mae(model_frame),
        )
        _add_metric(
            metric_rows,
            target_election,
            model_name,
            "winner_accuracy",
            None,
            _winner_accuracy(model_frame),
        )
        _add_metric(
            metric_rows,
            target_election,
            model_name,
            "national_vote_share_mae",
            None,
            _national_mae(model_frame),
        )

    evaluation = pd.DataFrame(metric_rows)
    return evaluation, regional_errors.reset_index(drop=True)


def summarize_contributions(contributions: pd.DataFrame, target_election: str) -> pd.DataFrame:
    """Summarize absolute variable contributions by model."""

    frame = contributions.loc[contributions["election_id"] == target_election].copy()
    if frame.empty:
        return pd.DataFrame(columns=["target_election_id", "model_name", "metric", "slot", "value", "notes"])
    summary = (
        frame.groupby(["model_name", "variable_name"], as_index=False)["contribution"]
        .apply(lambda series: series.abs().mean())
        .rename(columns={"contribution": "value"})
    )
    summary["target_election_id"] = target_election
    summary["metric"] = "mean_abs_variable_contribution"
    summary["slot"] = summary["variable_name"]
    summary["notes"] = "average absolute contribution across regions and slots"
    return summary[["target_election_id", "model_name", "metric", "slot", "value", "notes"]]


def _add_metric(
    rows: list[dict[str, object]],
    target_election: str,
    model_name: str,
    metric: str,
    slot: str | None,
    value: float,
    notes: str = "",
) -> None:
    rows.append(
        {
            "target_election_id": target_election,
            "model_name": model_name,
            "metric": metric,
            "slot": slot or "",
            "value": float(value),
            "notes": notes,
        }
    )


def _ab_margin_mae(frame: pd.DataFrame) -> float:
    margins = []
    for _, region in frame.groupby("region_id"):
        pred = _slot_value(region, "A", "predicted_vote_share") - _slot_value(
            region, "B", "predicted_vote_share"
        )
        actual = _slot_value(region, "A", "actual_vote_share") - _slot_value(region, "B", "actual_vote_share")
        margins.append(abs(pred - actual))
    return float(pd.Series(margins).mean()) if margins else 0.0


def _winner_accuracy(frame: pd.DataFrame) -> float:
    matches = []
    for _, region in frame.groupby("region_id"):
        active = region.loc[region["is_active_slot"].astype(bool)]
        pred_winner = active.sort_values("predicted_vote_share", ascending=False).iloc[0]["slot"]
        actual_winner = active.sort_values("actual_vote_share", ascending=False).iloc[0]["slot"]
        matches.append(pred_winner == actual_winner)
    return float(pd.Series(matches, dtype=float).mean()) if matches else 0.0


def _national_mae(frame: pd.DataFrame) -> float:
    national = frame.groupby("slot", as_index=False)[["predicted_vote_share", "actual_vote_share"]].mean()
    return float((national["predicted_vote_share"] - national["actual_vote_share"]).abs().mean())


def _slot_value(frame: pd.DataFrame, slot: str, column: str) -> float:
    row = frame.loc[frame["slot"] == slot]
    return float(row.iloc[0][column]) if not row.empty else 0.0

