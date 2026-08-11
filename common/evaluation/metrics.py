"""Shared scoring ruler.

If the two competitions scored their forecasts differently we could never say
whether the open-source project's more complex engine + AI numericalization
actually beats the statistics project's simple rule-based baseline. So the
metrics live here, once, and both sides import them.

This is a faithful generalisation of
``election_forecast.presidential.evaluate``: same metrics, same column
expectations, but keyed on a configurable ``contest`` column so legislative /
local forecasts score on districts instead of regions. For presidential data the
contest column defaults to ``region_id`` and the behaviour is identical.

Headline metric for the statistics poster: ``percentage_point_errors`` — the
signed and absolute %p gap between predicted and actual slot vote share.
"""

from __future__ import annotations

import pandas as pd


def percentage_point_errors(
    predictions: pd.DataFrame,
    actual: pd.DataFrame,
    join_keys: tuple[str, ...] = ("election_id", "region_id", "slot"),
) -> pd.DataFrame:
    """Return per-row predicted vs actual vote share with %p error columns.

    ``vote_share`` values are treated as fractions in ``[0, 1]``; the reported
    ``error_pp`` / ``abs_error_pp`` are in percentage points.
    """

    pred = predictions.copy()
    act = actual[list(join_keys) + ["vote_share"]].rename(columns={"vote_share": "actual_vote_share"})
    merged = pred.merge(act, on=list(join_keys), how="inner")
    merged["error"] = merged["predicted_vote_share"] - merged["actual_vote_share"]
    merged["abs_error"] = merged["error"].abs()
    merged["error_pp"] = merged["error"] * 100.0
    merged["abs_error_pp"] = merged["abs_error"] * 100.0
    return merged


def evaluate_predictions(
    predictions: pd.DataFrame,
    actual: pd.DataFrame,
    target_election: str,
    contest_col: str = "region_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute model metrics and per-contest error rows.

    Returns ``(evaluation, regional_errors)`` exactly like the presidential
    evaluator, so it is a drop-in replacement.
    """

    pred = predictions.loc[predictions["election_id"] == target_election].copy()
    act = actual.loc[actual["election_id"] == target_election].copy()
    if pred.empty:
        raise ValueError(f"No predictions found for target_election={target_election}")
    if act.empty:
        raise ValueError(f"No actual results found for target_election={target_election}")

    merged = pred.merge(
        act[["election_id", contest_col, "slot", "vote_share"]].rename(
            columns={"vote_share": "actual_vote_share"}
        ),
        on=["election_id", contest_col, "slot"],
        how="inner",
    )
    merged["error"] = merged["predicted_vote_share"] - merged["actual_vote_share"]
    merged["abs_error"] = merged["error"].abs()

    carry = [c for c in ("region_name", "province", "is_active_slot", "model_name") if c in merged.columns]
    regional_errors = merged[
        ["election_id", contest_col, "slot", *carry, "predicted_vote_share", "actual_vote_share", "error", "abs_error"]
    ].rename(columns={"election_id": "target_election_id"})

    metric_rows: list[dict[str, object]] = []
    group_col = "model_name" if "model_name" in regional_errors.columns else None
    groups = regional_errors.groupby(group_col) if group_col else [("default", regional_errors)]
    for model_name, model_frame in groups:
        _add_metric(metric_rows, target_election, model_name, "overall_mae", None, model_frame["abs_error"].mean())
        for slot, slot_frame in model_frame.groupby("slot"):
            if slot == "C" and "is_active_slot" in slot_frame and not slot_frame["is_active_slot"].any():
                continue
            _add_metric(metric_rows, target_election, model_name, "slot_mae", slot, slot_frame["abs_error"].mean())
        _add_metric(
            metric_rows, target_election, model_name, "region_mean_mae", None,
            model_frame.groupby(contest_col)["abs_error"].mean().mean(),
        )
        _add_metric(metric_rows, target_election, model_name, "ab_margin_mae", None, _ab_margin_mae(model_frame, contest_col))
        _add_metric(metric_rows, target_election, model_name, "winner_accuracy", None, _winner_accuracy(model_frame, contest_col))
        _add_metric(metric_rows, target_election, model_name, "national_vote_share_mae", None, _national_mae(model_frame))

    evaluation = pd.DataFrame(metric_rows)
    return evaluation, regional_errors.reset_index(drop=True)


def summarize_contributions(contributions: pd.DataFrame, target_election: str) -> pd.DataFrame:
    """Summarize mean absolute variable contributions by model."""

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
    summary["notes"] = "average absolute contribution across contests and slots"
    return summary[["target_election_id", "model_name", "metric", "slot", "value", "notes"]]


def _add_metric(rows, target_election, model_name, metric, slot, value, notes="") -> None:
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


def _ab_margin_mae(frame: pd.DataFrame, contest_col: str) -> float:
    margins = []
    for _, region in frame.groupby(contest_col):
        pred = _slot_value(region, "A", "predicted_vote_share") - _slot_value(region, "B", "predicted_vote_share")
        actual = _slot_value(region, "A", "actual_vote_share") - _slot_value(region, "B", "actual_vote_share")
        margins.append(abs(pred - actual))
    return float(pd.Series(margins).mean()) if margins else 0.0


def _winner_accuracy(frame: pd.DataFrame, contest_col: str) -> float:
    matches = []
    for _, region in frame.groupby(contest_col):
        active = region.loc[region["is_active_slot"].astype(bool)] if "is_active_slot" in region else region
        if active.empty:
            continue
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
