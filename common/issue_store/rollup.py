"""Roll the issue/event store up into feature_schema variables.

This is the single shared bridge from "issue memory" to the forecast engine. It
follows the same math the engine already uses in
``election_forecast.issue_score`` — time-decayed aggregation then a region
sensitivity join — but emits rows in ``common.feature_schema`` shape so any
populator (curated / aggregate / corpus) feeds the engine identically.

Routing:
- non-risk issues (policy, endorsement, achievement, ...) -> ``local_issue_fit``
  (favorable direction raises it, unfavorable lowers it).
- risk issues (scandal, gaffe) -> ``risk_or_negative`` as a negative burden.

Values are squashed with ``tanh`` so the accumulated sum always lands in the
``[-1, 1]`` range the feature contract requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.issue_store.schema import RISK_ISSUE_TYPES


def rollup_issue_features(
    issues: pd.DataFrame,
    region_sensitivity: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    half_life_days: float = 30.0,
) -> pd.DataFrame:
    """Aggregate the issue store into ``local_issue_fit`` / ``risk_or_negative``.

    Parameters
    ----------
    issues:
        Issue-store rows (see :class:`IssueEventRow`).
    region_sensitivity:
        ``region_id, issue_name, sensitivity_score`` — how much each region cares
        about each issue.
    forecast_date:
        Leakage cutoff. Rows with ``available_date`` after this are dropped.
    half_life_days:
        Time-decay half-life; older issues fade.

    Returns
    -------
    A ``common.feature_schema``-shaped frame:
    ``election_id, election_type, region_id, slot, variable_name,
    variable_value, available_date, aggregation_rule, scorer``.
    """

    cols = [
        "election_id", "election_type", "region_id", "slot",
        "variable_name", "variable_value", "available_date", "aggregation_rule", "scorer",
    ]
    if issues.empty or region_sensitivity.empty:
        return pd.DataFrame(columns=cols)

    cutoff = pd.Timestamp(forecast_date)
    frame = issues.copy()
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame = frame.loc[frame["available_date"].le(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame(columns=cols)

    for col in ("salience_score", "direction_score", "candidate_link_score", "media_reliability_score"):
        frame[col] = pd.to_numeric(frame.get(col, 0.0), errors="coerce").fillna(0.0)
    if "final_issue_score" not in frame.columns:
        frame["final_issue_score"] = np.nan
    frame["final_issue_score"] = pd.to_numeric(frame["final_issue_score"], errors="coerce")
    derived = frame["salience_score"] * frame["direction_score"] * frame["media_reliability_score"]
    frame["final_issue_score"] = frame["final_issue_score"].fillna(derived)

    age_days = (cutoff - frame["event_date"]).dt.days.clip(lower=0).fillna(0)
    frame["weight"] = np.exp(-age_days / float(half_life_days))
    frame["w_issue"] = frame["final_issue_score"] * frame["weight"]
    frame["w_link"] = frame["candidate_link_score"] * frame["weight"]

    grouped = frame.groupby(
        ["election_id", "slot", "issue_name", "issue_type", "region_scope"], as_index=False
    ).agg(w_issue=("w_issue", "sum"), w_link=("w_link", "sum"), weight=("weight", "sum"))
    grouped["issue_score"] = grouped["w_issue"] / grouped["weight"].replace(0, np.nan)
    grouped["link"] = grouped["w_link"] / grouped["weight"].replace(0, np.nan)
    grouped = grouped.fillna({"issue_score": 0.0, "link": 0.0})

    sens = region_sensitivity.copy()
    sens["sensitivity_score"] = pd.to_numeric(sens["sensitivity_score"], errors="coerce").fillna(0.0)

    # Apply each issue to its region(s): region_scope=="ALL" -> every region that
    # has a sensitivity row for the issue; otherwise only the named region.
    joined = grouped.merge(sens[["region_id", "issue_name", "sensitivity_score"]], on="issue_name", how="inner")
    specific = joined["region_scope"].ne("ALL")
    joined = joined.loc[~specific | (joined["region_scope"] == joined["region_id"])].copy()
    if joined.empty:
        return pd.DataFrame(columns=cols)

    joined["component"] = joined["issue_score"] * joined["sensitivity_score"] * joined["link"]
    joined["variable_name"] = np.where(
        joined["issue_type"].isin(RISK_ISSUE_TYPES), "risk_or_negative", "local_issue_fit"
    )
    # Risk issues are a negative burden regardless of stored direction sign.
    joined.loc[joined["variable_name"] == "risk_or_negative", "component"] = (
        -joined.loc[joined["variable_name"] == "risk_or_negative", "component"].abs()
    )

    out = joined.groupby(
        ["election_id", "region_id", "slot", "variable_name"], as_index=False
    )["component"].sum()
    out["variable_value"] = np.tanh(out["component"])
    out["election_type"] = "presidential"
    out["available_date"] = str(pd.Timestamp(forecast_date).date())
    out["aggregation_rule"] = "national_vote_share"
    out["scorer"] = "issue_rollup"
    return out[cols]
