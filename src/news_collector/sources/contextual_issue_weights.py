"""Context-sensitive issue weighting helpers.

These helpers do not add direct vote-share predictors. They only alter issue
mention weights during text reprocessing, so macro context changes issue
salience and candidate-issue links rather than entering the election model as a
separate fitted variable.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd

from election_forecast.features.issue_matcher import IssueContextRule
from presidential_issue_engine.point_in_time import forecast_cutoff


def load_issue_context_rules(
    path: str | Path,
    election_ids: Iterable[str],
) -> dict[str, list[IssueContextRule]]:
    """Load election-window context rules from CSV."""

    elections = list(election_ids)
    rules_by_election: dict[str, list[IssueContextRule]] = {election_id: [] for election_id in elections}
    csv_path = Path(path)
    if not csv_path.exists():
        return rules_by_election

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_issue = (row.get("source_issue") or "").strip()
            terms = tuple(term.strip() for term in (row.get("context_terms") or "").split("|") if term.strip())
            if not source_issue or not terms:
                continue
            rule = IssueContextRule(
                source_issue=source_issue,
                context_terms=terms,
                source_multiplier=float(row.get("source_multiplier") or 1.0),
                target_issue=(row.get("target_issue") or "").strip() or None,
                target_weight=float(row.get("target_weight") or 0.0),
            )
            start = (row.get("start_election") or "").strip()
            end = (row.get("end_election") or "").strip()
            active = _active_elections(elections, start, end)
            for election_id in active:
                rules_by_election[election_id].append(rule)
    return rules_by_election


def housing_issue_boosts_from_index(
    housing_index: str | Path,
    election_dates: dict[str, str],
    election_order: Iterable[str],
    max_boost: float = 1.25,
    scale_pct: float = 100.0,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Build election-level housing issue boosts from prior-cycle price growth.

    The returned boost is applied to ``housing`` text matches only. It uses the
    latest housing index available by each election date and compares it to the
    latest index available by the previous presidential election date.
    """

    elections = list(election_order)
    boosts: dict[str, dict[str, float]] = {election_id: {} for election_id in elections}
    path = Path(housing_index)
    if not path.exists():
        return boosts, _empty_diagnostics()

    frame = pd.read_csv(path)
    required = {"region_id", "period", "value", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return boosts, _empty_diagnostics()
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["region_id", "period", "available_date", "value"])

    snapshots: dict[str, pd.DataFrame] = {}
    for election_id in elections:
        election_date_text = election_dates.get(election_id)
        if election_date_text is None:
            continue
        cutoff = forecast_cutoff(election_id, election_dates)
        if cutoff is None:
            continue
        eligible = frame.loc[frame["available_date"] <= cutoff].copy()
        if eligible.empty:
            continue
        snapshots[election_id] = (
            eligible.sort_values(["region_id", "period"])
            .groupby("region_id", as_index=False)
            .tail(1)[["region_id", "period", "value"]]
        )

    diagnostics: list[dict[str, object]] = []
    for index, election_id in enumerate(elections[1:], start=1):
        previous_id = elections[index - 1]
        if election_id not in snapshots or previous_id not in snapshots:
            continue
        current = snapshots[election_id].rename(columns={"period": "current_period", "value": "current_value"})
        baseline = snapshots[previous_id].rename(columns={"period": "baseline_period", "value": "baseline_value"})
        joined = current.merge(baseline, on="region_id", how="inner")
        if joined.empty:
            continue
        joined["change_pct"] = (joined["current_value"] / joined["baseline_value"] - 1.0) * 100.0
        mean_positive = max(float(joined["change_pct"].mean()), 0.0)
        multiplier = min(1.0 + mean_positive / scale_pct * (max_boost - 1.0), max_boost)
        boosts[election_id]["housing"] = multiplier
        diagnostics.append(
            {
                "election_id": election_id,
                "previous_election_id": previous_id,
                "mean_housing_change_pct": float(joined["change_pct"].mean()),
                "housing_issue_multiplier": multiplier,
                "n_regions": len(joined),
                "baseline_period_min": joined["baseline_period"].min().date().isoformat(),
                "current_period_max": joined["current_period"].max().date().isoformat(),
            }
        )
    return boosts, pd.DataFrame(diagnostics)


def _active_elections(elections: list[str], start: str, end: str) -> list[str]:
    start_index = elections.index(start) if start in elections else 0
    end_index = elections.index(end) if end in elections else len(elections) - 1
    return elections[start_index : end_index + 1]


def _empty_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "election_id",
            "previous_election_id",
            "mean_housing_change_pct",
            "housing_issue_multiplier",
            "n_regions",
            "baseline_period_min",
            "current_period_max",
        ]
    )
