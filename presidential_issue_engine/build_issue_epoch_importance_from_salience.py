"""Build election-specific issue importance from Assembly salience.

The transform uses already-processed salience rows. It does not rerun Assembly
parsing. The goal is to distinguish era/election issue importance from plain
late-campaign recency by combining three observable signals:

- election-local issue share,
- late-campaign momentum,
- persistence across the observed campaign window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from presidential_issue_engine.point_in_time import cutoff_dates_as_strings, filter_available_by_election
except ModuleNotFoundError:  # supports direct script execution
    from point_in_time import cutoff_dates_as_strings, filter_available_by_election


DEFAULT_SALIENCE = Path("data/issue_salience_assembly.csv")
DEFAULT_MEGA_TERMS = Path("presidential_issue_engine/fixed_dataset/mega_issue_terms.csv")
DEFAULT_OUTPUT = Path("data/raw/issue_epoch_importance.csv")
DEFAULT_DIAGNOSTICS = Path("presidential_issue_engine/report/tables/issue_epoch_importance_diagnostics.csv")

ELECTION_DATES = {
    "pres_2002": "2002-12-19",
    "pres_2007": "2007-12-19",
    "pres_2012": "2012-12-19",
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}


def _election_index(elections: list[str]) -> dict[str, int]:
    return {election_id: index for index, election_id in enumerate(elections)}


def _mega_term_weights(mega_terms_path: Path, elections: list[str]) -> pd.DataFrame:
    """Return max scoped mega-term weight by election and issue."""

    if not mega_terms_path.exists():
        return pd.DataFrame(columns=["election_id", "issue_name", "mega_term_weight"])
    terms = pd.read_csv(mega_terms_path)
    required = {"issue_name", "weight", "start_election", "end_election"}
    if terms.empty or not required.issubset(terms.columns):
        return pd.DataFrame(columns=["election_id", "issue_name", "mega_term_weight"])

    order = _election_index(elections)
    rows: list[dict[str, object]] = []
    frame = terms.copy()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(1.0).clip(lower=1.0)
    for row in frame.itertuples(index=False):
        issue_name = str(getattr(row, "issue_name", "")).strip()
        if not issue_name:
            continue
        start = str(getattr(row, "start_election", "") or "").strip()
        end = str(getattr(row, "end_election", "") or "").strip()
        start_index = order.get(start, 0)
        end_index = order.get(end, len(elections) - 1) if end else len(elections) - 1
        if start and start not in order:
            continue
        if end and end not in order:
            continue
        for election_id in elections:
            index = order[election_id]
            if start_index <= index <= end_index:
                rows.append(
                    {
                        "election_id": election_id,
                        "issue_name": issue_name,
                        "mega_term_weight": float(getattr(row, "weight")),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["election_id", "issue_name", "mega_term_weight"])
    return (
        pd.DataFrame(rows)
        .groupby(["election_id", "issue_name"], as_index=False)["mega_term_weight"]
        .max()
    )


def _robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    median = float(values.median())
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    if iqr <= 1e-9:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return ((values - median) / iqr).clip(-2.0, 2.0)


def build_issue_epoch_importance(
    salience_path: Path = DEFAULT_SALIENCE,
    *,
    mega_terms_path: Path = DEFAULT_MEGA_TERMS,
    strength: float = 0.20,
    mega_term_strength: float = 0.75,
    min_multiplier: float = 0.85,
    max_multiplier: float = 1.35,
    momentum_days: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return issue-epoch multipliers and diagnostics from salience rows."""

    salience = pd.read_csv(salience_path)
    required = {"election_id", "issue_name", "period", "salience_score"}
    if salience.empty or not required.issubset(salience.columns):
        empty = pd.DataFrame(
            columns=[
                "election_id",
                "issue_name",
                "importance_multiplier",
                "available_date",
                "confidence",
                "notes",
            ]
        )
        return empty, empty.copy()

    frame = filter_available_by_election(
        salience,
        ELECTION_DATES,
        source_name="issue_salience_assembly",
    )[["election_id", "issue_name", "period", "salience_score"]].copy()
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
    frame["salience_score"] = pd.to_numeric(frame["salience_score"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["election_id", "issue_name", "period"])
    if frame.empty:
        empty = pd.DataFrame()
        return empty, empty

    frame["election_date"] = frame["election_id"].map(ELECTION_DATES).pipe(pd.to_datetime)
    observed_end = frame.groupby("election_id")["period"].transform("max")
    frame["effective_end"] = observed_end
    valid_date = frame["election_date"].notna()
    frame.loc[valid_date, "effective_end"] = frame.loc[valid_date, ["election_date", "effective_end"]].min(axis=1)
    frame["age_days"] = (frame["effective_end"] - frame["period"]).dt.days.clip(lower=0)
    frame["is_recent"] = frame["age_days"] <= max(momentum_days, 1)
    frame["is_active"] = frame["salience_score"] > 0

    grouped = frame.groupby(["election_id", "issue_name"], as_index=False).agg(
        mean_salience=("salience_score", "mean"),
        max_salience=("salience_score", "max"),
        observations=("salience_score", "size"),
        active_observations=("is_active", "sum"),
    )
    recent = (
        frame.loc[frame["is_recent"]]
        .groupby(["election_id", "issue_name"], as_index=False)["salience_score"]
        .mean()
        .rename(columns={"salience_score": "recent_salience"})
    )
    grouped = grouped.merge(recent, on=["election_id", "issue_name"], how="left")
    grouped["recent_salience"] = grouped["recent_salience"].fillna(grouped["mean_salience"])

    election_totals = (
        grouped.groupby("election_id", as_index=False)["mean_salience"]
        .sum()
        .rename(columns={"mean_salience": "election_salience_total"})
    )
    grouped = grouped.merge(election_totals, on="election_id", how="left")
    grouped["issue_share"] = grouped["mean_salience"] / grouped["election_salience_total"].replace(0, pd.NA)
    grouped["issue_share"] = grouped["issue_share"].fillna(0.0)
    grouped["share_z"] = grouped.groupby("election_id")["issue_share"].transform(_robust_z)

    momentum_denom = grouped["mean_salience"].replace(0, pd.NA)
    grouped["late_momentum"] = ((grouped["recent_salience"] / momentum_denom) - 1.0).fillna(0.0).clip(-1.0, 1.0)
    grouped["persistence"] = grouped["active_observations"] / grouped["observations"].replace(0, pd.NA)
    grouped["persistence"] = grouped["persistence"].fillna(0.0).clip(0.0, 1.0)
    grouped["importance_signal"] = (
        0.70 * grouped["share_z"]
        + 0.20 * grouped["late_momentum"]
        + 0.10 * (grouped["persistence"] - grouped.groupby("election_id")["persistence"].transform("median"))
    )
    grouped["importance_signal"] = grouped["importance_signal"].clip(-2.0, 2.0)
    grouped["importance_multiplier"] = (
        1.0 + strength * grouped["importance_signal"]
    ).clip(min_multiplier, max_multiplier)
    elections = sorted(frame["election_id"].dropna().astype(str).unique())
    mega_terms = _mega_term_weights(mega_terms_path, elections)
    grouped = grouped.merge(mega_terms, on=["election_id", "issue_name"], how="left")
    grouped["mega_term_weight"] = grouped["mega_term_weight"].fillna(1.0)
    grouped["political_term_boost"] = (
        (grouped["mega_term_weight"] - 1.0).clip(lower=0.0) * max(mega_term_strength, 0.0)
    )
    grouped["importance_multiplier"] = (
        grouped["importance_multiplier"] + grouped["political_term_boost"]
    ).clip(min_multiplier, max_multiplier)
    grouped["confidence"] = (
        0.35
        + 0.25 * grouped["persistence"]
        + 0.20 * grouped["issue_share"].rank(pct=True)
        + 0.10 * grouped["late_momentum"].clip(lower=0.0)
        + 0.10 * (grouped["mega_term_weight"] - 1.0).clip(lower=0.0)
    ).clip(0.25, 0.75)
    grouped["available_date"] = grouped["election_id"].map(cutoff_dates_as_strings(ELECTION_DATES)).fillna("")
    grouped["notes"] = (
        "assembly salience derived; combines election-local issue share, late momentum, and persistence"
    )

    output = grouped[
        [
            "election_id",
            "issue_name",
            "importance_multiplier",
            "available_date",
            "confidence",
            "notes",
        ]
    ].sort_values(["election_id", "importance_multiplier", "issue_name"], ascending=[True, False, True])
    diagnostics = grouped[
        [
            "election_id",
            "issue_name",
            "mean_salience",
            "recent_salience",
            "issue_share",
            "share_z",
            "late_momentum",
            "persistence",
            "importance_signal",
            "mega_term_weight",
            "political_term_boost",
            "importance_multiplier",
            "confidence",
        ]
    ].sort_values(["election_id", "importance_multiplier", "issue_name"], ascending=[True, False, True])
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salience", type=Path, default=DEFAULT_SALIENCE)
    parser.add_argument("--mega-terms", type=Path, default=DEFAULT_MEGA_TERMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--strength", type=float, default=0.20)
    parser.add_argument("--mega-term-strength", type=float, default=0.75)
    parser.add_argument("--min-multiplier", type=float, default=0.85)
    parser.add_argument("--max-multiplier", type=float, default=1.35)
    parser.add_argument("--momentum-days", type=int, default=28)
    args = parser.parse_args()

    output, diagnostics = build_issue_epoch_importance(
        args.salience,
        mega_terms_path=args.mega_terms,
        strength=args.strength,
        mega_term_strength=args.mega_term_strength,
        min_multiplier=args.min_multiplier,
        max_multiplier=args.max_multiplier,
        momentum_days=args.momentum_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(args.diagnostics, index=False, encoding="utf-8-sig")
    print(f"wrote {args.output} rows={len(output)}")
    print(f"wrote {args.diagnostics} rows={len(diagnostics)}")


if __name__ == "__main__":
    main()
