"""Strict point-in-time helpers for presidential forecast inputs.

Every election-specific feature must have been available by the end of the
day before the election. Missing dates and unknown election identifiers are
errors instead of neutral fallbacks so a backtest cannot silently admit
post-election information.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def forecast_cutoff(
    election_id: object,
    election_dates: Mapping[str, str],
) -> pd.Timestamp | None:
    """Return the D-1 forecast cutoff for a known election."""

    election_date = pd.to_datetime(election_dates.get(str(election_id)), errors="coerce")
    if pd.isna(election_date):
        return None
    return pd.Timestamp(election_date) - pd.Timedelta(days=1)


def forecast_cutoff_map(election_dates: Mapping[str, str]) -> dict[str, pd.Timestamp]:
    """Return D-1 cutoffs for all parseable election dates."""

    out: dict[str, pd.Timestamp] = {}
    for election_id in election_dates:
        cutoff = forecast_cutoff(election_id, election_dates)
        if cutoff is not None:
            out[str(election_id)] = cutoff
    return out


def filter_available_by_election(
    frame: pd.DataFrame,
    election_dates: Mapping[str, str],
    *,
    source_name: str,
    election_column: str = "election_id",
    available_column: str = "available_date",
) -> pd.DataFrame:
    """Keep rows available by D-1 and reject incomplete time metadata."""

    if frame.empty:
        return frame.copy()
    required = {election_column, available_column}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{source_name} is missing point-in-time columns: {missing_columns}")

    out = frame.copy()
    out[available_column] = pd.to_datetime(out[available_column], errors="coerce")
    cutoffs = forecast_cutoff_map(election_dates)
    out["_forecast_cutoff"] = out[election_column].astype(str).map(cutoffs)

    invalid = out[available_column].isna() | out["_forecast_cutoff"].isna()
    if invalid.any():
        examples = (
            out.loc[invalid, [election_column, available_column]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"{source_name} has {int(invalid.sum())} rows without auditable cutoff metadata: {examples}"
        )

    eligible = out[available_column] <= out["_forecast_cutoff"]
    return out.loc[eligible].drop(columns="_forecast_cutoff").copy()


def cutoff_dates_as_strings(election_dates: Mapping[str, str]) -> dict[str, str]:
    """Return ISO D-1 cutoff strings for CSV builders."""

    return {
        election_id: cutoff.date().isoformat()
        for election_id, cutoff in forecast_cutoff_map(election_dates).items()
    }


def parse_observed_dates(values: pd.Series) -> pd.Series:
    """Parse ISO and Korean Assembly date labels into timestamps."""

    text_values = values.astype("string").str.strip()
    parsed = pd.to_datetime(text_values, format="%Y-%m-%d", errors="coerce")
    unresolved = parsed.isna()
    if unresolved.any():
        parts = text_values.loc[unresolved].str.extract(
            r"(?P<year>\d{4})\s*년\s*(?P<month>\d{1,2})\s*월\s*(?P<day>\d{1,2})\s*일"
        )
        parsed.loc[unresolved] = pd.to_datetime(parts, errors="coerce")
    return parsed


def filter_observed_by_election(
    frame: pd.DataFrame,
    election_dates: Mapping[str, str],
    *,
    source_name: str,
    date_column: str,
    election_column: str = "election_id",
) -> pd.DataFrame:
    """Keep raw observations dated by D-1 and reject unauditable rows."""

    if frame.empty:
        return frame.copy()
    required = {election_column, date_column}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{source_name} is missing observation-date columns: {missing_columns}")

    out = frame.copy()
    out[date_column] = parse_observed_dates(out[date_column])
    out["_forecast_cutoff"] = out[election_column].astype(str).map(
        forecast_cutoff_map(election_dates)
    )
    invalid = out[date_column].isna() | out["_forecast_cutoff"].isna()
    if invalid.any():
        examples = (
            out.loc[invalid, [election_column, date_column]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"{source_name} has {int(invalid.sum())} rows without auditable observation dates: {examples}"
        )
    eligible = out[date_column] <= out["_forecast_cutoff"]
    return out.loc[eligible].drop(columns="_forecast_cutoff").copy()
