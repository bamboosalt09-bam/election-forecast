"""Date filters that prevent future information leakage."""

from __future__ import annotations

import pandas as pd


def filter_available_date(frame: pd.DataFrame, forecast_date: str | pd.Timestamp) -> pd.DataFrame:
    """Keep only rows with ``available_date <= forecast_date``.

    This is the core leakage-control filter. Event dates and survey dates may
    differ from availability; model inputs must use availability.
    """

    if "available_date" not in frame.columns:
        return frame.copy()
    cutoff = pd.Timestamp(forecast_date)
    available = pd.to_datetime(frame["available_date"], errors="coerce")
    if available.isna().any():
        raise ValueError(
            f"available_date contains {int(available.isna().sum())} missing or invalid values"
        )
    return frame.loc[available.le(cutoff)].copy()


def latest_by_available_date(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Return the latest available row for each key combination."""

    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(keys + ["available_date"])
    return ordered.groupby(keys, as_index=False, sort=False).tail(1).reset_index(drop=True)
