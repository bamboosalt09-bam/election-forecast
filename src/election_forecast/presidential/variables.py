"""Political variable preparation for presidential utility models."""

from __future__ import annotations

import pandas as pd

from election_forecast.presidential.schemas import SLOTS


def prepare_variables(
    variables: pd.DataFrame,
    election_id: str,
    available_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter variables to one election and optional information cutoff."""

    frame = variables.loc[variables["election_id"] == election_id].copy()
    if frame.empty:
        return frame
    if "available_date" not in frame.columns:
        raise ValueError("political_variables.csv is missing available_date")
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    if frame["available_date"].isna().any():
        raise ValueError("political_variables.csv contains missing or invalid available_date")
    if available_date is not None:
        cutoff = pd.to_datetime(available_date)
        frame = frame.loc[frame["available_date"] <= cutoff].copy()
    frame["slot"] = frame["slot"].astype(str)
    invalid_slots = sorted(set(frame["slot"]) - set(SLOTS))
    if invalid_slots:
        raise ValueError(f"political_variables.csv contains unsupported slots: {invalid_slots}")
    frame["variable_value"] = pd.to_numeric(frame["variable_value"], errors="coerce").fillna(0.0)
    frame["variable_value"] = frame["variable_value"].clip(-1.0, 1.0)
    return frame
