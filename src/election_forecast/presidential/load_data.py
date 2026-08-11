"""CSV IO helpers for the presidential variable model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from election_forecast.presidential.schemas import validate_required_columns


DATE_COLUMNS = {
    "political_variables": ["available_date"],
    "manual_political_variables": ["available_date"],
    "party_controversy_scores": ["period_start", "period_end", "available_date"],
    "candidate_tone_scores": ["period_start", "period_end", "available_date"],
    "transfer_events": ["event_date", "available_date"],
}


def read_presidential_csv(path: str | Path, name: str) -> pd.DataFrame:
    """Read and validate a presidential CSV."""

    frame = pd.read_csv(path)
    validate_required_columns(name, frame)
    for column in DATE_COLUMNS.get(name, []):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if column == "available_date" and frame[column].isna().any():
            raise ValueError(f"{name}.csv contains missing or invalid available_date")
    return frame


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV to an exact path and return it."""

    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
