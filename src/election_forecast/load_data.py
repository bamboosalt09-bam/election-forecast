"""CSV loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from election_forecast.schemas import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    validate_optional_columns,
    validate_required_columns,
)

DATE_COLUMNS = {
    "election_results": ["election_date", "available_date"],
    "candidates": ["available_date"],
    "candidate_party_vectors": ["available_date"],
    "polls": ["published_date", "start_date", "end_date", "available_date"],
    "region_issue_sensitivity": ["available_date"],
    "candidate_policy_positions": ["available_date"],
    "issue_scores": ["date", "available_date"],
    "event_effects": ["event_date", "available_date"],
    "candidate_issue_profile": ["available_date"],
    "mega_issue_axis": ["available_date"],
    "mega_issue_attribution": ["available_date"],
}


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    """Read a required CSV and parse known date columns."""

    frame = pd.read_csv(path)
    validate_required_columns(name, frame)
    for col in DATE_COLUMNS.get(name, []):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
        if col == "available_date" and frame[col].isna().any():
            raise ValueError(f"{name}.csv contains missing or invalid available_date")
    return frame


def _read_optional_csv(path: Path, name: str) -> pd.DataFrame:
    """Read an optional CSV, returning an empty typed frame when absent."""

    if not path.exists():
        return pd.DataFrame(columns=OPTIONAL_COLUMNS[name])
    frame = pd.read_csv(path)
    validate_optional_columns(name, frame)
    for col in DATE_COLUMNS.get(name, []):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
        if col == "available_date" and frame[col].isna().any():
            raise ValueError(f"{name}.csv contains missing or invalid available_date")
    return frame


def _optional_input_path(data_dir: Path, raw_dir: Path, name: str) -> Path:
    automatic_names = {
        "candidate_issue_profile",
        "mega_issue_axis",
        "mega_issue_attribution",
    }
    if name not in automatic_names:
        return raw_dir / f"{name}.csv"
    config_path = data_dir / "config" / "through2022_rederived_layers.json"
    if not config_path.exists():
        return raw_dir / f"{name}.csv"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    if config.get("automatic_issue_seed_enabled") is True:
        automatic = raw_dir / "auto_issue_seed" / f"{name}.csv"
        if automatic.exists():
            return automatic
    return raw_dir / f"{name}.csv"


def load_raw_data(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load all raw CSV inputs from ``data_dir/raw``."""

    data_path = Path(data_dir)
    raw_dir = data_path / "raw"
    data = {
        name: _read_csv(raw_dir / f"{name}.csv", name)
        for name in REQUIRED_COLUMNS
    }
    data.update(
        {
            name: _read_optional_csv(_optional_input_path(data_path, raw_dir, name), name)
            for name in OPTIONAL_COLUMNS
        }
    )
    return data


def write_processed_csv(frame: pd.DataFrame, output_dir: str | Path, file_name: str) -> Path:
    """Write a processed CSV and return the output path."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / file_name
    frame.to_csv(path, index=False)
    return path
