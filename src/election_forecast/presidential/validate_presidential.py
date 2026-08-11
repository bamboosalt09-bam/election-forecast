"""Validation routines for standardized presidential outputs."""

from __future__ import annotations

import pandas as pd

from election_forecast.presidential.schemas import SLOTS


def validate_slot_coverage(frame: pd.DataFrame) -> None:
    """Validate every election-region has A/B/C/alpha rows."""

    expected = set(SLOTS)
    for keys, group in frame.groupby(["election_id", "region_id"]):
        actual = set(group["slot"])
        if actual != expected:
            raise ValueError(f"{keys} has slots {sorted(actual)}, expected {sorted(expected)}")


def validate_vote_share_sums(frame: pd.DataFrame, tolerance: float = 1e-9) -> None:
    """Validate standardized vote shares sum to one by election-region."""

    sums = frame.groupby(["election_id", "region_id"])["vote_share"].sum()
    bad = sums.loc[(sums - 1.0).abs() > tolerance]
    if not bad.empty:
        raise ValueError(f"vote_share does not sum to 1.0 for: {bad.to_dict()}")

