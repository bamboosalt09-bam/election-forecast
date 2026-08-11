"""A/B/C/alpha analytic slots.

Slots are *comparison positions* within a contest, NOT ideology, camp, party, or
candidate-orientation labels:

- ``A``    : main candidate / list 1.
- ``B``    : main candidate / list 2.
- ``C``    : meaningful third slot when active (else explicit inactive row).
- ``alpha``: aggregate of all remaining candidates.

This is the canonical definition; the open-source engine and the statistics
analysis both bind to it. It mirrors the rules already enforced in
``election_forecast.presidential.schemas`` so the engine can adopt ``common``
without behaviour change.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

SLOTS = ("A", "B", "C", "alpha")


class SlotRow(BaseModel):
    election_id: str
    contest_id: str = ""  # presidential: single national contest; may be blank
    slot: str
    candidate_name: str
    party_name: str
    is_active_slot: bool
    notes: str | None = None


def is_valid_slot(slot: str) -> bool:
    return slot in SLOTS


def normalize_slot_frame(slots: pd.DataFrame, key: tuple[str, ...] = ("election_id", "slot")) -> pd.DataFrame:
    """Validate slot membership/uniqueness and coerce the active-slot flag.

    Generalised from ``presidential.schemas.normalize_slots`` with a configurable
    uniqueness key so legislative/local data can key on
    ``(election_id, contest_id, slot)`` without a separate implementation.
    """

    frame = slots.copy()
    frame["slot"] = frame["slot"].astype(str)
    invalid = sorted(set(frame["slot"]) - set(SLOTS))
    if invalid:
        raise ValueError(f"slot frame contains unsupported slots: {invalid}")
    missing_key = [col for col in key if col not in frame.columns]
    if missing_key:
        raise ValueError(f"slot frame is missing key columns: {missing_key}")
    if frame.duplicated(list(key)).any():
        dupes = frame.loc[frame.duplicated(list(key), keep=False), list(key)]
        raise ValueError(f"slot frame has duplicate rows for {key}: {dupes.to_dict('records')}")
    frame["is_active_slot"] = frame["is_active_slot"].map(_to_bool)
    return frame


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
