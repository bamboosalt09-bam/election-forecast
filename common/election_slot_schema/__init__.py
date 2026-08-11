"""A/B/C/alpha analytic slot definition (shared by both competitions)."""

from common.election_slot_schema.slots import (
    SLOTS,
    SlotRow,
    is_valid_slot,
    normalize_slot_frame,
)

__all__ = ["SLOTS", "SlotRow", "is_valid_slot", "normalize_slot_frame"]
