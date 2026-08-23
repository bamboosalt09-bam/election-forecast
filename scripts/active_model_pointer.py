"""Resolve the directory the active-model pointer names.

The diagnostics were written against V26 and kept reporting it after V27 and
V28 were promoted, which is how V27 figures ended up under a README heading
that said V28. Following the pointer is what prevents that recurring, so the
resolution lives in one place rather than being copied into each report - and
the guards import the same function, so a guard cannot keep validating a
retired version after its report has moved on.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
# Used only when the pointer is unreadable, so a diagnostic still runs rather
# than failing on a concern that is not the one it exists to measure.
FALLBACK_OUTPUT = "outputs/active_presidential_nested_v28"


def active_version() -> str:
    """The pointer's declared version, for skip messages and report headers."""

    try:
        return str(json.loads(POINTER.read_text(encoding="utf-8"))["active_version"])
    except Exception:  # noqa: BLE001 - a missing pointer must not break a report
        return Path(FALLBACK_OUTPUT).name.rsplit("_", 1)[-1]


def active_output_dir() -> Path:
    try:
        return ROOT / json.loads(POINTER.read_text(encoding="utf-8"))["output"]
    except Exception:  # noqa: BLE001 - as above
        return ROOT / FALLBACK_OUTPUT
