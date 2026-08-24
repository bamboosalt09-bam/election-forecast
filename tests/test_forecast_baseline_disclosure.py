"""The baseline summary must say which model it measured.

`outputs/forecast_baselines/` is not versioned and `baseline_summary.json` has
a generic name, so a reader has no way to tell that its `model` figure is V23's
and not the active version's. The disclosure was added by hand once and the
generating script, which rebuilds the file from scratch, dropped it on the next
re-run. It is derived from the input path now; this pins that it survives.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "forecast_baselines" / "baseline_summary.json"
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"


def test_the_summary_names_the_model_it_measured() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    measured = str(summary["model_version"])
    assert measured.startswith("v")
    assert measured in str(summary["model_version_note"]).lower()

    predictions = ROOT / f"outputs/active_presidential_nested_{measured}/nested_predictions.csv"
    assert predictions.is_file(), "the disclosed version has no frozen artifact"


def test_the_disclosure_is_load_bearing_only_when_it_differs() -> None:
    """If the baselines are ever recomputed against the active model, the note
    stops being a caveat - but it must still be true either way."""

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    active = str(json.loads(POINTER.read_text(encoding="utf-8"))["active_version"])
    note = str(summary["model_version_note"])
    if summary["model_version"] != active:
        assert "does not imply the current active version" in note
