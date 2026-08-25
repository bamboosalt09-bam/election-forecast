"""What the run actually opened, recorded where nothing rewrites it.

V28 establishes its external-model-free claim like this::

    with external_model_free_runtime():
        v27.run(...)
    strip_external_model_inputs(manifest_path)      # delete the rows
    assert_external_model_free_manifest(manifest_path)   # check what is left

The assertion inspects a file the line above it just edited. A manifest that
passes therefore says nothing about what the process opened - only that the
overlay row was removed before anyone looked.

It is not hypothetical. ``issue_vote_engine`` reads the overlay through a bare
module constant::

    _read_csv_if_exists(ASSEMBLY_ISSUE_CHARACTER_OVERLAY)

which the guard does not touch: the guard sets an environment variable and two
config flags, and the other overlay consumer honours those, but this call does
not consult them. Measured, the file is opened on every run.

That read turns out to be harmless - removing the file entirely reproduces
V31's predictions byte for byte, and every ``issue_pref_*`` column has been
zero since V28 - but "harmless" was established by experiment, not by the
manifest, and the manifest was the thing being offered as evidence.

This module records reads at the point they happen and never edits the record.
Selecting what may be redistributed is a separate question answered from the
trace, not by deleting from it.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

#: Paths whose appearance in a trace is a policy failure for a version that
#: claims not to read them. Kept as fragments so a relative or absolute path
#: matches the same way.
EXTERNAL_MODEL_DERIVED_FRAGMENTS: tuple[str, ...] = (
    "assembly_issue_character_overlay.csv",
    "data/raw/auto_issue_seed/mega_issue_axis.csv",
    "data/raw/auto_issue_seed/mega_issue_attribution.csv",
)

_lock = threading.Lock()
_trace: list[dict[str, object]] = []
_active = False


def _normalise(path: object) -> str:
    return str(path).replace("\\", "/")


def record(path: object, *, reader: str) -> None:
    """Note one file read. Never removes, never rewrites."""

    if not _active:
        return
    with _lock:
        _trace.append({"path": _normalise(path), "reader": reader})


@contextmanager
def tracing() -> Iterator[list[dict[str, object]]]:
    """Collect reads for the duration of a run."""

    global _active
    with _lock:
        _trace.clear()
        _active = True
    try:
        yield _trace
    finally:
        with _lock:
            _active = False


def to_frame(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    source = _trace if rows is None else rows
    frame = pd.DataFrame(source, columns=["path", "reader"])
    if frame.empty:
        return frame
    counted = (
        frame.groupby(["path", "reader"], as_index=False)
        .size()
        .rename(columns={"size": "reads"})
        .sort_values(["path", "reader"], ignore_index=True)
    )
    return counted


def external_model_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of a trace that name an external-model-derived input."""

    if frame.empty:
        return frame
    paths = frame["path"].astype(str)
    hit = pd.Series(False, index=frame.index)
    for fragment in EXTERNAL_MODEL_DERIVED_FRAGMENTS:
        hit |= paths.str.endswith(fragment)
    return frame.loc[hit]


#: A reader label meaning the path was requested and denied. The request still
#: belongs in the trace - it records that the caller wanted the file - but a
#: denied request is the policy working, not breaking it.
REFUSED_READER = "refused_by_v32"


def assert_no_external_model_derived_reads(frame: pd.DataFrame, *, site: str) -> None:
    """Fail on the trace itself, not on a filtered copy of it.

    A refusal is not a read. The trace keeps both so the record shows that the
    engine still asks for the overlay and is denied, which is more informative
    than a trace where the request never appears.
    """

    offending = external_model_derived(frame)
    if not offending.empty and "reader" in offending.columns:
        offending = offending.loc[offending["reader"].astype(str) != REFUSED_READER]
    if not offending.empty:
        listed = sorted(set(offending["path"].astype(str)))
        raise RuntimeError(
            f"{site} opened external-model-derived input(s) {listed}. This is "
            "checked against the raw read trace, which is never edited - a "
            "version that claims not to read these must not open them, and "
            "removing the row afterwards would prove nothing."
        )


def write(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination
