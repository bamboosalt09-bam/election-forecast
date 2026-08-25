"""What a prospective target frame owes the model, stated column by column.

The prospective assembly used to close the gap between the historical frame and
the target frame like this::

    for column in historical.columns:
        if column not in out.columns:
            out[column] = 0.0

with a comment asserting that whatever was missing was diagnostic-only and
therefore inert. Two of the things it caught were not.

* the whole ``regional_accent_*`` family - 27 columns - which feeds the accent
  gain, which gates the log shift, which moves the prediction. The published
  2025 forecast ran with that layer contributing exactly nothing.
* ``major_party_core_eligible``, which decides whether a candidate's durable
  core survives at all. Every 2025 candidate was marked ineligible, including
  both major-party nominees, so their durable core was zeroed and folded into
  critical support.

Neither is visible in the output: a zero is a legal value everywhere it landed.

The rule here replaces the assumption with a declaration. Every column the
target lacks falls into exactly one of three kinds, and anything that fits none
of them stops the run:

``OUTCOME_COLUMNS``
    quantities that only exist once the election has happened. They are set to
    ``NaN``, never to zero - a zero here is a fabricated outcome, and code
    downstream cannot tell it from a real one.

``DECLARED_DEFAULTS``
    columns where a default is genuinely safe, each named individually with the
    value and the reason. Membership is a claim someone made deliberately.

*model-active*
    anything that reaches a predictor or a transform. These must be **built**.
    A caller supplies a builder per family; a family with no builder is a hard
    failure, not a zero.

The last line is the point. Before, an unknown missing column became zero and
nothing said so. Now it stops the run and names itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

#: Known only after the count. NaN, never 0.
OUTCOME_COLUMNS = frozenset(
    {
        "actual",
        "official_pred",
        "err_pp",
        "abs_err_pp",
        "reproduced_legacy_pred",
        "baseline_pre_layer_pred",
        "baseline_pre_layer_err_pp",
        "baseline_pre_layer_abs_err_pp",
    }
)

#: Safe defaults, each declared on purpose. The reason is carried so that
#: adding a column here is a statement rather than a shortcut.
DECLARED_DEFAULTS: Mapping[str, tuple[object, str]] = {
    "frozen_reproduction_difference": (
        0.0,
        "there is no frozen artifact to reproduce for a forecast target",
    ),
    "frozen_reproduction_guard_required": (
        False,
        "the reproduction guard applies to the scored panel only",
    ),
}

#: Families that must be constructed. The key is what a caller passes a builder
#: under; the predicate decides which column names belong to it.
ACTIVE_FAMILIES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("regional_accent", lambda column: column.startswith("regional_accent")),
    ("major_party_core_eligible", lambda column: column == "major_party_core_eligible"),
)


class ProspectiveFeatureError(RuntimeError):
    """A target frame is missing something no default can stand in for."""


def classify(column: str) -> str:
    if column in OUTCOME_COLUMNS:
        return "outcome"
    if column in DECLARED_DEFAULTS:
        return "declared_default"
    for family, matches in ACTIVE_FAMILIES:
        if matches(column):
            return f"active:{family}"
    return "unclassified"


def missing_columns(frame: pd.DataFrame, historical_columns: Iterable[str]) -> list[str]:
    return [column for column in historical_columns if column not in frame.columns]


def resolve(
    frame: pd.DataFrame,
    historical_columns: Iterable[str],
    builders: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    *,
    site: str = "prospective target",
) -> pd.DataFrame:
    """Fill what the target lacks, by kind, and refuse what has no kind."""

    builders = dict(builders or {})
    out = frame.copy()
    missing = missing_columns(out, historical_columns)
    if not missing:
        return out

    unclassified = [c for c in missing if classify(c) == "unclassified"]
    if unclassified:
        raise ProspectiveFeatureError(
            f"{site} is missing {len(unclassified)} column(s) with no declared kind: "
            f"{sorted(unclassified)[:10]}. Classify each as an outcome, a declared "
            "default, or a model-active family with a builder - do not let it "
            "become zero by omission."
        )

    needed_families = sorted(
        {
            classify(column).split(":", 1)[1]
            for column in missing
            if classify(column).startswith("active:")
        }
    )
    absent = [family for family in needed_families if family not in builders]
    if absent:
        raise ProspectiveFeatureError(
            f"{site} is missing model-active families {absent} and no builder was "
            "supplied. These reach a predictor or a transform; a zero would run a "
            "different model and say nothing about it."
        )

    for family in needed_families:
        built = builders[family](out)
        if not isinstance(built, pd.DataFrame):
            raise ProspectiveFeatureError(f"the {family} builder did not return a frame")
        wanted = [c for c in missing if classify(c) == f"active:{family}"]
        produced = [c for c in wanted if c in built.columns]
        if sorted(produced) != sorted(wanted):
            raise ProspectiveFeatureError(
                f"the {family} builder did not produce {sorted(set(wanted) - set(produced))}"
            )
        for column in wanted:
            out[column] = built[column].to_numpy()

    for column in missing:
        kind = classify(column)
        if kind == "outcome":
            out[column] = np.nan
        elif kind == "declared_default":
            out[column] = DECLARED_DEFAULTS[column][0]
    return out
