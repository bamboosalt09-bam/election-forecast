"""What a prospective target frame owes the model, column by column.

The prospective assembly used to close the gap between the historical frame and
the target frame like this::

    for column in historical.columns:
        if column not in out.columns:
            out[column] = 0.0

with a comment asserting that whatever was missing was diagnostic-only and
therefore inert. A sweep of the shipped 2025 artifact found **40 columns**
identically zero across all 51 rows while populated for the scored elections,
and five separate families among them were model-active:

============================== ==== ===============================================
family                         cols why the zero was wrong
============================== ==== ===============================================
``regional_accent_*``            28 computable from pre-cutoff history, and the
                                    estimator does compute it; the accent gain,
                                    log shift and prediction all followed to zero
``major_party_core_eligible``     1 decidable from the ballot bloc; every 2025
                                    candidate was marked ineligible, including
                                    both major-party nominees
``lineage_identity_*``            5 the profile needs only strictly-prior events;
                                    the builder simply never generated pres_2025
``wasted_vote_resistance``        1 present 16/16 in the forecast context table;
                                    the consumer reads a history-only path
``strategic_transfer_confidence`` 1 same table, same path, same cause
============================== ==== ===============================================

Zero is a legal value everywhere they landed, so none of it showed in the
output.

The rule here replaces the assumption with a declaration. Every column the
target lacks belongs to exactly one class, and anything that belongs to none
stops the run:

``REQUIRED_DERIVED``
    must be produced from point-in-time information. A caller supplies a
    builder per family; a family with no builder is a hard failure, never a
    zero.

``EXPLICIT_ZERO``
    zero by design, named individually with the reason. Membership is a claim
    someone made deliberately, not a default that happened.

``OUTCOME_ONLY``
    exists only after the election. Set to ``NaN``, never zero - a zero here is
    a fabricated result that downstream code cannot distinguish from a real one.

``DIAGNOSTIC_ONLY``
    not used in the prediction, which is proven by a test rather than asserted
    here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

REQUIRED_DERIVED = "REQUIRED_DERIVED"
EXPLICIT_ZERO = "EXPLICIT_ZERO"
OUTCOME_ONLY = "OUTCOME_ONLY"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"

#: Known only after the count. NaN, never 0.
OUTCOME_COLUMNS: frozenset[str] = frozenset(
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

#: Zero by design. Each entry states why, so adding one is a decision.
EXPLICIT_ZERO_COLUMNS: Mapping[str, str] = {
    "frozen_reproduction_difference": (
        "there is no frozen artifact for a forecast target to reproduce against"
    ),
}

#: Not read by any prediction stage. The claim is enforced by
#: tests/test_prospective_feature_contract.py, which fails if one of these is
#: found feeding a prediction column.
DIAGNOSTIC_ONLY_COLUMNS: Mapping[str, str] = {
    "frozen_reproduction_guard_required": (
        "a bookkeeping flag recording whether the reproduction guard was demanded"
    ),
}

#: Families that must be built from point-in-time evidence. The key is what a
#: caller registers a builder under; the predicate decides membership.
REQUIRED_DERIVED_FAMILIES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("regional_accent", lambda column: column.startswith("regional_accent")),
    ("major_party_core_eligible", lambda column: column == "major_party_core_eligible"),
    ("lineage_identity", lambda column: column.startswith("lineage_identity")),
    (
        "strategic_lane_context",
        lambda column: column
        in {"wasted_vote_resistance", "strategic_transfer_confidence"},
    ),
)


class ProspectiveFeatureError(RuntimeError):
    """A target frame is missing something no default can stand in for."""


def classify(column: str) -> tuple[str, str]:
    """Return ``(class, detail)`` for one column name."""

    if column in OUTCOME_COLUMNS:
        return OUTCOME_ONLY, "known only after the count"
    if column in EXPLICIT_ZERO_COLUMNS:
        return EXPLICIT_ZERO, EXPLICIT_ZERO_COLUMNS[column]
    if column in DIAGNOSTIC_ONLY_COLUMNS:
        return DIAGNOSTIC_ONLY, DIAGNOSTIC_ONLY_COLUMNS[column]
    for family, matches in REQUIRED_DERIVED_FAMILIES:
        if matches(column):
            return REQUIRED_DERIVED, family
    return "UNCLASSIFIED", "no class declared"


def missing_columns(frame: pd.DataFrame, historical_columns: Iterable[str]) -> list[str]:
    return [column for column in historical_columns if column not in frame.columns]


def resolve(
    frame: pd.DataFrame,
    historical_columns: Iterable[str],
    builders: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    *,
    site: str = "prospective target",
) -> pd.DataFrame:
    """Fill what the target lacks, by class, and refuse what has no class."""

    builders = dict(builders or {})
    out = frame.copy()
    missing = missing_columns(out, historical_columns)
    if not missing:
        return out

    unclassified = [c for c in missing if classify(c)[0] == "UNCLASSIFIED"]
    if unclassified:
        raise ProspectiveFeatureError(
            f"{site} is missing {len(unclassified)} column(s) with no declared class: "
            f"{sorted(unclassified)[:10]}. Each must be REQUIRED_DERIVED, "
            "EXPLICIT_ZERO, OUTCOME_ONLY or DIAGNOSTIC_ONLY - none of them may "
            "become zero by omission."
        )

    families = sorted(
        {
            classify(column)[1]
            for column in missing
            if classify(column)[0] == REQUIRED_DERIVED
        }
    )
    absent = [family for family in families if family not in builders]
    if absent:
        raise ProspectiveFeatureError(
            f"{site} is missing required-derived families {absent} and no builder "
            "was supplied. These are computable from point-in-time evidence; a "
            "zero would run a different model and say nothing about it."
        )

    for family in families:
        built = builders[family](out)
        if not isinstance(built, pd.DataFrame):
            raise ProspectiveFeatureError(f"the {family} builder did not return a frame")
        wanted = [
            c for c in missing if classify(c) == (REQUIRED_DERIVED, family)
        ]
        absent_columns = sorted(set(wanted) - set(built.columns))
        if absent_columns:
            raise ProspectiveFeatureError(
                f"the {family} builder did not produce {absent_columns}"
            )
        for column in wanted:
            out[column] = built[column].to_numpy()

    for column in missing:
        kind, _ = classify(column)
        if kind == OUTCOME_ONLY:
            out[column] = np.nan
        elif kind == EXPLICIT_ZERO:
            out[column] = 0.0
        elif kind == DIAGNOSTIC_ONLY:
            out[column] = False if column.endswith("_required") else 0.0
    return out
