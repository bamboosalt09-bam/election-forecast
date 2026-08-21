"""Graded mega-issue intensity between an inert and a saturated shock.

``SHOCK_CLASS_INTENSITY`` maps a shock class to one of four values and
``compile_direct_mega_scores`` ramps attribution with
``(intensity - 1).clip(0, 1)``. The ramp is continuous but its reachable inputs
are not, so activation lands on ``{0, 0, 0, 1}`` and a direct shock is either
inert or fully saturated. This module fills the gap from below.

    proximity = clip(min_regime_evidence / CRISIS_MIN_REGIME_EVIDENCE, 0, 1)
              * clip(accountability_component / CRISIS_ACCOUNTABILITY, 0, 1)
    intensity = floor + (crisis_ceiling - floor) * proximity

The ceiling is the existing institutional-crisis level, each floor is the
election's existing intensity, and both gates are the classifier's own
thresholds, so no constant is introduced here. ``proximity`` is exactly 1.0 for
any election at or above both gates, which makes the rule one-sided: an
election the classifier already calls a crisis cannot move, and an election
with no crisis evidence keeps its current value.

Filling from above - scaling intensity by the margin above the gate - is not
available. 2017 clears the regime gate by 0.027 of its 0.35 range because a
snap election leaves the Assembly record a short window, so a margin-scaled
intensity collapses the one scored calibration point this path has.

This layer must be paired with ``align_profile_to_event_class``. Raising the
floors exposes the winner-take-all issue race on elections whose leading
political-shock issue is off-class, which the intensity gate at 1.00 had been
suppressing. See ``docs/EXPERIMENT_V25_INTENSITY_LADDER_20260822.md``.
"""

from __future__ import annotations

import pandas as pd

from presidential_issue_engine.automatic_controls_v22 import (
    CRISIS_ACCOUNTABILITY,
    CRISIS_MIN_REGIME_EVIDENCE,
    SHOCK_CLASS_INTENSITY,
)

CRISIS_INTENSITY = float(SHOCK_CLASS_INTENSITY["institutional_crisis"])
REGIME_COMPONENTS = ("salience_component", "severity_component", "breadth_component")
REQUIRED_COMPONENTS = REGIME_COMPONENTS + ("accountability_component",)


def crisis_proximity(diagnostics: pd.DataFrame) -> pd.Series:
    """How close each election's evidence sits to the institutional-crisis gate."""

    missing = sorted(set(REQUIRED_COMPONENTS) - set(diagnostics.columns))
    if missing:
        raise ValueError(f"classifier diagnostics are missing columns: {missing}")
    frame = diagnostics.copy()
    for column in REQUIRED_COMPONENTS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    regime = frame[list(REGIME_COMPONENTS)].min(axis=1) / CRISIS_MIN_REGIME_EVIDENCE
    accountability = frame["accountability_component"] / CRISIS_ACCOUNTABILITY
    return (regime.clip(0.0, 1.0) * accountability.clip(0.0, 1.0)).rename("proximity")


def ladder_intensity(
    intensity: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    intensity_column: str = "mega_issue_intensity",
) -> pd.DataFrame:
    """Raise each floor toward the crisis ceiling in proportion to proximity.

    An election absent from ``diagnostics`` keeps its floor: an unmeasured
    election must never be raised toward the ceiling by default.
    """

    proximity = dict(
        zip(
            diagnostics["election_id"].astype(str),
            crisis_proximity(diagnostics).astype(float),
        )
    )
    out = intensity.copy()
    floor = pd.to_numeric(out[intensity_column], errors="coerce")
    weight = out["election_id"].astype(str).map(proximity).fillna(0.0)
    out[intensity_column] = (floor + (CRISIS_INTENSITY - floor) * weight).round(6)
    return out
