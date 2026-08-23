"""Expand regional deviations where a third candidate compresses them.

Predicted regional spread matches the realised spread in 2002, 2012 and 2022
and falls short of it in 2007 and 2017 - the two scored elections with a
substantial third candidate. The shortfall is present at the earliest modelled
stage, so it belongs to the fitted base rather than to any postprocess layer.

Shrinkage is not by itself a defect; a regularised predictor should have less
variance than the outcome. What marks 2007 and 2017 is that the slope of
realised on predicted exceeds 1 there - near 2 for 이회창 2007 - while it sits
at 1 elsewhere. That is a calibration gap, not optimal shrinkage.

Each candidate's regional deviations are expanded around that candidate's own
weighted national level::

    scaled = level + (1 + gain * predicted_third_share) * (pred - level)

and each region is then renormalised. Two properties follow from the form
rather than from tuning:

* the index is the model's **predicted** third share, so no outcome is read;
* expanding around each candidate's own level is compositional, so the
  candidate levels - and therefore the national level - are conserved;
* at the default gain of 1 the factor is simply ``1 + predicted_third_share``,
  so there is no constant for the scored panel to select.

``predicted_third_share`` is near zero for the three elections that were not
compressed, so those elections are left almost exactly where they were. The
transform scopes itself by the same quantity that diagnoses the gap.

This complements :mod:`party_regionalism_dispersion`, which restores dispersion
a bloc's own prior evidences and is inert wherever that evidence is absent.
This module addresses the compression that a third candidate induces, which
that prior does not describe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FLOOR = 1e-6
# A gain of exactly 1 leaves no constant to choose: the expansion factor is
# then the predicted third share itself, and the transform has no free
# parameter for the five scored outcomes to select. Smaller gains score better
# on that panel - 0.50 improves both headline metrics where 1.00 costs a little
# national accuracy - but only because 0.50 was swept against the same five
# outcomes it is then measured on. The parameter-free value is preferred over
# the better-scoring one for that reason; the sweep is recorded in
# docs/EXPERIMENT_V29_THIRD_SHARE_DISPERSION_20260823.md so the cost of the
# choice stays visible.
DEFAULT_GAIN = 1.0


def _name_column(frame: pd.DataFrame) -> str:
    return "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"


def _levels(group: pd.DataFrame, name: str, prediction: str, weight: str) -> pd.Series:
    """Each candidate's weighted national level within one election."""

    return group.groupby(name, sort=False).apply(
        lambda part: float(np.average(part[prediction], weights=part[weight]))
    )


def _feasible_factor(prediction: pd.Series, level: pd.Series, factor: float) -> float:
    """The largest expansion this election admits without a negative share.

    A factor that drives a regional share below zero does not describe a
    dispersion. Clipping such a row and renormalising would inject vote mass and
    break the level conservation the transform is promoted on, so the expansion
    stops at the boundary instead - the transform's own definition running out,
    not a constant being chosen.

    The cap is per election rather than per candidate for a structural reason:
    candidate levels sum to one in every region, so a factor applied uniformly
    leaves each region summing to one exactly and renormalisation is a no-op.
    Give two candidates different factors and the regional sums drift, the
    renormalisation stops being neutral, and the levels move - which is the very
    failure the cap exists to prevent. On the scored panel this binds once, in
    2017, where 홍준표's 광주 and 전남 predictions would otherwise go negative.
    """

    below = prediction < level
    if not bool(below.any()):
        return factor
    room = ((level - FLOOR) / (level - prediction))[below]
    return float(min(factor, room.min()))


def predicted_third_share(
    group: pd.DataFrame,
    name: str,
    prediction: str = "layer_pred",
    weight: str = "contest_votes",
) -> float:
    """The model's own third-placed national level - no outcome is read."""

    ordered = _levels(group, name, prediction, weight).sort_values(ascending=False)
    return float(ordered.iloc[2]) if len(ordered) > 2 else 0.0


def apply_third_share_dispersion_expansion(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "layer_pred",
    weight_column: str = "contest_votes",
    gain: float = DEFAULT_GAIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand regional deviations in proportion to the predicted third share.

    Candidate national levels and regional compositions are conserved.
    """

    required = {"election_id", "region_id", prediction_column, weight_column}
    name = _name_column(frame)
    required.add(name)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"third-share dispersion expansion needs {missing}")
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError(f"gain must be finite and non-negative, got {gain!r}")

    parts: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for election, group in frame.groupby("election_id", sort=False):
        out = group.copy()
        third = predicted_third_share(out, name, prediction_column, weight_column)
        factor = 1.0 + gain * third
        levels = _levels(out, name, prediction_column, weight_column)
        level = out[name].map(levels)
        feasible = _feasible_factor(out[prediction_column], level, factor)
        expanded = level + feasible * (out[prediction_column] - level)
        total = expanded.groupby(out["region_id"]).transform("sum")
        out[prediction_column] = expanded / total
        parts.append(out)

        after = _levels(out, name, prediction_column, weight_column)
        rows.append(
            {
                "election_id": str(election),
                "predicted_third_share": third,
                "gain": gain,
                "expansion_factor": factor,
                "applied_factor": feasible,
                "feasibility_capped": bool(feasible < factor - 1e-12),
                "max_candidate_level_shift_pp": float(
                    (after - levels).abs().max() * 100
                ),
                "outcome_fields_used": "none",
            }
        )
    adjusted = pd.concat(parts, ignore_index=False).loc[frame.index]
    return adjusted, pd.DataFrame(rows)
