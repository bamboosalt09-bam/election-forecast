"""Expand regional dispersion multiplicatively, so no share can reach zero.

V29 and V30 expand each candidate's regional deviations around that candidate's
own weighted national level by an additive rule::

    scaled = level + factor * (pred - level)

which is linear in the deviation and therefore has no lower bound. A candidate
far below their own national level is pushed toward zero and, for a large
enough factor, through it. V29 handled that with a per-election feasibility cap
- stop at the largest factor the election admits without a negative share.

The cap has a consequence that was not noticed when it was adopted. It is
defined as the factor at which *some* region reaches zero, so the region that
sets the cap lands on exactly zero, by construction, every time the cap binds.
On the scored panel that is 홍준표's 광주 in 2017: the stage feeding the
transform says 3.55%, the realised share is 1.68%, and the transform published
0.00%. In the 2025 demonstration it is 김문수's 광주: 2.67% in, 0.00% out.

Zero is not a prediction. It is where the arithmetic stopped.

This module expands in log space instead::

    scaled = level * (pred / level) ** factor

The deviation being scaled is now a ratio rather than a difference, so the
result is positive whenever the input is, and the constraint the cap existed to
enforce is satisfied by the form. There is no cap and no floor constant.

What that costs, and how it is paid back
----------------------------------------

The additive form conserves each candidate's weighted national level exactly,
which is the property V29 was promoted on. The multiplicative form does not:
scaling ratios preserves a geometric mean, not the arithmetic one the level is.
Measured on the scored panel, levels move by up to 0.465 percentage points -
small, but it would make the transform change the national forecast, which it
has no business doing.

Restoring the level with one rescale per candidate would break the regional
sums, and renormalising the regions would move the levels again. The two
constraints are satisfied together by alternating them to convergence, which
is iterative proportional fitting. It introduces no constant: the targets are
the input levels and one.

Convergence is not assumed. If the alternation does not reach tolerance the
transform raises, because a nearly-normalised artifact that looks finished is
worse than one that refuses to be produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The expansion index and its gain are V29's, unchanged. At a gain of 1 the
#: factor is the predicted third share itself, so there is no constant for the
#: scored panel to select.
DEFAULT_GAIN = 1.0
#: Alternation limit and tolerance. Both are numerical, not fitted: the panel
#: converges in 1-17 rounds, and the limit exists to fail rather than to tune.
MAX_ROUNDS = 500
LEVEL_TOLERANCE = 1e-14


def _name_column(frame: pd.DataFrame) -> str:
    return "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"


def _levels(
    group: pd.DataFrame, name: str, values: pd.Series, weight: str
) -> pd.Series:
    """Each candidate's weighted national level within one election."""

    return (
        group.assign(_value=values)
        .groupby(name, sort=False)
        .apply(
            lambda part: float(np.average(part["_value"], weights=part[weight])),
            include_groups=False,
        )
    )


def predicted_third_share(
    group: pd.DataFrame, name: str, prediction: str, weight: str
) -> float:
    """The third-placed candidate's own predicted national level.

    Predicted, not realised: no outcome of the target election is read.
    """

    levels = _levels(group, name, group[prediction], weight).sort_values(ascending=False)
    return float(levels.iloc[2]) if len(levels) >= 3 else 0.0


def _normalise_regions(values: pd.Series, regions: pd.Series) -> pd.Series:
    return values / values.groupby(regions).transform("sum")


def apply_multiplicative_dispersion_expansion(
    frame: pd.DataFrame,
    prediction_column: str = "layer_pred",
    weight_column: str = "forecast_time_region_weight",
    gain: float = DEFAULT_GAIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the expanded frame and one audit row per election."""

    name = _name_column(frame)
    missing = sorted(
        {"election_id", "region_id", prediction_column, weight_column} - set(frame.columns)
    )
    if missing:
        raise KeyError(f"multiplicative expansion needs {missing}")

    out = frame.copy()
    records: list[dict[str, object]] = []
    expanded = pd.Series(np.nan, index=out.index, dtype=float)

    for election, group in out.groupby("election_id", sort=False):
        values = group[prediction_column].astype(float)
        if not bool((values > 0.0).all()):
            # The log form is undefined at zero. Nothing in the panel comes
            # near it - the lowest stage value is above 2% - so this is a
            # guard, and it refuses rather than nudging the input.
            raise ValueError(
                f"{election} carries a non-positive {prediction_column}; the "
                "multiplicative expansion is undefined there"
            )

        targets = _levels(group, name, values, weight_column)
        level = group[name].map(targets)
        share = predicted_third_share(group, name, prediction_column, weight_column)
        factor = 1.0 + gain * share

        scaled = level * np.power(values / level, factor)
        scaled = _normalise_regions(scaled, group["region_id"])

        rounds, drift = 0, float("inf")
        for rounds in range(1, MAX_ROUNDS + 1):
            reached = _levels(group, name, scaled, weight_column)
            ratio = group[name].map(targets / reached)
            drift = float((ratio - 1.0).abs().max())
            if drift < LEVEL_TOLERANCE:
                break
            scaled = _normalise_regions(scaled * ratio, group["region_id"])
        else:
            raise RuntimeError(
                f"{election} did not reconcile levels and regional sums in "
                f"{MAX_ROUNDS} rounds (worst level ratio drift {drift:.3e})"
            )

        expanded.loc[group.index] = scaled.astype(float)
        reached = _levels(group, name, scaled, weight_column)
        records.append(
            {
                "election_id": str(election),
                "predicted_third_share": share,
                "gain": float(gain),
                "expansion_factor": factor,
                "reconciliation_rounds": int(rounds),
                "max_candidate_level_shift_pp": float((reached - targets).abs().max() * 100.0),
                "min_expanded_share": float(scaled.min()),
                "outcome_fields_used": "none",
            }
        )

    out[prediction_column] = expanded
    return out, pd.DataFrame(records)
