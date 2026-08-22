"""Guards for the dispersion-calibration experiment.

The transform is measured and rejected, so these tests pin the properties that
made it measurable at all: that it reads no outcome, that it conserves each
regional contest, and that a zero gain is exactly the shipped model. If a later
change makes the transform look attractive, the rejection should be revisited
on evidence rather than because the harness drifted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_regional_dispersion_calibration as dispersion


def _frame() -> pd.DataFrame:
    rows = []
    for region, (a, b, c) in {
        "r1": (0.55, 0.30, 0.15),
        "r2": (0.40, 0.45, 0.15),
        "r3": (0.35, 0.50, 0.15),
    }.items():
        for name, pred, actual in (
            ("major_a", a, a + 0.02),
            ("major_b", b, b - 0.02),
            ("third", c, c),
        ):
            rows.append(
                {
                    "election_id": "pres_test",
                    "region_id": region,
                    "candidate_name": name,
                    "layer_pred": pred,
                    "actual": actual,
                    "contest_votes": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_a_zero_gain_leaves_the_shipped_prediction_alone() -> None:
    frame = _frame()
    result = dispersion.rescale(frame, 0.0)
    np.testing.assert_allclose(
        result.sort_values(["region_id", "candidate_name"])["layer_pred"].to_numpy(),
        frame.sort_values(["region_id", "candidate_name"])["layer_pred"].to_numpy(),
        atol=1e-12,
    )


def test_every_region_stays_compositional_at_any_gain() -> None:
    frame = _frame()
    for gain in (0.0, 0.5, 1.0, 2.0):
        result = dispersion.rescale(frame, gain)
        totals = result.groupby("region_id")["layer_pred"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-12)


def test_a_positive_gain_expands_rather_than_contracts_spread() -> None:
    frame = _frame()
    base = frame.loc[frame.candidate_name.eq("major_a"), "layer_pred"].std(ddof=1)
    widened = dispersion.rescale(frame, 1.0)
    after = widened.loc[widened.candidate_name.eq("major_a"), "layer_pred"].std(ddof=1)
    assert after > base


def test_the_index_is_the_predicted_third_share_not_the_realised_one() -> None:
    """The transform must remain usable before an outcome exists."""

    frame = _frame()
    name = dispersion._name_column(frame)
    expected = dispersion.predicted_third_share(frame, name)

    corrupted = frame.copy()
    corrupted["actual"] = 1.0 / 3.0  # destroy the outcome entirely
    assert dispersion.predicted_third_share(corrupted, name) == pytest.approx(expected)
    pd.testing.assert_series_equal(
        dispersion.rescale(frame, 1.0)["layer_pred"],
        dispersion.rescale(corrupted, 1.0)["layer_pred"],
    )


def test_an_election_without_a_third_candidate_is_untouched() -> None:
    frame = _frame()
    two_way = frame.loc[~frame.candidate_name.eq("third")].copy()
    totals = two_way.groupby("region_id")["layer_pred"].transform("sum")
    two_way["layer_pred"] = two_way["layer_pred"] / totals

    result = dispersion.rescale(two_way, 1.0)
    np.testing.assert_allclose(
        result.sort_values(["region_id", "candidate_name"])["layer_pred"].to_numpy(),
        two_way.sort_values(["region_id", "candidate_name"])["layer_pred"].to_numpy(),
        atol=1e-12,
    )
