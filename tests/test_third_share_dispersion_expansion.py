"""Guards for the V30 third-share dispersion expansion.

The transform is promoted on three claims that are properties of its form
rather than of the panel it was measured on: it reads no outcome, it conserves
each candidate's national level, and it sizes itself by the same quantity that
diagnoses the compression - so an election without a third candidate is left
exactly alone. These tests pin all three, and pin that the gain stays the
parameter-free value rather than drifting to the better-scoring swept one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from presidential_issue_engine.third_share_dispersion_expansion import (
    DEFAULT_GAIN,
    apply_third_share_dispersion_expansion,
)


def _panel(third: float = 0.15) -> pd.DataFrame:
    """One election, three regions, three candidates with regional spread."""

    shares = {
        "r1": (0.55, 0.30, third),
        "r2": (0.40, 0.45, third),
        "r3": (0.30, 0.55, third),
    }
    rows = []
    for region, values in shares.items():
        total = sum(values)
        for name, value in zip(("A", "B", "C"), values):
            rows.append(
                {
                    "election_id": "pres_test",
                    "region_id": region,
                    "candidate_name": name,
                    "contest_votes": 100.0,
                    "layer_pred": value / total,
                }
            )
    return pd.DataFrame(rows)


def test_the_default_gain_is_the_parameter_free_value() -> None:
    """0.50 scores better on the panel; promoting it would be a fitted constant."""

    assert DEFAULT_GAIN == 1.0


def test_each_candidate_national_level_is_conserved() -> None:
    frame = _panel()
    adjusted, _ = apply_third_share_dispersion_expansion(frame)
    for name, part in frame.groupby("candidate_name"):
        after = adjusted.loc[adjusted.candidate_name.eq(name)]
        assert np.average(after.layer_pred, weights=after.contest_votes) == pytest.approx(
            np.average(part.layer_pred, weights=part.contest_votes), abs=1e-12
        )


def test_every_region_still_sums_to_one() -> None:
    adjusted, _ = apply_third_share_dispersion_expansion(_panel())
    totals = adjusted.groupby("region_id")["layer_pred"].sum()
    assert totals.round(12).eq(1.0).all()


def test_an_election_without_a_third_candidate_is_untouched() -> None:
    """2012 has two candidates; the transform must leave it exactly alone."""

    frame = _panel()
    frame = frame.loc[frame.candidate_name.ne("C")].copy()
    total = frame.groupby("region_id")["layer_pred"].transform("sum")
    frame["layer_pred"] = frame["layer_pred"] / total

    adjusted, audit = apply_third_share_dispersion_expansion(frame)
    assert float(audit.iloc[0]["predicted_third_share"]) == 0.0
    assert float(audit.iloc[0]["expansion_factor"]) == 1.0
    pd.testing.assert_series_equal(adjusted["layer_pred"], frame["layer_pred"])


def test_dispersion_grows_with_the_third_share_and_not_otherwise() -> None:
    """The index is the correction: a bigger third candidate means more expansion.

    A larger third share also changes the input spread through renormalisation,
    so the panels are not comparable directly - what must grow is the ratio of
    expanded spread to the spread the transform was handed.
    """

    def expansion(third: float) -> float:
        frame = _panel(third)
        adjusted, _ = apply_third_share_dispersion_expansion(frame)
        before = frame.loc[frame.candidate_name.eq("A"), "layer_pred"].std(ddof=1)
        after = adjusted.loc[adjusted.candidate_name.eq("A"), "layer_pred"].std(ddof=1)
        return float(after / before)

    assert expansion(0.02) > 1.0
    assert expansion(0.30) > expansion(0.02)


def test_no_outcome_column_is_required_or_read() -> None:
    """The frame carries no `actual`; the transform must still run."""

    frame = _panel()
    assert "actual" not in frame.columns
    adjusted, audit = apply_third_share_dispersion_expansion(frame)
    assert len(adjusted) == len(frame)
    assert (audit["outcome_fields_used"] == "none").all()


def test_a_negative_gain_is_refused() -> None:
    with pytest.raises(ValueError):
        apply_third_share_dispersion_expansion(_panel(), gain=-0.5)


def test_a_missing_required_column_names_itself() -> None:
    frame = _panel().drop(columns=["contest_votes"])
    with pytest.raises(KeyError, match="contest_votes"):
        apply_third_share_dispersion_expansion(frame)


def test_row_order_and_identity_are_preserved() -> None:
    frame = _panel()
    adjusted, _ = apply_third_share_dispersion_expansion(frame)
    pd.testing.assert_frame_equal(
        adjusted[["election_id", "region_id", "candidate_name"]],
        frame[["election_id", "region_id", "candidate_name"]],
    )

def test_the_feasibility_cap_keeps_the_level_conserved() -> None:
    """A gain large enough to drive a share negative must not inject vote mass.

    Clipping and renormalising would; stopping at the boundary does not. This is
    the 홍준표 2017 case, where 광주 and 전남 would otherwise go below zero.
    """

    frame = _panel(0.05)
    before = frame.groupby("candidate_name").apply(
        lambda p: np.average(p.layer_pred, weights=p.contest_votes)
    )
    adjusted, audit = apply_third_share_dispersion_expansion(frame, gain=100.0)

    assert bool(audit.iloc[0]["feasibility_capped"]), "gain 100 must hit the boundary"
    assert float(audit.iloc[0]["applied_factor"]) < float(
        audit.iloc[0]["expansion_factor"]
    )
    assert adjusted["layer_pred"].min() >= 0.0
    after = adjusted.groupby("candidate_name").apply(
        lambda p: np.average(p.layer_pred, weights=p.contest_votes)
    )
    pd.testing.assert_series_equal(after, before, atol=1e-12)


def test_the_applied_factor_is_uniform_across_candidates() -> None:
    """Per-candidate factors would make renormalisation move the levels.

    Candidate levels sum to one in every region, so one factor for the whole
    election leaves each region summing to one and renormalisation is a no-op.
    Different factors per candidate would break exactly that.
    """

    adjusted, audit = apply_third_share_dispersion_expansion(_panel(0.05), gain=100.0)
    assert len(audit) == 1
    assert "applied_factor" in audit.columns
    totals = adjusted.groupby("region_id")["layer_pred"].sum()
    assert totals.round(12).eq(1.0).all()
