"""Guards for the party-regionalism retention experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import evaluate_party_regionalism_retention as experiment


def fixture() -> pd.DataFrame:
    rows = []
    for region, weight, a, b, pa, pb in (
        ("r1", 60.0, 0.55, 0.45, 0.80, 0.20),
        ("r2", 40.0, 0.45, 0.55, 0.20, 0.80),
    ):
        rows.extend(
            [
                dict(election_id="e", region_id=region, candidate_name="A", layer_pred=a, actual=a, contest_votes=weight, recent_bloc_base=pa, direct_party_reliability=1.0),
                dict(election_id="e", region_id=region, candidate_name="B", layer_pred=b, actual=b, contest_votes=weight, recent_bloc_base=pb, direct_party_reliability=1.0),
            ]
        )
    return pd.DataFrame(rows)


def test_zero_gain_is_identity() -> None:
    frame = fixture()
    result = experiment.retain(frame, 0.0)
    np.testing.assert_allclose(result.layer_pred, frame.layer_pred, atol=1e-10)


def test_calibration_preserves_regions_and_candidate_levels() -> None:
    frame = fixture()
    # Add a forecast-time third candidate so the activation is non-zero.
    third = pd.DataFrame(
        [
            dict(election_id="e", region_id="r1", candidate_name="C", layer_pred=0.05, actual=0.05, contest_votes=60.0, recent_bloc_base=0.05, direct_party_reliability=0.0),
            dict(election_id="e", region_id="r2", candidate_name="C", layer_pred=0.05, actual=0.05, contest_votes=40.0, recent_bloc_base=0.05, direct_party_reliability=0.0),
        ]
    )
    frame = pd.concat([frame, third], ignore_index=True)
    frame["layer_pred"] /= frame.groupby("region_id")["layer_pred"].transform("sum")
    before = frame.groupby("candidate_name").apply(
        lambda part: np.average(part.layer_pred, weights=part.contest_votes)
    )
    result = experiment.retain(frame, 4.0)
    after = result.groupby("candidate_name").apply(
        lambda part: np.average(part.layer_pred, weights=part.contest_votes)
    )
    np.testing.assert_allclose(after.sort_index(), before.sort_index(), atol=1e-9)
    np.testing.assert_allclose(
        result.groupby("region_id").layer_pred.sum().to_numpy(), 1.0, atol=1e-10
    )


def test_opposite_signed_realignments_are_not_bound() -> None:
    frame = fixture()
    frame.loc[frame.candidate_name.eq("A"), "recent_bloc_base"] = [0.20, 0.80]
    result = experiment.retain(frame, 4.0)
    assert not result.loc[result.candidate_name.eq("A"), "party_regionalism_floor_bound"].any()
