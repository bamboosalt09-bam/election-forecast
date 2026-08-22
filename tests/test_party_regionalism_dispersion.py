"""Guards for core-weighted party-regional dispersion."""

import numpy as np

from scripts import evaluate_party_regionalism_dispersion as experiment
from tests.test_party_regionalism_retention import fixture


def test_zero_gain_is_identity() -> None:
    frame = fixture()
    frame["core_voting_mass"] = 0.5
    result = experiment.expand(frame, 0.0)
    np.testing.assert_allclose(result.layer_pred, frame.layer_pred, atol=1e-10)


def test_no_expansion_when_prior_is_not_wider() -> None:
    frame = fixture()
    frame["core_voting_mass"] = 0.5
    frame["recent_bloc_base"] = frame["layer_pred"]
    result = experiment.expand(frame, 1.0)
    np.testing.assert_allclose(result.layer_pred, frame.layer_pred, atol=1e-10)


def test_expansion_preserves_candidate_levels_and_region_sums() -> None:
    frame = fixture()
    frame["core_voting_mass"] = 0.5
    before = frame.groupby("candidate_name").apply(
        lambda part: np.average(part.layer_pred, weights=part.contest_votes)
    )
    result = experiment.expand(frame, 1.0)
    after = result.groupby("candidate_name").apply(
        lambda part: np.average(part.layer_pred, weights=part.contest_votes)
    )
    assert result.party_regionalism_dispersion_factor.max() > 1.0
    np.testing.assert_allclose(before.sort_index(), after.sort_index(), atol=1e-9)
    np.testing.assert_allclose(
        result.groupby("region_id").layer_pred.sum().to_numpy(), 1.0, atol=1e-10
    )
