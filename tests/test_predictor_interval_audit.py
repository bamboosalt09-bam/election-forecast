from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine.audit_predictors_and_intervals import (
    _common_residual_draws,
)


def test_common_residual_draws_do_not_read_target_outcomes() -> None:
    train = pd.DataFrame(
        [
            {
                "election_id": "past",
                "region_id": region,
                "slot": slot,
                "votes": votes,
            }
            for region, votes in [("r1", 100), ("r2", 200)]
            for slot in ["A", "B"]
        ]
    )
    residuals = np.array([0.03, -0.03, 0.01, -0.01])
    test = pd.DataFrame(
        [
            {
                "election_id": "future",
                "region_id": region,
                "slot": slot,
                "vote_share": value,
            }
            for region, value in [("r1", 0.99), ("r2", 0.01)]
            for slot in ["A", "B"]
        ]
    )
    changed = test.copy()
    changed["vote_share"] = 1.0 - changed["vote_share"]

    draws, sigma = _common_residual_draws(
        np.random.default_rng(7), train, residuals, test, 50
    )
    changed_draws, changed_sigma = _common_residual_draws(
        np.random.default_rng(7), train, residuals, changed, 50
    )

    assert sigma == changed_sigma
    assert np.array_equal(draws, changed_draws)
    assert np.allclose(draws.reshape(50, 2, 2).sum(axis=2), 0.0)
