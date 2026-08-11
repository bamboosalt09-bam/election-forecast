from __future__ import annotations

import numpy as np

from scripts.evaluate_preliminary_slot_shadow_nested import (
    _orthogonalize_predictor_pairs,
)


def test_orthogonalization_uses_training_predictors_only() -> None:
    predictors = ("base", "duplicate")
    train = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    test = np.array([[4.0, 9.0], [5.0, 20.0]])

    transformed_train, transformed_test, audit = _orthogonalize_predictor_pairs(
        train, test, predictors, (("base", "duplicate"),)
    )

    assert np.allclose(transformed_train[:, 1], 0.0, atol=1e-12)
    assert np.isclose(transformed_test[0, 1], 0.0, atol=1e-12)
    assert np.isclose(transformed_test[1, 1], 9.0, atol=1e-12)
    assert np.isclose(float(audit[0]["training_slope"]), 2.0)


def test_orthogonalization_does_not_accept_unknown_predictor() -> None:
    train = np.ones((3, 2))
    test = np.ones((1, 2))
    try:
        _orthogonalize_predictor_pairs(
            train, test, ("a", "b"), (("missing", "b"),)
        )
    except ValueError as error:
        assert "not in predictor set" in str(error)
    else:
        raise AssertionError("unknown predictor pair should fail")
