from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from presidential_issue_engine.v24_calibration import (
    apply_national_preserving_regional_shape,
    draw_region_weight_uncertainty,
    hierarchical_residual_draws,
)


def _frame(election_id: str = "pres_2012") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": [election_id] * 4,
            "region_id": ["r1", "r1", "r2", "r2"],
            "slot": ["A", "B", "A", "B"],
            "pred": [0.6, 0.4, 0.4, 0.6],
            "actual": [0.65, 0.35, 0.35, 0.65],
            "regional_accent_signal_scaled": [0.8, -0.8, -0.8, 0.8],
            "regional_accent_reliability": [0.75] * 4,
            "core_voting_mass_effective": [0.4] * 4,
            "candidate_camp": ["left", "right", "left", "right"],
            "contest_votes": [100.0, 100.0, 300.0, 300.0],
        }
    )


def test_regional_shape_preserves_rows_and_forecast_national_totals() -> None:
    frame = _frame()
    weights = pd.Series({"r1": 0.25, "r2": 0.75})
    result = apply_national_preserving_regional_shape(frame, weights, gain=0.2)

    assert np.allclose(result.groupby("region_id")["v24_regional_shape_pred"].sum(), 1.0)
    base = frame.assign(weight=frame["region_id"].map(weights)).groupby("slot").apply(
        lambda group: float((group["pred"] * group["weight"]).sum()),
        include_groups=False,
    )
    adjusted = result.groupby("slot").apply(
        lambda group: float(
            (group["v24_regional_shape_pred"] * group["v24_forecast_region_weight"]).sum()
        ),
        include_groups=False,
    )
    assert np.allclose(base.sort_index(), adjusted.sort_index(), atol=1e-10)


def test_regional_shape_zero_gain_is_exact_noop() -> None:
    frame = _frame()
    result = apply_national_preserving_regional_shape(
        frame,
        pd.Series({"r1": 0.5, "r2": 0.5}),
        gain=0.0,
    )
    assert np.allclose(result["pred"], result["v24_regional_shape_pred"], atol=1e-14)


def test_hierarchical_draws_are_compositional_deterministic_and_target_blind() -> None:
    train = apply_national_preserving_regional_shape(
        _frame("pres_2007"),
        pd.Series({"r1": 0.5, "r2": 0.5}),
        gain=0.1,
    )
    target = apply_national_preserving_regional_shape(
        _frame("pres_2012"),
        pd.Series({"r1": 0.5, "r2": 0.5}),
        gain=0.1,
    ).drop(columns="actual")
    draws, components = hierarchical_residual_draws(
        train, target, n_sim=100, seed=9
    )
    changed = target.copy()
    changed["irrelevant_target_outcome"] = [0.99, 0.01, 0.99, 0.01]
    changed_draws, changed_components = hierarchical_residual_draws(
        train, changed, n_sim=100, seed=9
    )

    assert components == changed_components
    assert np.array_equal(draws, changed_draws)
    assert np.allclose(draws[:, :2].sum(axis=1), 1.0)
    assert np.allclose(draws[:, 2:].sum(axis=1), 1.0)

    common_only, _ = hierarchical_residual_draws(
        train,
        target,
        n_sim=100,
        seed=9,
        regional_multiplier=0.0,
        local_multiplier=0.0,
    )
    assert not np.array_equal(draws, common_only)

    empirical, _ = hierarchical_residual_draws(
        train, target, n_sim=100, seed=9, distribution="empirical"
    )
    assert np.allclose(empirical[:, :2].sum(axis=1), 1.0)


def test_hierarchical_draws_reject_target_in_training() -> None:
    frame = apply_national_preserving_regional_shape(
        _frame(), pd.Series({"r1": 0.5, "r2": 0.5}), gain=0.1
    )
    with pytest.raises(ValueError, match="target election"):
        hierarchical_residual_draws(frame, frame, n_sim=10, seed=1)


def test_region_weight_draws_use_only_prior_elections_and_sum_to_one() -> None:
    prior = apply_national_preserving_regional_shape(
        _frame("pres_2007"), pd.Series({"r1": 0.5, "r2": 0.5}), gain=0.0
    )
    older = apply_national_preserving_regional_shape(
        _frame("pres_2002"), pd.Series({"r1": 0.5, "r2": 0.5}), gain=0.0
    )
    older.loc[older["region_id"].eq("r1"), "contest_votes"] = 80.0
    target = apply_national_preserving_regional_shape(
        _frame("pres_2012"), pd.Series({"r1": 0.25, "r2": 0.75}), gain=0.0
    ).drop(columns="actual")
    train = pd.concat([older, prior], ignore_index=True)
    regions, draws, components = draw_region_weight_uncertainty(
        train, target, n_sim=100, seed=11
    )

    assert regions == ["r1", "r2"]
    assert components.training_transitions == 1
    assert np.allclose(draws.sum(axis=1), 1.0)
    changed = target.copy()
    changed["irrelevant_target_outcome"] = 0.99
    _, changed_draws, _ = draw_region_weight_uncertainty(
        train, changed, n_sim=100, seed=11
    )
    assert np.array_equal(draws, changed_draws)
