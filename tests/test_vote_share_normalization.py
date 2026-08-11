from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine
from presidential_issue_engine.issue_vote_engine import (
    apply_region_residual_calibration,
    normalize_vote_share_predictions,
    normalized_vote_share_target,
)


def test_normalize_vote_share_predictions_matches_region_total() -> None:
    frame = pd.DataFrame(
        [
            {"election_id": "pres_x", "region_id": "r1", "slot": "A", "vote_share": 0.4},
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "vote_share": 0.3},
            {"election_id": "pres_x", "region_id": "r1", "slot": "C", "vote_share": 0.2},
            {"election_id": "pres_x", "region_id": "r2", "slot": "A", "vote_share": 0.5},
            {"election_id": "pres_x", "region_id": "r2", "slot": "B", "vote_share": 0.4},
        ]
    )

    pred = normalize_vote_share_predictions(frame, np.array([0.8, 0.4, 0.4, 2.0, 1.0]))
    frame["pred"] = pred
    sums = frame.groupby(["election_id", "region_id"])[["pred"]].sum()

    assert sums.loc[("pres_x", "r1"), "pred"] == pytest.approx(1.0)
    assert sums.loc[("pres_x", "r2"), "pred"] == pytest.approx(1.0)


def test_normalize_vote_share_predictions_handles_nonpositive_group() -> None:
    frame = pd.DataFrame(
        [
            {"election_id": "pres_x", "region_id": "r1", "slot": "A", "vote_share": 0.6},
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "vote_share": 0.3},
        ]
    )

    pred = normalize_vote_share_predictions(frame, np.array([-1.0, -2.0]))

    assert pred.tolist() == pytest.approx([0.5, 0.5])


def test_normalized_vote_share_target_uses_modeled_candidate_ratio() -> None:
    frame = pd.DataFrame(
        [
            {"election_id": "pres_x", "region_id": "r1", "slot": "A", "vote_share": 0.4},
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "vote_share": 0.3},
            {"election_id": "pres_x", "region_id": "r1", "slot": "C", "vote_share": 0.2},
        ]
    )

    target = normalized_vote_share_target(frame)

    assert target.tolist() == pytest.approx([4 / 9, 3 / 9, 2 / 9])
    assert target.sum() == pytest.approx(1.0)


def test_region_residual_calibration_uses_shrunken_training_residuals(monkeypatch) -> None:
    train = pd.DataFrame(
        [
            {"election_id": "pres_x", "region_id": "r1", "slot": "A", "vote_share": 0.7},
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "vote_share": 0.3},
        ]
    )
    test = pd.DataFrame(
        [
            {"election_id": "pres_y", "region_id": "r1", "slot": "A", "vote_share": 0.5},
            {"election_id": "pres_y", "region_id": "r1", "slot": "B", "vote_share": 0.5},
        ]
    )
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_enabled", True)
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_scale", 1.0)
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_shrinkage", 0.0)

    adjusted = apply_region_residual_calibration(
        train,
        test,
        train_pred=np.array([0.6, 0.4]),
        test_pred=np.array([0.5, 0.5]),
    )

    assert adjusted.tolist() == pytest.approx([0.6, 0.4])


def test_region_residual_calibration_can_be_disabled(monkeypatch) -> None:
    train = pd.DataFrame(
        [
            {"election_id": "pres_x", "region_id": "r1", "slot": "A", "vote_share": 0.7},
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "vote_share": 0.3},
        ]
    )
    test = pd.DataFrame(
        [
            {"election_id": "pres_y", "region_id": "r1", "slot": "A", "vote_share": 0.5},
            {"election_id": "pres_y", "region_id": "r1", "slot": "B", "vote_share": 0.5},
        ]
    )
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_enabled", False)

    adjusted = apply_region_residual_calibration(train, test, [0.6, 0.4], [0.5, 0.5])

    assert adjusted.tolist() == pytest.approx([0.5, 0.5])


def test_region_residual_calibration_skips_when_prior_history_is_sufficient(monkeypatch) -> None:
    train = pd.DataFrame(
        [
            {"election_id": "pres_w", "region_id": "r1", "slot": "A", "vote_share": 0.7},
            {"election_id": "pres_w", "region_id": "r1", "slot": "B", "vote_share": 0.3},
            {"election_id": "pres_x", "region_id": "r1", "slot": "A", "vote_share": 0.7},
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "vote_share": 0.3},
        ]
    )
    test = pd.DataFrame(
        [
            {"election_id": "pres_y", "region_id": "r1", "slot": "A", "vote_share": 0.5},
            {"election_id": "pres_y", "region_id": "r1", "slot": "B", "vote_share": 0.5},
        ]
    )
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_enabled", True)
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_scale", 1.0)
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_shrinkage", 0.0)

    adjusted = apply_region_residual_calibration(
        train,
        test,
        train_pred=np.array([0.6, 0.4, 0.6, 0.4]),
        test_pred=np.array([0.5, 0.5]),
    )

    assert adjusted.tolist() == pytest.approx([0.5, 0.5])


def test_region_residual_calibration_prior_history_limit_cannot_be_broadened_by_env(
    monkeypatch,
) -> None:
    train = pd.DataFrame(
        [
            {"election_id": "pres_w", "region_id": "r1", "slot": "A", "vote_share": 0.7},
            {"election_id": "pres_w", "region_id": "r1", "slot": "B", "vote_share": 0.3},
            {"election_id": "pres_x", "region_id": "r1", "slot": "A", "vote_share": 0.7},
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "vote_share": 0.3},
        ]
    )
    test = pd.DataFrame(
        [
            {"election_id": "pres_y", "region_id": "r1", "slot": "A", "vote_share": 0.5},
            {"election_id": "pres_y", "region_id": "r1", "slot": "B", "vote_share": 0.5},
        ]
    )
    monkeypatch.setenv("POLL_PROJECT_REGION_RESIDUAL_CALIBRATION_MAX_PRIOR_ELECTIONS", "2")
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_enabled", True)
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_scale", 1.0)
    monkeypatch.setitem(issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG, "residual_shrinkage", 0.0)

    adjusted = apply_region_residual_calibration(
        train,
        test,
        train_pred=np.array([0.6, 0.4, 0.6, 0.4]),
        test_pred=np.array([0.5, 0.5]),
    )

    assert adjusted.tolist() == pytest.approx([0.5, 0.5])
