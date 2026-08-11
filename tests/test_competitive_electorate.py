from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_nested_competitive_electorate import (
    contestable_target,
    preserve_candidate_means,
    recompose_total_share,
)


def _two_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "camp_core_voting_mass": 0.30,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "camp_core_voting_mass": 0.10,
            },
        ]
    )


def test_contestable_target_and_recomposition_are_inverse() -> None:
    frame = _two_candidate_frame()
    actual = np.array([0.60, 0.40])

    contestable = contestable_target(frame, actual, strength=1.0)
    recomposed = recompose_total_share(frame, contestable, strength=1.0)

    assert contestable == pytest.approx([0.50, 0.50])
    assert recomposed == pytest.approx(actual)


def test_zero_strength_preserves_raw_ridge_predictions() -> None:
    frame = _two_candidate_frame()
    raw = np.array([0.57, 0.39])

    assert recompose_total_share(frame, raw, strength=0.0) == pytest.approx(raw)


def test_candidate_mean_raking_preserves_rows_and_baseline_columns() -> None:
    rows = [
        {"election_id": "target", "region_id": "r1", "slot": "A"},
        {"election_id": "target", "region_id": "r1", "slot": "B"},
        {"election_id": "target", "region_id": "r2", "slot": "A"},
        {"election_id": "target", "region_id": "r2", "slot": "B"},
    ]
    baseline = pd.DataFrame(rows)
    baseline["layer_pred"] = [0.55, 0.45, 0.45, 0.55]
    baseline["prior_region_weight"] = [3.0, 3.0, 1.0, 1.0]
    candidate = pd.DataFrame(rows)
    candidate["layer_pred"] = [0.80, 0.20, 0.30, 0.70]
    candidate["prior_region_weight"] = [3.0, 3.0, 1.0, 1.0]

    raked = preserve_candidate_means(candidate, baseline)

    assert raked.groupby(["election_id", "region_id"])["layer_pred"].sum().to_numpy() == pytest.approx(
        [1.0, 1.0]
    )
    raked_weighted = (
        raked["layer_pred"] * raked["prior_region_weight"]
    ).groupby(raked["slot"]).sum().sort_index()
    baseline_weighted = (
        baseline["layer_pred"] * baseline["prior_region_weight"]
    ).groupby(baseline["slot"]).sum().sort_index()
    assert raked_weighted.to_numpy() == pytest.approx(baseline_weighted.to_numpy())
