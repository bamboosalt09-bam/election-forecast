from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.rejection_beneficiary_routing import (
    apply_rejection_beneficiary_routing,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": ["pres_x"] * 3,
            "region_id": ["r1"] * 3,
            "source_slot": ["A", "B", "C"],
            "layer_pred": [0.44, 0.33, 0.23],
            "core_voting_mass_effective": [0.12, 0.10, 0.0],
            "direct_party_core_raw": [0.14, 0.12, 0.0],
            "direct_party_reliability": [0.8, 0.8, 0.0],
            "major_party_core_eligible": [True, True, False],
        }
    )


def _regime(activation: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": ["pres_x"],
            "dominant_slot": ["A"],
            "runner_up_slot": ["B"],
            "regime_rejection_activation": [activation],
            "regime_certainty": [1.0],
            "cumulative_rejection_advantage": [0.20],
        }
    )


def test_rejection_routing_preserves_total_and_third_candidate() -> None:
    before = _frame()
    after, audit = apply_rejection_beneficiary_routing(
        before, _regime(), prediction_column="layer_pred"
    )
    assert after["layer_pred"].sum() == pytest.approx(1.0)
    assert after.loc[after.source_slot.eq("C"), "layer_pred"].iloc[0] == pytest.approx(0.23)
    assert after.loc[after.source_slot.eq("A"), "layer_pred"].iloc[0] > 0.44
    assert after.loc[after.source_slot.eq("B"), "layer_pred"].iloc[0] > 0.08
    assert audit["transfer"].iloc[0] == pytest.approx(0.20 * (0.33 - 0.08))


def test_rejection_routing_is_identity_without_activation() -> None:
    before = _frame()
    after, audit = apply_rejection_beneficiary_routing(
        before, _regime(0.0), prediction_column="layer_pred"
    )
    assert after["layer_pred"].tolist() == pytest.approx(before["layer_pred"].tolist())
    assert audit.empty
