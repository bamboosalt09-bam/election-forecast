from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import contest_regime
from scripts import evaluate_cumulative_regime_rejection as experiment


def test_v5_contest_response_is_inverted_before_v6_evaluation() -> None:
    original = pd.DataFrame(
        {
            "election_id": ["pres_test"] * 3,
            "region_id": ["r1"] * 3,
            "source_slot": ["A", "B", "C"],
            "layer_pred": [0.46, 0.36, 0.18],
            "core_voting_mass_effective": [0.20, 0.18, 0.05],
            "direct_party_core_raw": [0.22, 0.19, 0.06],
            "direct_party_reliability": [0.80, 0.75, 0.50],
        }
    )
    regimes = pd.DataFrame(
        {
            "election_id": ["pres_test"],
            "dominant_slot": ["A"],
            "runner_up_slot": ["B"],
            "dominance_activation": [0.60],
        }
    )
    shifted = contest_regime.apply_contest_regime_response(
        original,
        regimes,
        prediction_column="layer_pred",
        expansion_gain=0.50,
        log_shift_cap=0.30,
    )
    recovered = experiment._remove_existing_contest_response(shifted)
    assert recovered["precontest_pred"].tolist() == pytest.approx(
        original["layer_pred"].tolist(), abs=1e-12
    )

