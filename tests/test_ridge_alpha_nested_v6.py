from __future__ import annotations

import pandas as pd
import pytest

from scripts import evaluate_ridge_alpha_nested_v6 as experiment


def test_alpha_grid_contains_active_baseline_and_near_unregularized_case() -> None:
    assert experiment.BASELINE_MULTIPLIER in experiment.ALPHA_MULTIPLIERS
    assert 0.0 in experiment.ALPHA_MULTIPLIERS
    assert 0.01 in experiment.ALPHA_MULTIPLIERS


def test_margin_diagnostic_uses_actual_winner_against_best_rival() -> None:
    national = pd.DataFrame(
        {
            "election_id": ["pres_test"] * 3,
            "candidate_key": ["A", "B", "C"],
            "candidate_name": ["A-name", "B-name", "C-name"],
            "pred_pct": [45.0, 40.0, 15.0],
            "actual_pct": [42.0, 48.0, 10.0],
        }
    )
    result = experiment.margin_diagnostics(national, 0.30).iloc[0]
    assert result["actual_winner"] == "B-name"
    assert result["predicted_winner"] == "A-name"
    assert not bool(result["winner_correct"])
    assert result["predicted_actual_winner_margin_pp"] == pytest.approx(-5.0)
    assert result["actual_margin_pp"] == pytest.approx(6.0)
    assert result["margin_error_pp"] == pytest.approx(-11.0)
