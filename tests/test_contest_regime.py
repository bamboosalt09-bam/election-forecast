from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.contest_regime import (
    apply_contest_regime_response,
    conservative_core_floor,
    derive_contest_regimes,
)


def _frame(reliability: float = 0.8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": ["pres_test"] * 3,
            "region_id": ["r1"] * 3,
            "source_slot": ["A", "B", "C"],
            "pred": [0.44, 0.39, 0.17],
            "direct_party_recent_base": [0.44, 0.34, 0.12],
            "direct_party_reliability": [reliability] * 3,
            "core_voting_mass_effective": [0.35, 0.31, 0.04],
            "critical_voting_mass_effective": [0.06, 0.08, 0.03],
            "direct_party_core_raw": [0.34, 0.30, 0.08],
            "government_direction_score": [0.0, -0.2, 0.0],
            "direct_mega_score": [0.0, 0.0, 0.0],
            "incumbent_shock_log_shift": [0.0, -0.03, 0.0],
            "mega_issue_intensity_response": [1.0] * 3,
            "government_rejection_strength": [0.0, 0.18, 0.0],
            "actual": [0.0, 1.0, 0.0],
            "contest_votes": [1.0, 1000.0, 1.0],
        }
    )


def test_core_floor_is_discounted_and_never_exceeds_input_core() -> None:
    frame = _frame(reliability=0.6)
    floor = conservative_core_floor(frame)
    assert (floor <= frame["core_voting_mass_effective"]).all()
    assert floor.iloc[0] == pytest.approx(0.34 * 0.6)


def test_low_reliability_disables_margin_expansion() -> None:
    regimes = derive_contest_regimes(
        _frame(reliability=0.4), prediction_column="pred"
    )
    assert regimes.loc[0, "dominance_activation"] == 0.0
    assert regimes.loc[0, "contest_regime"] == "balanced_two_bloc"


def test_response_preserves_third_candidate_and_composition() -> None:
    frame = _frame()
    regimes = derive_contest_regimes(frame, prediction_column="pred")
    out = apply_contest_regime_response(
        frame, regimes, prediction_column="pred"
    )
    assert out["pred"].sum() == pytest.approx(1.0)
    assert out.loc[out["source_slot"].eq("C"), "pred"].iloc[0] == pytest.approx(0.17)
    assert out.loc[out["source_slot"].eq("A"), "pred"].iloc[0] > 0.44
    assert out.loc[out["source_slot"].eq("B"), "pred"].iloc[0] < 0.39


def test_response_keeps_core_fixed_and_moves_swing_more_than_critical() -> None:
    frame = _frame()
    regimes = derive_contest_regimes(frame, prediction_column="pred")
    out = apply_contest_regime_response(
        frame,
        regimes,
        prediction_column="pred",
        critical_elasticity=0.75,
        swing_elasticity=1.25,
        swing_log_shift_cap=0.50,
    )
    dominant = out.loc[out["source_slot"].eq("A")].iloc[0]
    assert dominant["swing_regime_log_shift"] > dominant[
        "critical_regime_log_shift"
    ] > 0.0
    assert dominant["pred"] >= dominant["regime_core_floor"]


def test_response_is_invariant_to_outcome_columns() -> None:
    frame = _frame()
    regimes = derive_contest_regimes(frame, prediction_column="pred")
    first = apply_contest_regime_response(
        frame, regimes, prediction_column="pred"
    )
    changed = frame.copy()
    changed["actual"] = [1.0, 0.0, 0.0]
    changed["contest_votes"] = [10000.0, 1.0, 1.0]
    second_regimes = derive_contest_regimes(changed, prediction_column="pred")
    second = apply_contest_regime_response(
        changed, second_regimes, prediction_column="pred"
    )
    assert second["pred"].tolist() == pytest.approx(first["pred"].tolist())


def test_cumulative_rejection_requires_erosion_or_rupture_route() -> None:
    frame = _frame()
    regimes = derive_contest_regimes(frame, prediction_column="pred")
    assert regimes.loc[0, "runner_cumulative_rejection"] > 0.0

    protected = frame.copy()
    protected.loc[protected["source_slot"].eq("B"), "direct_party_recent_base"] = 0.44
    protected.loc[protected["source_slot"].eq("B"), "pred"] = 0.44
    protected.loc[protected["source_slot"].eq("A"), "pred"] = 0.39
    protected_regimes = derive_contest_regimes(protected, prediction_column="pred")
    assert protected_regimes.loc[0, "dominant_cumulative_rejection"] == 0.0
    assert protected_regimes.loc[0, "runner_cumulative_rejection"] == 0.0
