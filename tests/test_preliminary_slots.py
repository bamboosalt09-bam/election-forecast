from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_preliminary_slot_assignments import _withdrawal_profiles

from presidential_issue_engine.preliminary_slots import (
    PreliminarySlotConfig,
    apply_hierarchical_third_constraint,
    attenuate_withdrawn_endorsement_transfer,
    assign_preliminary_slots,
    assign_role_aware_slots,
    classify_competition_regime,
    latent_withdrawn_candidate_weight,
    redistribute_withdrawn_vote_mass,
    slot_free_predictors,
    third_viability_probability,
)


def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"election_id": "pres_test", "candidate_id": "a", "candidate_name": "A candidate", "preliminary_mean_share": 0.42, "preliminary_std": 0.03, "candidate_status": "active_ballot"},
            {"election_id": "pres_test", "candidate_id": "b", "candidate_name": "B candidate", "preliminary_mean_share": 0.36, "preliminary_std": 0.03, "candidate_status": "active_ballot"},
            {"election_id": "pres_test", "candidate_id": "c", "candidate_name": "C candidate", "preliminary_mean_share": 0.12, "preliminary_std": 0.04, "candidate_status": "active_ballot"},
            {"election_id": "pres_test", "candidate_id": "d", "candidate_name": "D candidate", "preliminary_mean_share": 0.10, "preliminary_std": 0.04, "candidate_status": "active_ballot"},
        ]
    )


def test_slot_predictors_are_removed() -> None:
    predictors = ("slot_A", "issue_advantage", "slotA_prior", "partisan_prior", "slot_B", "slotB_prior")
    assert slot_free_predictors(predictors) == ("issue_advantage", "partisan_prior")


def test_latent_withdrawn_weight_matches_two_stage_structure() -> None:
    weight = latent_withdrawn_candidate_weight(
        viability=0.55,
        centrist_appeal=0.80,
        anti_major_party_appeal=0.70,
        regional_base_overlap=0.35,
        confidence=0.70,
    )
    attention = 0.55 * (0.55 + 0.25 * 0.80 + 0.20 * 0.70) * 0.70
    conversion = 0.20 + 0.35 * 0.55 + 0.25 * 0.35 + 0.20 * 0.80
    assert weight**2 == pytest.approx(attention * conversion)


def test_withdrawn_vote_mass_is_conserved_across_targets_and_nonvoters() -> None:
    pre = {"A": 0.45, "B": 0.40, "C": 0.15}
    post, unconverted = redistribute_withdrawn_vote_mass(
        pre,
        source_slot="C",
        target_fractions={"A": 0.55 * 0.70, "B": 0.25 * 0.55},
    )
    assert post["C"] == 0.0
    assert post["A"] + post["B"] == pytest.approx(1.0)
    assert unconverted == pytest.approx(0.15 * (1.0 - 0.385 - 0.1375))
    active_mass = 0.45 + 0.40 + 0.15 * (0.385 + 0.1375)
    assert post["A"] == pytest.approx((0.45 + 0.15 * 0.385) / active_mass)


def test_support_withdrawal_attenuates_target_without_restoring_candidate() -> None:
    adjusted, retention = attenuate_withdrawn_endorsement_transfer(
        {"A": 0.60, "B": 0.10},
        target_slot="A",
        event_strength=1.0,
        voter_reach=0.35,
        days_to_election=1.0,
    )
    assert 0.0 < retention < 1.0
    assert adjusted["A"] == pytest.approx(0.60 * retention)
    assert adjusted["B"] == pytest.approx(0.10)


def test_support_withdrawal_is_more_effective_close_to_election() -> None:
    _, close = attenuate_withdrawn_endorsement_transfer(
        {"A": 0.60}, target_slot="A", event_strength=1.0, voter_reach=0.35, days_to_election=1.0
    )
    _, distant = attenuate_withdrawn_endorsement_transfer(
        {"A": 0.60}, target_slot="A", event_strength=1.0, voter_reach=0.35, days_to_election=30.0
    )
    assert close < distant


def test_2002_withdrawal_profile_uses_two_stage_point_in_time_events() -> None:
    profiles = _withdrawal_profiles().set_index("election_id")
    row = profiles.loc["pres_2002"]
    assert row["candidate_name"] == "Chung Mong-joon"
    assert row["available_date"] == "2002-12-18"
    assert 0.0 < row["support_retention"] < 1.0
    assert 0.0 < row["target_fractions"]["A"] < 0.80 * 0.75


def test_assigns_ab_and_continuous_c_without_dropping_minor() -> None:
    result = assign_preliminary_slots(candidates()).set_index("candidate_id")
    assert result.loc["a", "assigned_slot"] == "A"
    assert result.loc["b", "assigned_slot"] == "B"
    assert result.loc["c", "assigned_slot"] == "C"
    assert result.loc["d", "assigned_slot"] == "alpha"
    assert 0.0 < result.loc["c", "third_viability"] < 1.0
    assert result["preliminary_mean_share"].sum() == pytest.approx(1.0)


def test_role_aware_assignment_keeps_nonmajor_candidate_as_c_when_ranked_second() -> None:
    frame = candidates().iloc[:3].copy()
    frame["major_party_core_eligible"] = [True, True, False]
    frame.loc[frame["candidate_id"].eq("a"), "preliminary_mean_share"] = 0.29
    frame.loc[frame["candidate_id"].eq("b"), "preliminary_mean_share"] = 0.40
    frame.loc[frame["candidate_id"].eq("c"), "preliminary_mean_share"] = 0.31
    frame["automatic_third_viability"] = [0.0, 0.0, 0.72]

    ranked = assign_preliminary_slots(frame)
    result = assign_role_aware_slots(ranked).set_index("candidate_id")

    assert result.loc["b", "assigned_slot"] == "A"
    assert result.loc["a", "assigned_slot"] == "B"
    assert result.loc["c", "rank_slot"] == "B"
    assert result.loc["c", "assigned_slot"] == "C"
    assert result.loc["c", "political_role"] == "nonmajor_candidate"
    assert result.loc["c", "third_viability"] == pytest.approx(0.72)
    assert result["role_assignment_applied"].all()


def test_role_aware_assignment_falls_back_when_major_lineage_is_incomplete() -> None:
    frame = candidates().iloc[:3].copy()
    frame["major_party_core_eligible"] = [True, False, False]
    ranked = assign_preliminary_slots(frame)
    result = assign_role_aware_slots(ranked)
    assert result["assigned_slot"].tolist() == ranked["assigned_slot"].tolist()
    assert not result["role_assignment_applied"].any()


def test_withdrawn_candidate_is_transfer_reservoir_not_c() -> None:
    frame = candidates()
    frame.loc[frame["candidate_id"].eq("c"), "candidate_status"] = "withdrawn"
    result = assign_preliminary_slots(frame).set_index("candidate_id")
    assert result.loc["c", "assigned_slot"] == "withdrawn"
    assert result.loc["c", "competition_role"] == "transfer_reservoir"
    assert result.loc["d", "assigned_slot"] == "C"


def test_viability_is_monotonic_around_reference_share() -> None:
    low = third_viability_probability(0.03, standard_deviation=0.02)
    threshold = third_viability_probability(0.05, standard_deviation=0.02)
    high = third_viability_probability(0.08, standard_deviation=0.02)
    assert low < threshold < high
    assert threshold == pytest.approx(0.5)


def test_actual_outcome_columns_are_rejected() -> None:
    frame = candidates()
    frame["actual_vote_share"] = [0.4, 0.3, 0.2, 0.1]
    with pytest.raises(ValueError, match="target outcome columns"):
        assign_preliminary_slots(frame)


def test_post_cutoff_rows_are_rejected() -> None:
    frame = candidates()
    frame["forecast_date"] = "2022-03-08"
    frame["available_date"] = "2022-03-08"
    frame.loc[frame["candidate_id"].eq("d"), "available_date"] = "2022-03-09"
    with pytest.raises(ValueError, match="post-cutoff"):
        assign_preliminary_slots(frame)


def test_tie_break_and_gate_do_not_change_share() -> None:
    frame = candidates()
    frame.loc[frame["candidate_id"].isin(["a", "b"]), "preliminary_mean_share"] = 0.39
    before = frame.set_index("candidate_id")["preliminary_mean_share"].copy()
    result = assign_preliminary_slots(frame, PreliminarySlotConfig(major_third_probability=0.99)).set_index("candidate_id")
    assert result.loc["a", "assigned_slot"] == "A"
    assert result.loc["b", "assigned_slot"] == "B"
    pd.testing.assert_series_equal(result["preliminary_mean_share"].sort_index(), before.sort_index(), check_names=False)


def test_regime_classification_prevents_three_equal_category() -> None:
    assert classify_competition_regime(0.46, 0.42, 0.12, 0.9) == "two_strong_one_medium"
    assert classify_competition_regime(0.65, 0.20, 0.15, 0.9) == "one_strong_two_medium"
    assert classify_competition_regime(0.50, 0.47, 0.03, 0.1) == "two_strong_one_weak"


def test_hierarchy_pools_c_and_preserves_ab_ratio() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_test"] * 3,
            "region_id": ["region"] * 3,
            "slot": ["A", "B", "C"],
            "preliminary_mean_share": [0.48, 0.39, 0.13],
        }
    )
    before = pd.Series([0.36, 0.34, 0.30])
    adjusted, diagnostics = apply_hierarchical_third_constraint(frame, before)
    assert adjusted.sum() == pytest.approx(1.0)
    assert adjusted[2] < before.iloc[2]
    assert adjusted[2] <= 0.30
    assert adjusted[2] <= 0.95 * adjusted[1]
    assert adjusted[0] / adjusted[1] == pytest.approx(before.iloc[0] / before.iloc[1])
    assert len(diagnostics) == 1


def test_hierarchy_without_c_is_identity() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_test"] * 2,
            "region_id": ["region"] * 2,
            "slot": ["A", "B"],
            "preliminary_mean_share": [0.52, 0.48],
        }
    )
    before = pd.Series([0.51, 0.49])
    adjusted, diagnostics = apply_hierarchical_third_constraint(frame, before)
    assert adjusted.tolist() == pytest.approx(before.tolist())
    assert diagnostics.empty


def test_hierarchy_is_national_and_preserves_regional_shape() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_test"] * 6,
            "region_id": ["r1"] * 3 + ["r2"] * 3,
            "slot": ["A", "B", "C"] * 2,
            "preliminary_mean_share": [0.45, 0.40, 0.15] * 2,
        }
    )
    before = pd.Series([0.45, 0.35, 0.20, 0.30, 0.30, 0.40])
    adjusted, diagnostics = apply_hierarchical_third_constraint(frame, before)

    assert adjusted[:3].sum() == pytest.approx(1.0)
    assert adjusted[3:].sum() == pytest.approx(1.0)
    assert adjusted[0] / adjusted[1] == pytest.approx(before.iloc[0] / before.iloc[1])
    assert adjusted[3] / adjusted[4] == pytest.approx(before.iloc[3] / before.iloc[4])
    assert adjusted[5] > 0.30
    assert diagnostics.loc[0, "third_prediction_after"] <= 0.30
