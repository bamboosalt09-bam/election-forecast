from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_active_presidential_model as active
from presidential_issue_engine import issue_vote_engine as engine


def _policy() -> dict[str, object]:
    return {
        "active": True,
        "variant": active.EXPECTED_VARIANT,
        "predictors": list(active.nested.BASE_PREDICTORS),
        "forbidden_predictors": sorted(active.nested.OLD_SLOT_PREDICTORS),
        "structural_layers": {
            "outer_config_overrides": {
                "conversion_scale": 0.05,
                "regionalism_scale": 0.15,
                "within_bloc_transfer_scale": 0.50,
                "within_bloc_stronghold_gain": 0.25,
            },
            "electorate_response": {
                "preference_gain_floor": 0.04,
                "concrete_support": {
                    "eligible_lineages": ["국민의힘", "더불어민주당"],
                    "matching": "exact_pre_normalization_party_lineage",
                    "other_lineage_core": 0.0,
                    "other_stable_support_reclassification": "critical_support",
                    "cross_candidate_core_sharing": False,
                },
                "terrain_anchor": {
                    "reliability_multiplier": 0.50,
                    "gain_cap": 0.25,
                    "mega_shock_attenuation": True,
                },
                "regional_accent": {
                    "source": "strictly_prior_direct_party_multiaxis_history",
                    "reliability_multiplier": 0.30,
                    "gain_cap": 0.20,
                    "signal_width": 0.10,
                    "core_policy": "inverse_core_share_mobility",
                    "mega_shock_attenuation": False,
                },
                "regional_swing_offset": {
                    "enabled": True,
                    "source": "rolling_nonpresidential_direct_party_ballots",
                    "method": "hierarchical_logit_offset",
                    "base_gain": 0.25,
                    "prior_strength": 2.0,
                    "minimum_prior_scored_elections": 2,
                    "vif_threshold": 20.0,
                    "activation_gate": "minimum_two_scored_elections_and_max_finite_vif_gt_20",
                    "third_candidate_mass_preserved": True,
                    "outcome_fields_used": [],
                },
            },
            "strategic_lane_transfer": {
                "enabled": True,
                "reservoir": "nonmajor_effective_critical_support",
                "recipient_pool": "aligned_major_party_candidates",
                "viability_source": "preliminary_expected_share",
                "affinity_power": 2.0,
                "outcome_fields_used": [],
            },
            "chungcheong_regional_identity": {
                "enabled": True,
                "reservoir_source": "strictly_prior_regional_third_bloc_excess",
                "routing_evidence": [
                    "candidate_regional_base",
                    "dated_pre_election_alignment",
                ],
                "gain": 0.50,
                "regional_shift_cap": 0.08,
                "half_life_years": 12.0,
                "prior_strength": 1.5,
                "unrouted_mass_policy": "remain_critical_or_swing",
                "outcome_fields_used": [],
            },
            "general_regional_identity": {
                "enabled": True,
                "region_scope": "non_chungcheong_with_dated_candidate_base_only",
                "distinctiveness_source": "strictly_prior_direct_party_and_downweighted_presidential_ballots",
                "routing_evidence": ["candidate_regional_base"],
                "donor_policy": "least_compatible_prior_regional_camp_first",
                "gain": 0.10,
                "regional_shift_cap": 0.04,
                "half_life_years": 12.0,
                "prior_strength": 1.5,
                "outcome_fields_used": [],
            },
            "party_context_cohesion": {
                "mode": "supporter_retention",
                "direct_vote_adjustment": False,
                "core_defection_cap": 0.02,
                "critical_defection_cap": 0.15,
                "released_mass_allocation": "regional_pre_adjustment_prediction",
                "candidate_conversion_direct_input": "nonparty_candidate_stature",
                "same_orientation_use": "within_bloc_dispersion_only",
            },
        },
        "postprocess": {
            "neutral_context_direct_adjustment": False,
            "withdrawn_candidate_transfer_adjustment": True,
            "direct_mega_issue_shift": True,
            "incumbent_shock_response": True,
            "government_burden_gain": 1.0,
            "rupture_extra_gain": 0.40,
            "incumbent_shock_log_shift_cap": 0.15,
            "contest_regime_response": True,
            "contest_regime_expansion_gain": 0.50,
            "contest_regime_log_shift_cap": 0.40,
            "contest_regime_critical_elasticity": 0.75,
            "contest_regime_swing_elasticity": 1.25,
            "contest_regime_swing_log_shift_cap": 0.50,
            "contest_regime_rejection_double_discount": False,
            "cumulative_regime_rejection": True,
            "cumulative_rejection_breadth_reference": 4,
            "cumulative_rejection_party_erosion_width": 0.08,
            "cumulative_rejection_conversion_buffer": 0.15,
            "cumulative_rejection_rupture_score_reference": 0.25,
        },
        "strict_nested_selection": {
            "enabled": True,
            "mode": "fixed_universal_evidence_gated_pipeline",
            "fixed_deployment_stage": "structural_mega_shock_regime",
            "minimum_prior_scored_elections": 2,
            "ordered_stages": [
                stage.name for stage in active.fully_nested_policy.ORDERED_STAGES
            ],
            "undated_issue_importance": "neutral_default_0.5",
            "undated_region_issue_sensitivity": "neutral_default_0.3",
        },
    }


def test_active_policy_forbids_realized_slots_and_keeps_final_transfer() -> None:
    policy = _policy()
    loaded = active.validate_policy(policy)
    assert loaded["variant"] == active.EXPECTED_VARIANT


def test_active_policy_rejects_disabled_transfer() -> None:
    policy = _policy()
    policy["postprocess"]["withdrawn_candidate_transfer_adjustment"] = False
    with pytest.raises(RuntimeError, match="retain final withdrawn-candidate transfer"):
        active.validate_policy(policy)


def test_active_policy_rejects_direct_party_context_vote_adjustment() -> None:
    policy = _policy()
    policy["structural_layers"]["party_context_cohesion"][
        "direct_vote_adjustment"
    ] = True
    with pytest.raises(RuntimeError, match="party-context cohesion policy"):
        active.validate_policy(policy)


def test_active_policy_rejects_nonmajor_concrete_support() -> None:
    policy = _policy()
    policy["structural_layers"]["electorate_response"]["concrete_support"][
        "other_lineage_core"
    ] = 0.5
    with pytest.raises(RuntimeError, match="major-party concrete-support policy"):
        active.validate_policy(policy)


def test_active_policy_rejects_outcome_driven_regional_offset() -> None:
    policy = _policy()
    policy["structural_layers"]["electorate_response"]["regional_swing_offset"][
        "outcome_fields_used"
    ] = ["actual_vote_share"]
    with pytest.raises(RuntimeError, match="regional-swing offset policy"):
        active.validate_policy(policy)


def test_active_policy_rejects_outcome_driven_strategic_transfer() -> None:
    policy = _policy()
    policy["structural_layers"]["strategic_lane_transfer"][
        "outcome_fields_used"
    ] = ["actual_vote_share"]
    with pytest.raises(RuntimeError, match="strategic-lane transfer policy"):
        active.validate_policy(policy)


def test_active_policy_rejects_outcome_driven_chungcheong_identity() -> None:
    policy = _policy()
    policy["structural_layers"]["chungcheong_regional_identity"][
        "outcome_fields_used"
    ] = ["actual_vote_share"]
    with pytest.raises(RuntimeError, match="Chungcheong regional-identity policy"):
        active.validate_policy(policy)


def test_active_policy_rejects_disabled_direct_mega_shift() -> None:
    policy = _policy()
    policy["postprocess"]["direct_mega_issue_shift"] = False
    with pytest.raises(RuntimeError, match="bounded direct mega-issue shift"):
        active.validate_policy(policy)


def test_active_policy_rejects_disabled_incumbent_shock_response() -> None:
    policy = _policy()
    policy["postprocess"]["incumbent_shock_response"] = False
    with pytest.raises(RuntimeError, match="incumbent-shock response"):
        active.validate_policy(policy)


def test_active_policy_rejects_disabled_contest_regime_response() -> None:
    policy = _policy()
    policy["postprocess"]["contest_regime_response"] = False
    with pytest.raises(RuntimeError, match="contest-regime response"):
        active.validate_policy(policy)


def test_active_policy_rejects_disabled_cumulative_regime_rejection() -> None:
    policy = _policy()
    policy["postprocess"]["cumulative_regime_rejection"] = False
    with pytest.raises(RuntimeError, match="cumulative regime rejection"):
        active.validate_policy(policy)


def test_active_policy_rejects_target_specific_stage_selection() -> None:
    policy = _policy()
    policy["strict_nested_selection"]["mode"] = "prior_fold_stage_selection"
    with pytest.raises(RuntimeError, match="deployment pipeline mode"):
        active.validate_policy(policy)


def test_fixed_pipeline_applies_same_stage_to_every_target() -> None:
    rows = []
    for stage in active.fully_nested_policy.ORDERED_STAGES:
        rows.append(
            pd.DataFrame(
                {
                    "election_id": list(active.nested.ELECTIONS),
                    "layer_pred": [stage.complexity / 10.0] * len(active.nested.ELECTIONS),
                }
            )
        )
    candidates = {
        stage.name: frame
        for stage, frame in zip(active.fully_nested_policy.ORDERED_STAGES, rows, strict=True)
    }
    predictions, audit = active._compose_fixed_pipeline_predictions(
        candidates, "structural_mega_shock_regime"
    )
    assert predictions["selected_stage"].eq("structural_mega_shock_regime").all()
    assert audit["selected_stage"].eq("structural_mega_shock_regime").all()
    assert audit["target_excluded_from_selection"].all()
    assert not audit["selection_fallback"].any()


def test_structural_terrain_gain_uses_prior_reliability_and_shock_attenuation() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2007", "pres_2007", "pres_2017", "pres_2017"],
            "direct_party_reliability": [0.60, 0.70, 0.80, 0.80],
        }
    )
    intensity = pd.DataFrame(
        {
            "election_id": ["pres_2007", "pres_2017"],
            "mega_issue_intensity": [1.0, 2.0],
            "available_date": ["2007-11-01", "2017-04-01"],
        }
    )
    gains, audit = active.structural_terrain_gain_by_target(
        frame,
        intensity,
        {
            "reliability_multiplier": 0.50,
            "gain_cap": 0.25,
            "mega_shock_attenuation": True,
        },
    )
    assert gains["pres_2007"] == pytest.approx(0.25)
    assert gains["pres_2017"] == pytest.approx(0.125)
    assert gains["pres_2002"] == pytest.approx(0.0)
    assert set(audit["target_election"]) == set(active.nested.ELECTIONS)


def test_strict_undated_curated_inputs_use_neutral_fallback(monkeypatch) -> None:
    monkeypatch.setenv(engine.STRICT_UNDATED_CURATED_INPUTS_ENV, "1")
    assert engine._issue_importance() == {}
    assert engine._region_issue_sensitivity().empty
