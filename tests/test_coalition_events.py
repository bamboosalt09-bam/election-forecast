from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.issue_vote_engine import (
    PREDICTORS,
    _apply_coalition_events,
    _excluded_event_slots,
    _official_issue_signal_weights,
    assemble,
    monte_carlo,
    scored_contest_rows,
)


def test_apply_coalition_events_transfers_issue_features() -> None:
    issue_advantage = pd.DataFrame(
        [
            {"election_id": "pres_x", "slot": "B", "issue_advantage": 0.2},
            {"election_id": "pres_x", "slot": "C", "issue_advantage": 0.1},
        ]
    )
    region_issue_fit = pd.DataFrame(
        [
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "rif": 0.03},
            {"election_id": "pres_x", "region_id": "r1", "slot": "C", "rif": 0.02},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "source_slot": "C",
                "target_slot": "B",
                "transfer_rate": 0.8,
                "voter_compliance": 0.5,
                "source_viability_after_event": 0.1,
                "exclude_source_from_evaluation": True,
            }
        ]
    )

    adv, rif = _apply_coalition_events(issue_advantage, region_issue_fit, events)

    assert adv.loc[adv["slot"] == "B", "issue_advantage"].iloc[0] == pytest.approx(0.24)
    assert adv.loc[adv["slot"] == "C", "issue_advantage"].iloc[0] == pytest.approx(0.01)
    assert rif.loc[rif["slot"] == "B", "rif"].iloc[0] == pytest.approx(0.038)
    assert rif.loc[rif["slot"] == "C", "rif"].iloc[0] == pytest.approx(0.002)


def test_excluded_event_slots() -> None:
    events = pd.DataFrame(
        [
            {"election_id": "pres_2012", "source_slot": "C", "exclude_source_from_evaluation": True},
            {"election_id": "pres_2022", "source_slot": "C", "exclude_source_from_evaluation": False},
        ]
    )

    assert _excluded_event_slots(events) == {("pres_2012", "C")}


def test_assemble_excludes_2012_withdrawn_c_slot() -> None:
    frame = scored_contest_rows(assemble())

    pres_2002 = frame.loc[frame["election_id"] == "pres_2002"]
    assert not pres_2002.empty
    assert pres_2002["regional_base"].abs().max() > 0
    assert frame.loc[(frame["election_id"] == "pres_2012") & (frame["slot"] == "C")].empty
    assert frame.loc[(frame["election_id"] == "pres_2022") & (frame["slot"] == "C")].empty


def test_stats_monte_carlo_output_has_consistent_intervals_and_diagnostics() -> None:
    frame = scored_contest_rows(assemble())
    out = monte_carlo(frame, PREDICTORS, n_sim=50)

    assert "issue_advantage" in PREDICTORS
    assert "landscape_bloc_alignment" in PREDICTORS
    assert {"issue_attention_score", "support_conversion_score", "issue_attention_overhang"}.issubset(
        out.columns
    )
    assert {"party_context_support", "party_elite_fragmentation_score", "organization_strength"}.issubset(
        out.columns
    )
    assert {"serious_contender_score", "public_treatment_support_centered"}.issubset(out.columns)
    assert {
        "prediction_residual_sigma",
        "prediction_residual_variance",
        "prediction_residual_common_sigma",
        "prediction_residual_local_sigma",
        "prediction_residual_structure",
        "interval_includes_residual_uncertainty",
        "mean_interval_includes_residual_uncertainty",
        "neutral_issue_context_scale",
    }.issubset(out.columns)
    assert (out["prediction_residual_sigma"] > 0).all()
    assert (out["prediction_residual_common_sigma"] > 0).all()
    assert (out["prediction_residual_local_sigma"] > 0).all()
    assert set(out["prediction_residual_structure"]) == {"common_shock"}
    assert not out["interval_includes_local_residual_uncertainty"].any()
    assert out["interval_includes_residual_uncertainty"].all()
    assert not out["mean_interval_includes_residual_uncertainty"].any()
    for level in (90, 95, 99):
        assert ((out[f"lo{level}"] <= out["pred"]) & (out["pred"] <= out[f"hi{level}"])).all()
        assert (
            (out[f"mean_lo{level}"] <= out["pred"])
            & (out["pred"] <= out[f"mean_hi{level}"])
        ).all()
        assert (
            out[f"mean_hi{level}"] - out[f"mean_lo{level}"]
        ).mean() < (out[f"hi{level}"] - out[f"lo{level}"]).mean()
    assert (out["lo99"] <= out["lo95"]).all()
    assert (out["lo95"] <= out["lo90"]).all()
    assert (out["hi90"] <= out["hi95"]).all()
    assert (out["hi95"] <= out["hi99"]).all()
    assert (out["mean_lo99"] <= out["mean_lo95"]).all()
    assert (out["mean_lo95"] <= out["mean_lo90"]).all()
    assert (out["mean_hi90"] <= out["mean_hi95"]).all()
    assert (out["mean_hi95"] <= out["mean_hi99"]).all()
    assert "assembly_neutral_issue_signal" in out.columns
    assert {"contest_type", "contest_active_slots", "contest_votes", "contest_vote_share"}.issubset(
        out.columns
    )
    assert set(out.loc[out["election_id"] == "pres_2022", "contest_type"]) == {"two_way"}
    assert set(out.loc[out["election_id"] == "pres_2002", "contest_type"]) == {"two_way"}
    assert set(out.loc[out["election_id"] == "pres_2017", "contest_type"]) == {"three_way"}
    assert (
        out.groupby(["election_id", "region_id"])["contest_vote_share"].sum().round(10) == 1.0
    ).all()
    assert not out[["housing_price_period", "housing_baseline_period", "housing_current_period"]].isna().any().any()


def test_official_issue_signal_weights_use_prior_viability_and_activity() -> None:
    results = pd.DataFrame(
        [
            {"election_id": "pres_2002", "slot": "A", "vote_share": 0.51, "is_active_slot": True},
            {"election_id": "pres_2002", "slot": "C", "vote_share": 0.03, "is_active_slot": True},
            {"election_id": "pres_2007", "slot": "A", "vote_share": 0.43, "is_active_slot": True},
            {"election_id": "pres_2007", "slot": "C", "vote_share": 0.16, "is_active_slot": True},
            {"election_id": "pres_2012", "slot": "C", "vote_share": 0.01, "is_active_slot": False},
        ]
    )

    weights = _official_issue_signal_weights(results, {("pres_2012", "C")})
    pres_2007_a = weights.loc[
        (weights["election_id"] == "pres_2007") & (weights["slot"] == "A"),
        "issue_signal_weight",
    ].iloc[0]
    pres_2007_c = weights.loc[
        (weights["election_id"] == "pres_2007") & (weights["slot"] == "C"),
        "issue_signal_weight",
    ].iloc[0]
    pres_2012_c = weights.loc[
        (weights["election_id"] == "pres_2012") & (weights["slot"] == "C"),
        "issue_signal_weight",
    ].iloc[0]

    assert pres_2007_a == pytest.approx(0.30)
    assert pres_2007_c == pytest.approx(0.26)
    assert pres_2012_c == pytest.approx(0.0)
