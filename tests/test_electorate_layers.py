from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from presidential_issue_engine.electorate_layers import (
    ElectorateLayerConfig,
    apply_electorate_layer_response,
    apply_electorate_layer_response_draws,
    compile_issue_class_signals,
    estimate_electorate_layers,
)
from presidential_issue_engine import issue_vote_engine


DATES = {
    "old": pd.Timestamp("2000-01-01"),
    "target": pd.Timestamp("2004-01-01"),
    "future": pd.Timestamp("2008-01-01"),
}


def _date_resolver(value: str) -> pd.Timestamp | None:
    return DATES.get(value)


def test_layer_estimation_is_point_in_time_and_compositional() -> None:
    candidates = pd.DataFrame(
        [
            {"election_id": "target", "region_id": "r1", "slot": "A", "bloc": "국민의힘"},
            {"election_id": "target", "region_id": "r1", "slot": "B", "bloc": "더불어민주당"},
        ]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "old",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "국민의힘",
                "vote_share": 0.60,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "old",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "더불어민주당",
                "vote_share": 0.40,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "future",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "국민의힘",
                "vote_share": 0.05,
                "data_quality_weight": 1.0,
            },
        ]
    )

    out = estimate_electorate_layers(candidates, history, date_resolver=_date_resolver)

    conservative = out.loc[out["slot"].eq("A")].iloc[0]
    assert conservative["recent_bloc_base"] == pytest.approx(0.60)
    assert conservative["core_voting_mass"] > 0.0
    assert (out["core_voting_mass"] >= 0.0).all()
    assert (out["critical_voting_mass"] >= 0.0).all()
    total = out["core_voting_mass"].sum() + out["critical_voting_mass"].sum()
    assert total <= 0.90 + 1e-12
    assert out["swing_voting_mass"].iloc[0] == pytest.approx(1.0 - total)


def test_concrete_support_is_reserved_for_two_major_party_lineages() -> None:
    candidates = pd.DataFrame(
        [
            {"election_id": "target", "region_id": "r1", "slot": "A", "bloc": "국민의힘"},
            {"election_id": "target", "region_id": "r1", "slot": "B", "bloc": "더불어민주당"},
            {"election_id": "target", "region_id": "r1", "slot": "C", "bloc": "진보정당계"},
            {"election_id": "target", "region_id": "r1", "slot": "D", "bloc": "제3지대"},
            {"election_id": "target", "region_id": "r1", "slot": "E", "bloc": "우리공화당"},
        ]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "old",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": bloc,
                "vote_share": share,
                "data_quality_weight": 1.0,
            }
            for bloc, share in (
                ("국민의힘", 0.30),
                ("우리공화당", 0.10),
                ("더불어민주당", 0.30),
                ("진보정당계", 0.15),
                ("제3지대", 0.15),
            )
        ]
    )

    out = estimate_electorate_layers(
        candidates,
        history,
        date_resolver=_date_resolver,
        mass_profile="direct_party_layers",
    ).set_index("slot")

    assert out.loc["A", "core_voting_mass"] > 0.0
    assert out.loc["B", "core_voting_mass"] > 0.0
    assert out.loc[["C", "D", "E"], "core_voting_mass"].eq(0.0).all()
    assert out.loc[["C", "D", "E"], "direct_party_core_raw"].eq(0.0).all()
    assert out.loc[["C", "D", "E"], "critical_voting_mass"].gt(0.0).all()
    assert bool(out.loc["A", "major_party_core_eligible"])
    assert not bool(out.loc["E", "major_party_core_eligible"])


def test_durable_floor_broadens_critical_support_without_future_data() -> None:
    candidates = pd.DataFrame(
        [
            {"election_id": "target", "region_id": "r1", "slot": "A", "bloc": "국민의힘"},
            {"election_id": "target", "region_id": "r1", "slot": "B", "bloc": "더불어민주당"},
        ]
    )
    dates = {
        "e1": pd.Timestamp("1996-01-01"),
        "e2": pd.Timestamp("1998-01-01"),
        "e3": pd.Timestamp("2000-01-01"),
        "target": pd.Timestamp("2002-01-01"),
        "future": pd.Timestamp("2004-01-01"),
    }
    rows = []
    for election_id, share_a, share_b in (
        ("e1", 0.20, 0.55),
        ("e2", 0.40, 0.40),
        ("e3", 0.50, 0.35),
        ("future", 0.90, 0.05),
    ):
        rows.extend(
            [
                {
                    "election_id": election_id,
                    "election_type": "national_assembly_pr",
                    "region_id": "r1",
                    "bloc": "국민의힘",
                    "vote_share": share_a,
                    "data_quality_weight": 1.0,
                },
                {
                    "election_id": election_id,
                    "election_type": "national_assembly_pr",
                    "region_id": "r1",
                    "bloc": "더불어민주당",
                    "vote_share": share_b,
                    "data_quality_weight": 1.0,
                },
            ]
        )
    history = pd.DataFrame(rows)
    resolver = dates.get
    legacy = estimate_electorate_layers(candidates, history, date_resolver=resolver)
    challenger = estimate_electorate_layers(
        candidates,
        history,
        date_resolver=resolver,
        mass_profile="durable_floor_broad_critical",
    )

    assert challenger["core_voting_mass"].sum() < legacy["core_voting_mass"].sum()
    assert challenger["critical_voting_mass"].sum() > legacy["critical_voting_mass"].sum()
    assert challenger["recent_bloc_base"].to_numpy() == pytest.approx(
        legacy["recent_bloc_base"].to_numpy()
    )


def test_unknown_electorate_mass_profile_is_rejected() -> None:
    candidates = pd.DataFrame(
        [{"election_id": "target", "region_id": "r1", "slot": "A", "bloc": "bloc_a"}]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "old",
                "election_type": "national_assembly_pr",
                "region_id": "r1",
                "bloc": "bloc_a",
                "vote_share": 0.4,
                "data_quality_weight": 1.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="unknown electorate mass profile"):
        estimate_electorate_layers(
            candidates,
            history,
            date_resolver=_date_resolver,
            mass_profile="unknown",
        )


def test_direct_party_ballots_are_distinguished_from_candidate_ballots() -> None:
    candidates = pd.DataFrame(
        [
            {"election_id": "target", "region_id": "r1", "slot": "A", "bloc": "국민의힘"},
            {"election_id": "target", "region_id": "r1", "slot": "B", "bloc": "더불어민주당"},
        ]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "old",
                "election_type": election_type,
                "region_id": "r1",
                "bloc": bloc,
                "vote_share": share,
                "data_quality_weight": 1.0,
            }
            for election_type, bloc, share in (
                ("assembly_pr", "국민의힘", 0.45),
                ("assembly_district", "국민의힘", 0.65),
                ("assembly_pr", "더불어민주당", 0.55),
                ("assembly_district", "더불어민주당", 0.35),
            )
        ]
    )

    out = estimate_electorate_layers(candidates, history, date_resolver=_date_resolver)
    candidate_a = out.loc[out["slot"].eq("A")].iloc[0]

    assert candidate_a["direct_party_core_raw"] == pytest.approx(0.45)
    assert candidate_a["candidate_ballot_core_raw"] == pytest.approx(0.65)
    assert 0.0 < candidate_a["direct_party_reliability"] < 1.0
    assert 0.45 < candidate_a["durable_core_raw"] < 0.65


def test_direct_party_layer_keeps_candidate_popularity_out_of_party_base() -> None:
    candidates = pd.DataFrame(
        [{"election_id": "target", "region_id": "r1", "slot": "A", "bloc": "국민의힘"}]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "old",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "국민의힘",
                "vote_share": 0.40,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "old",
                "election_type": "assembly_district",
                "region_id": "r1",
                "bloc": "국민의힘",
                "vote_share": 0.60,
                "data_quality_weight": 1.0,
            },
        ]
    )
    legacy = estimate_electorate_layers(
        candidates, history, date_resolver=_date_resolver
    ).iloc[0]
    separated = estimate_electorate_layers(
        candidates,
        history,
        date_resolver=_date_resolver,
        mass_profile="direct_party_layers",
    ).iloc[0]

    assert legacy["recent_bloc_base"] > 0.40
    assert separated["recent_bloc_base"] == pytest.approx(
        legacy["recent_bloc_base"] - 0.025
    )
    assert abs(separated["core_voting_mass"] - legacy["core_voting_mass"]) <= 0.03 + 1e-12
    assert (
        abs(separated["critical_voting_mass"] - legacy["critical_voting_mass"])
        <= 0.03 + 1e-12
    )
    assert separated["candidate_personal_vote_raw"] == pytest.approx(0.20)
    assert separated["candidate_conversion_gap_raw"] == pytest.approx(0.20)


def test_same_camp_candidates_share_one_regional_core() -> None:
    candidates = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "bloc": "국민의힘",
                "landscape_axis_conservative": 0.8,
                "landscape_axis_centrist": 0.2,
                "landscape_confidence": 1.0,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "C",
                "bloc": "무소속",
                "landscape_axis_conservative": 0.8,
                "landscape_axis_centrist": 0.2,
                "landscape_confidence": 1.0,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "old",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "국민의힘",
                "vote_share": 0.60,
                "data_quality_weight": 1.0,
            }
        ]
    )
    out = estimate_electorate_layers(
        candidates,
        history,
        date_resolver=_date_resolver,
        mass_profile="direct_party_layers",
    )

    assert out["candidate_camp"].nunique() == 1
    assert out["candidate_camp_claim"].sum() == pytest.approx(1.0)
    assert out.loc[out["slot"].eq("C"), "candidate_camp_claim"].iloc[0] > 0.0
    assert out["camp_core_voting_mass"].sum() == pytest.approx(
        out["camp_core_total"].iloc[0]
    )


def test_regional_accent_distinguishes_liberal_and_progressive_lanes() -> None:
    candidates = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": region,
                "slot": slot,
                "bloc": bloc,
                "landscape_axis_liberal": liberal,
                "landscape_axis_progressive": progressive,
                "landscape_confidence": 1.0,
            }
            for region in ("r1", "r2")
            for slot, bloc, liberal, progressive in (
                ("A", "더불어민주당", 1.0, 0.0),
                ("C", "진보정당계", 0.0, 1.0),
            )
        ]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "old",
                "election_type": "assembly_pr",
                "region_id": region,
                "bloc": bloc,
                "vote_share": share,
                "data_quality_weight": 1.0,
            }
            for region, values in (
                ("r1", {"더불어민주당": 0.42, "진보정당계": 0.22}),
                ("r2", {"더불어민주당": 0.58, "진보정당계": 0.05}),
            )
            for bloc, share in values.items()
        ]
    )

    out = estimate_electorate_layers(
        candidates,
        history,
        date_resolver=_date_resolver,
        mass_profile="direct_party_layers",
    )

    liberal = out.loc[out["slot"].eq("A")].set_index("region_id")
    progressive = out.loc[out["slot"].eq("C")].set_index("region_id")
    assert liberal.loc["r2", "regional_accent_signal"] > liberal.loc[
        "r1", "regional_accent_signal"
    ]
    assert progressive.loc["r1", "regional_accent_signal"] > progressive.loc[
        "r2", "regional_accent_signal"
    ]
    assert out.groupby("region_id")["regional_accent_signal"].sum().to_numpy() == pytest.approx(
        [0.0, 0.0]
    )


def test_regional_accent_mobility_is_smaller_for_core_heavy_support() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "core_voting_mass": 0.45,
                "critical_voting_mass": 0.02,
                "swing_voting_mass": 0.48,
                "regional_accent_signal": 1.0,
                "regional_accent_reliability": 1.0,
                "regional_accent_volatility": 0.0,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "core_voting_mass": 0.05,
                "critical_voting_mass": 0.02,
                "swing_voting_mass": 0.48,
                "regional_accent_signal": -1.0,
                "regional_accent_reliability": 1.0,
                "regional_accent_volatility": 0.0,
            },
        ]
    )

    _, diagnostics = apply_electorate_layer_response(
        frame,
        np.array([0.50, 0.50]),
        ElectorateLayerConfig(regional_accent_gain=0.20),
    )

    shifts = diagnostics.set_index("slot")["regional_accent_log_shift"].abs()
    assert shifts["A"] < shifts["B"]


def test_camp_core_anchor_preserves_floor_before_allocating_swing() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "core_voting_mass": 0.45,
                "critical_voting_mass": 0.0,
                "swing_voting_mass": 0.50,
                "camp_core_voting_mass": 0.45,
                "camp_critical_voting_mass": 0.0,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "core_voting_mass": 0.05,
                "critical_voting_mass": 0.0,
                "swing_voting_mass": 0.50,
                "camp_core_voting_mass": 0.05,
                "camp_critical_voting_mass": 0.0,
            },
        ]
    )
    baseline = np.array([0.30, 0.70])
    predicted, diagnostics = apply_electorate_layer_response(
        frame,
        baseline,
        ElectorateLayerConfig(camp_core_anchor_gain=1.0),
    )

    assert predicted == pytest.approx([0.45, 0.55])
    assert predicted[0] >= frame.loc[0, "camp_core_voting_mass"]
    assert diagnostics["camp_anchored_pred"].to_numpy() == pytest.approx(predicted)


def test_camp_composition_removes_core_from_the_competitive_pool() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": region,
                "slot": slot,
                "core_voting_mass": core,
                "critical_voting_mass": 0.10,
                "swing_voting_mass": 0.40,
                "camp_core_voting_mass": core,
                "camp_critical_voting_mass": 0.10,
            }
            for region, masses in (
                ("r1", {"A": 0.40, "B": 0.10}),
                ("r2", {"A": 0.10, "B": 0.40}),
            )
            for slot, core in masses.items()
        ]
    )
    baseline = np.array([0.50, 0.50, 0.50, 0.50])

    predicted, diagnostics = apply_electorate_layer_response(
        frame,
        baseline,
        ElectorateLayerConfig(camp_composition_gain=1.0),
    )

    assert predicted == pytest.approx([0.65, 0.35, 0.35, 0.65])
    assert diagnostics["national_contestable_profile"].to_numpy() == pytest.approx(
        [0.50, 0.50, 0.50, 0.50]
    )


def test_presidential_warmup_is_used_when_direct_party_history_is_absent() -> None:
    candidates = pd.DataFrame(
        [{"election_id": "target", "region_id": "r1", "slot": "A", "bloc": "국민의힘"}]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "old",
                "election_type": "presidential",
                "region_id": "r1",
                "bloc": "국민의힘",
                "vote_share": 0.52,
                "data_quality_weight": 1.0,
            }
        ]
    )

    out = estimate_electorate_layers(candidates, history, date_resolver=_date_resolver).iloc[0]

    assert out["direct_party_effective_elections"] == 0.0
    assert out["candidate_ballot_effective_elections"] == pytest.approx(1.0)
    assert out["candidate_ballot_core_raw"] == pytest.approx(0.52)
    assert out["core_voting_mass"] > 0.0


def test_neutral_layer_config_is_identity() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "core_voting_mass": 0.30,
                "critical_voting_mass": 0.10,
                "swing_voting_mass": 0.40,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "core_voting_mass": 0.10,
                "critical_voting_mass": 0.10,
                "swing_voting_mass": 0.40,
            },
        ]
    )
    pred = np.array([0.55, 0.45])

    out, _ = apply_electorate_layer_response(frame, pred, ElectorateLayerConfig())

    assert out == pytest.approx(pred)


def test_swing_heavy_support_is_more_issue_sensitive_than_core_support() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "core_region",
                "slot": "A",
                "core_voting_mass": 0.45,
                "critical_voting_mass": 0.00,
                "swing_voting_mass": 0.10,
                "issue_pref_economy": 1.0,
            },
            {
                "election_id": "target",
                "region_id": "core_region",
                "slot": "B",
                "core_voting_mass": 0.45,
                "critical_voting_mass": 0.00,
                "swing_voting_mass": 0.10,
                "issue_pref_economy": -1.0,
            },
            {
                "election_id": "target",
                "region_id": "swing_region",
                "slot": "A",
                "core_voting_mass": 0.05,
                "critical_voting_mass": 0.00,
                "swing_voting_mass": 0.90,
                "issue_pref_economy": 1.0,
            },
            {
                "election_id": "target",
                "region_id": "swing_region",
                "slot": "B",
                "core_voting_mass": 0.05,
                "critical_voting_mass": 0.00,
                "swing_voting_mass": 0.90,
                "issue_pref_economy": -1.0,
            },
        ]
    )

    out, _ = apply_electorate_layer_response(
        frame,
        np.full(4, 0.50),
        ElectorateLayerConfig(preference_gain=0.30),
    )

    core_shift = out[0] - 0.50
    swing_shift = out[2] - 0.50
    assert swing_shift > core_shift > 0.0
    assert out[:2].sum() == pytest.approx(1.0)
    assert out[2:].sum() == pytest.approx(1.0)


def test_layer_separation_preserves_core_and_strengthens_critical_defection() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "core_voting_mass": 0.20,
                "critical_voting_mass": 0.25,
                "swing_voting_mass": 0.05,
                "issue_pref_regime": -1.0,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "core_voting_mass": 0.20,
                "critical_voting_mass": 0.05,
                "swing_voting_mass": 0.25,
                "issue_pref_regime": 1.0,
            },
        ]
    )
    baseline = np.array([0.50, 0.50])
    unseparated, unseparated_diag = apply_electorate_layer_response(
        frame,
        baseline,
        ElectorateLayerConfig(preference_gain=0.10, layer_separation=0.0),
    )
    separated, separated_diag = apply_electorate_layer_response(
        frame,
        baseline,
        ElectorateLayerConfig(preference_gain=0.10, layer_separation=1.0),
    )

    assert separated[0] < unseparated[0]
    a = separated_diag.loc[separated_diag["slot"].eq("A")].iloc[0]
    assert abs(a["critical_preference_log_shift"]) > abs(
        unseparated_diag.loc[unseparated_diag["slot"].eq("A"), "critical_preference_log_shift"].iloc[0]
    )
    assert abs(a["core_preference_log_shift"]) < abs(
        unseparated_diag.loc[unseparated_diag["slot"].eq("A"), "core_preference_log_shift"].iloc[0]
    )


def test_critical_defection_profile_changes_only_critical_response() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "core_voting_mass": 0.20,
                "critical_voting_mass": 0.25,
                "swing_voting_mass": 0.05,
                "issue_pref_regime": -1.0,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "core_voting_mass": 0.20,
                "critical_voting_mass": 0.05,
                "swing_voting_mass": 0.25,
                "issue_pref_regime": 1.0,
            },
        ]
    )
    baseline = np.array([0.50, 0.50])
    _, neutral_diag = apply_electorate_layer_response(
        frame,
        baseline,
        ElectorateLayerConfig(preference_gain=0.10),
    )
    _, critical_diag = apply_electorate_layer_response(
        frame,
        baseline,
        ElectorateLayerConfig(
            preference_gain=0.10,
            layer_separation=1.0,
            layer_response_profile="critical_defection",
        ),
    )

    assert critical_diag["core_preference_log_shift"].to_numpy() == pytest.approx(
        neutral_diag["core_preference_log_shift"].to_numpy()
    )
    assert critical_diag["swing_preference_log_shift"].to_numpy() == pytest.approx(
        neutral_diag["swing_preference_log_shift"].to_numpy()
    )
    assert abs(critical_diag.loc[0, "critical_preference_log_shift"]) > abs(
        neutral_diag.loc[0, "critical_preference_log_shift"]
    )
    assert critical_diag.loc[1, "critical_preference_log_shift"] == pytest.approx(
        neutral_diag.loc[1, "critical_preference_log_shift"]
    )


def test_unknown_layer_response_profile_is_rejected() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "core_voting_mass": 0.25,
                "critical_voting_mass": 0.15,
                "swing_voting_mass": 0.10,
                "issue_pref_regime": -1.0,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "core_voting_mass": 0.25,
                "critical_voting_mass": 0.15,
                "swing_voting_mass": 0.10,
                "issue_pref_regime": 1.0,
            },
        ]
    )
    with pytest.raises(ValueError, match="unknown electorate layer response profile"):
        apply_electorate_layer_response(
            frame,
            np.array([0.50, 0.50]),
            ElectorateLayerConfig(
                preference_gain=0.10,
                layer_separation=1.0,
                layer_response_profile="unknown",
            ),
        )


def test_issue_compiler_keeps_future_rows_out_and_neutral_as_attention() -> None:
    candidates = pd.DataFrame(
        [
            {"election_id": "target", "region_id": "r1", "slot": "A"},
            {"election_id": "target", "region_id": "r1", "slot": "B"},
        ]
    )
    salience = pd.DataFrame(
        [
            {
                "election_id": "target",
                "issue_name": "economy_growth",
                "salience_score": 1.0,
                "available_date": "2003-12-01",
            },
            {
                "election_id": "target",
                "issue_name": "housing",
                "salience_score": 100.0,
                "available_date": "2004-02-01",
            },
        ]
    )
    link = pd.DataFrame(
        [
            {
                "election_id": "target",
                "slot": slot,
                "issue_name": issue,
                "emphasis_within": 1.0,
                "available_date": "2003-12-15",
            }
            for slot in ("A", "B")
            for issue in ("economy_growth", "housing")
        ]
    )
    overlay = pd.DataFrame(
        [
            {
                "election_id": "target",
                "slot": "A",
                "issue_name": "economy_growth",
                "character_score": 0.0,
                "character_intensity": 0.0,
                "informational_score": 1.0,
                "issue_confidence_quality": 1.0,
            },
            {
                "election_id": "target",
                "slot": "B",
                "issue_name": "economy_growth",
                "character_score": 0.0,
                "character_intensity": 0.0,
                "informational_score": 0.5,
                "issue_confidence_quality": 1.0,
            },
            {
                "election_id": "target",
                "slot": "A",
                "issue_name": "housing",
                "character_score": 1.0,
                "character_intensity": 1.0,
                "informational_score": 0.0,
                "issue_confidence_quality": 1.0,
            },
        ]
    )

    out = compile_issue_class_signals(
        candidates,
        salience,
        link,
        overlay,
        election_dates={"target": "2004-01-01"},
    )

    assert (out["issue_pref_economy"] == 0.0).all()
    assert out.loc[out["slot"].eq("A"), "issue_attention_economy"].iloc[0] > 0.0
    assert (out["issue_pref_housing"] == 0.0).all()
    assert (out["issue_attention_housing"] == 0.0).all()


def test_issue_compiler_preserves_candidate_stance_magnitude_between_elections() -> None:
    candidates = pd.DataFrame(
        [
            {"election_id": election, "region_id": "r1", "slot": slot}
            for election in ("small", "large")
            for slot in ("A", "B")
        ]
    )
    salience = pd.DataFrame(
        [
            {
                "election_id": election,
                "issue_name": "economy_growth",
                "salience_score": 1.0,
                "available_date": "2003-12-01",
            }
            for election in ("small", "large")
        ]
    )
    link = pd.DataFrame(
        [
            {
                "election_id": election,
                "slot": slot,
                "issue_name": "economy_growth",
                "emphasis_within": 1.0,
                "available_date": "2003-12-15",
            }
            for election in ("small", "large")
            for slot in ("A", "B")
        ]
    )
    overlay = pd.DataFrame(
        [
            {
                "election_id": election,
                "slot": slot,
                "issue_name": "economy_growth",
                "character_intensity": 1.0,
                "informational_score": 0.0,
                "issue_confidence_quality": 1.0,
            }
            for election in ("small", "large")
            for slot in ("A", "B")
        ]
    )
    stance = pd.DataFrame(
        [
            {
                "election_id": election,
                "slot": slot,
                "party_stance_signal_centered": direction,
                "confidence": 1.0,
                "available_date": "2003-12-20",
            }
            for election, magnitude in (("small", 0.10), ("large", 0.80))
            for slot, direction in (("A", magnitude), ("B", -magnitude))
        ]
    )

    out = compile_issue_class_signals(
        candidates,
        salience,
        link,
        overlay,
        candidate_stance=stance,
        election_dates={"small": "2004-01-01", "large": "2004-01-01"},
    )
    small = out.loc[out["election_id"].eq("small") & out["slot"].eq("A")].iloc[0]
    large = out.loc[out["election_id"].eq("large") & out["slot"].eq("A")].iloc[0]

    assert small["issue_pref_economy"] == pytest.approx(0.10)
    assert large["issue_pref_economy"] == pytest.approx(0.80)
    assert large["issue_preference_strength"] > small["issue_preference_strength"]


def test_active_engine_postprocess_can_apply_and_disable_electorate_layer(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "core_voting_mass": 0.05,
                "critical_voting_mass": 0.05,
                "swing_voting_mass": 0.80,
                "issue_pref_economy": 1.0,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "core_voting_mass": 0.05,
                "critical_voting_mass": 0.05,
                "swing_voting_mass": 0.80,
                "issue_pref_economy": -1.0,
            },
        ]
    )
    monkeypatch.setattr(issue_vote_engine, "ELECTORATE_LAYER_ENABLED", True)
    monkeypatch.setattr(
        issue_vote_engine,
        "ELECTORATE_LAYER_CONFIG",
        ElectorateLayerConfig(preference_gain=0.10),
    )
    baseline = np.array([0.50, 0.50])

    disabled = issue_vote_engine.apply_prediction_postprocess(
        frame,
        baseline,
        partisan_layer=False,
        party_tone=False,
        electorate_layer=False,
    )
    active = issue_vote_engine.apply_prediction_postprocess(
        frame,
        baseline,
        partisan_layer=False,
        party_tone=False,
    )

    assert disabled == pytest.approx(baseline)
    assert active[0] > disabled[0]
    assert active.sum() == pytest.approx(1.0)


def test_party_context_changes_camp_retention_not_total_support_directly() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "A",
                "party_context_support": 0.80,
                "party_context_confidence": 1.0,
                "party_elite_fragmentation_score": 0.05,
                "core_voting_mass": 0.30,
                "critical_voting_mass": 0.10,
            },
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": "B",
                "party_context_support": -0.40,
                "party_context_confidence": 1.0,
                "party_elite_fragmentation_score": 0.80,
                "core_voting_mass": 0.30,
                "critical_voting_mass": 0.10,
            },
        ]
    )
    baseline = np.array([0.50, 0.50])

    adjusted = issue_vote_engine.apply_party_context_prediction_adjustment(
        frame, baseline
    )

    assert adjusted[0] > baseline[0]
    assert adjusted[1] < baseline[1]
    assert adjusted.sum() == pytest.approx(1.0)


def test_party_context_zero_confidence_is_exact_identity() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": slot,
                "party_context_support": support,
                "party_context_confidence": 0.0,
                "party_elite_fragmentation_score": fragmentation,
                "core_voting_mass": 0.25,
                "critical_voting_mass": 0.12,
            }
            for slot, support, fragmentation in (
                ("A", -1.0, 1.0),
                ("B", 1.0, 0.0),
            )
        ]
    )
    baseline = np.array([0.55, 0.45])

    adjusted = issue_vote_engine.apply_party_context_prediction_adjustment(
        frame, baseline
    )

    assert adjusted == pytest.approx(baseline)


def test_party_context_core_is_less_elastic_than_critical_support() -> None:
    common = {
        "election_id": "target",
        "region_id": "r1",
        "party_context_support": -1.0,
        "party_context_confidence": 1.0,
        "party_elite_fragmentation_score": 1.0,
    }
    core_frame = pd.DataFrame(
        [
            {**common, "slot": "A", "core_voting_mass": 0.20, "critical_voting_mass": 0.0},
            {**common, "slot": "B", "party_context_confidence": 0.0, "core_voting_mass": 0.0, "critical_voting_mass": 0.0},
        ]
    )
    critical_frame = core_frame.copy()
    critical_frame.loc[0, "core_voting_mass"] = 0.0
    critical_frame.loc[0, "critical_voting_mass"] = 0.20
    baseline = np.array([0.50, 0.50])

    core_adjusted = issue_vote_engine.apply_party_context_prediction_adjustment(
        core_frame, baseline
    )
    critical_adjusted = issue_vote_engine.apply_party_context_prediction_adjustment(
        critical_frame, baseline
    )

    assert abs(critical_adjusted[0] - baseline[0]) > abs(
        core_adjusted[0] - baseline[0]
    )


def test_candidate_conversion_ignores_party_cohesion_direct_inputs(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": "r1",
                "slot": slot,
                "serious_contender_score": 0.60,
                "legitimacy_score": 0.60,
                "organization_strength": 0.60,
                "alternative_score": 0.60,
                "conversion_capacity_centered": conversion,
                "coalition_mobilization_centered": conversion,
                "wasted_vote_resistance": 1.0,
                "major_party_gravity": 0.0,
                "third_candidate_overexposure_risk": 0.0,
                "candidate_conversion_confidence": 1.0,
            }
            for slot, conversion in (("A", 1.0), ("B", -1.0))
        ]
    )
    monkeypatch.setattr(
        issue_vote_engine,
        "_rederived_float",
        lambda key, default=0.0: 0.05 if key == "conversion_scale" else default,
    )
    baseline = np.array([0.50, 0.50])

    adjusted = issue_vote_engine.apply_candidate_conversion_context_adjustment(
        frame, baseline
    )

    assert adjusted == pytest.approx(baseline)


def test_vectorized_draw_response_matches_single_draw_path() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "target",
                "region_id": region,
                "slot": slot,
                "core_voting_mass": core,
                "critical_voting_mass": 0.10,
                "swing_voting_mass": 0.50,
                "issue_pref_regime": signal,
                "issue_attention_regime": abs(signal),
            }
            for region in ("r1", "r2")
            for slot, core, signal in (("A", 0.25, 0.8), ("B", 0.15, -0.8))
        ]
    )
    draws = np.array(
        [
            [0.55, 0.45, 0.40, 0.60],
            [0.50, 0.50, 0.65, 0.35],
            [0.60, 0.40, 0.45, 0.55],
        ]
    )
    config = ElectorateLayerConfig(
        terrain_anchor_gain=0.10,
        camp_composition_gain=0.20,
        preference_gain=0.04,
        layer_separation=0.75,
        turnout_gain=0.02,
    )

    batch = apply_electorate_layer_response_draws(frame, draws, config)
    singles = np.vstack(
        [apply_electorate_layer_response(frame, draw, config)[0] for draw in draws]
    )

    assert batch == pytest.approx(singles, abs=1e-12)
