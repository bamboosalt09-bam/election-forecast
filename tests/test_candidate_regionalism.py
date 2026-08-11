from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine


def test_candidate_regional_base_excludes_future_rows(tmp_path, monkeypatch) -> None:
    path = tmp_path / "candidate_regional_base.csv"
    path.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,region_id,regional_affinity,organization_depth,available_date,confidence,source_type,notes",
                "pres_2017,C,Third,r1,0.8,0.5,2017-05-08,0.75,test,available",
                "pres_2017,C,Third,r2,1.0,1.0,2017-05-10,1.0,test,future",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_REGIONAL_BASE", str(path))

    out = issue_vote_engine._load_candidate_regional_base()

    assert out[["region_id", "candidate_regional_base_raw"]].to_dict("records") == [
        {"region_id": "r1", "candidate_regional_base_raw": pytest.approx(0.8 * 0.5 * 0.75)}
    ]


def test_candidate_regionalism_signal_is_candidate_and_region_centered() -> None:
    frame = pd.DataFrame(
        [
            {"election_id": "pres_x", "region_id": region, "slot": slot}
            for region in ["r1", "r2"]
            for slot in ["A", "B", "C"]
        ]
    )
    frame["candidate_regional_base_raw"] = 0.0
    frame.loc[(frame["region_id"] == "r1") & (frame["slot"] == "C"), "candidate_regional_base_raw"] = 0.8
    frame["third_competitiveness_gate"] = np.where(frame["slot"].eq("C"), 0.5, 0.0)

    out = issue_vote_engine._finalize_candidate_regionalism_features(frame)

    assert out.groupby(["election_id", "slot"])["candidate_regionalism_signal"].sum().tolist() == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert out.groupby(["election_id", "region_id"])["candidate_regionalism_signal"].sum().tolist() == pytest.approx(
        [0.0, 0.0]
    )
    c_signal = out.loc[out["slot"].eq("C")].sort_values("region_id")[
        "candidate_regionalism_signal"
    ]
    assert c_signal.tolist() == pytest.approx([0.8 * 0.5 / 3.0, -0.8 * 0.5 / 3.0])


def test_candidate_regional_base_is_not_erased_by_national_third_gate() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "region_id": region,
                "slot": slot,
                "candidate_regional_base_raw": 0.9 if region == "r1" and slot == "C" else 0.0,
                "third_competitiveness_gate": 0.01 if slot == "C" else 0.0,
                "third_regime_bloc_split_score": 0.36 if slot == "C" else 0.0,
            }
            for region in ["r1", "r2"]
            for slot in ["A", "B", "C"]
        ]
    )

    out = issue_vote_engine._finalize_candidate_regionalism_features(frame)

    assert out.loc[
        out["region_id"].eq("r1") & out["slot"].eq("C"),
        "candidate_regionalism_signal",
    ].iloc[0] == pytest.approx(0.9 * 0.6 / 3.0)


def test_third_competitiveness_gate_distinguishes_conversion_capacity() -> None:
    rows = []
    for election_id, strength in [("weak", 0.1), ("strong", 0.8)]:
        for slot in ["A", "B", "C"]:
            rows.append(
                {
                    "election_id": election_id,
                    "region_id": "r1",
                    "slot": slot,
                    "third_viability": 0.8 if slot == "C" else 0.0,
                    "third_profile_confidence": 0.8 if slot == "C" else 0.0,
                    "candidate_weight": strength if slot == "C" else 0.5,
                    "wasted_vote_resistance": strength if slot == "C" else 0.5,
                    "coalition_mobilization_score": strength if slot == "C" else 0.5,
                    "conversion_capacity": strength if slot == "C" else 0.5,
                }
            )
    out = issue_vote_engine._third_candidate_competitiveness_features(pd.DataFrame(rows))
    c_rows = out.loc[out["slot"].eq("C")].set_index("election_id")

    assert c_rows.loc["strong", "third_competitiveness_gate"] > c_rows.loc[
        "weak", "third_competitiveness_gate"
    ]
    assert c_rows.loc["strong", "third_competitiveness_multiplier"] > c_rows.loc[
        "weak", "third_competitiveness_multiplier"
    ]


def test_third_regime_character_scores_are_compositional() -> None:
    frame = issue_vote_engine.assemble()
    character_score_columns = [
        "third_regime_niche_minor_score",
        "third_regime_reform_minor_score",
        "third_regime_bloc_split_score",
        "third_regime_independent_pole_score",
    ]
    has_active_third = frame["third_regime_character"].ne("two_way_withdrawn_or_absent")
    assert frame.loc[has_active_third, character_score_columns].sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert frame.loc[~has_active_third, character_score_columns].sum(axis=1).to_numpy() == pytest.approx(0.0)
    assert (
        frame["third_regime_two_way_score"]
        + frame["third_regime_competitiveness"]
    ).to_numpy() == pytest.approx(1.0)

    character = frame.groupby("election_id")["third_regime_character"].first()
    assert character["pres_2002"] == "niche_minor"
    assert character["pres_2007"] == "bloc_split"
    assert character["pres_2012"] == "two_way_withdrawn_or_absent"
    assert character["pres_2017"] == "independent_pole"
    assert character["pres_2022"] == "two_way_withdrawn_or_absent"
    split_rows = frame.loc[
        frame["election_id"].eq("pres_2007") & frame["slot"].eq("C")
    ]
    assert split_rows["within_bloc_reservoir_confirmation"].max() > 0.0
    strong_personal_base = split_rows["candidate_regionalism_signal"].eq(
        split_rows["candidate_regionalism_signal"].max()
    )
    assert (
        split_rows.loc[strong_personal_base, "within_bloc_transfer_profile"]
        > split_rows.loc[strong_personal_base, "within_bloc_transfer_base_profile"]
    ).all()


def test_low_organization_outsider_gets_stronger_centered_regional_anchor() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "region_id": region,
                "slot": slot,
                "candidate_regional_base_raw": 0.6 if region == "r1" and slot == "C" else 0.0,
                "third_competitiveness_gate": 0.5 if slot == "C" else 0.0,
                "outsider_status": 1.0 if slot == "C" else 0.0,
                "organization_strength": 0.0 if slot == "C" else 1.0,
            }
            for region in ["r1", "r2"]
            for slot in ["A", "B", "C"]
        ]
    )
    baseline = frame.copy()
    baseline["outsider_status"] = 0.0

    anchored = issue_vote_engine._finalize_candidate_regionalism_features(frame)
    unanchored = issue_vote_engine._finalize_candidate_regionalism_features(baseline)

    anchored_c = anchored.loc[(anchored["region_id"] == "r1") & anchored["slot"].eq("C")]
    unanchored_c = unanchored.loc[(unanchored["region_id"] == "r1") & unanchored["slot"].eq("C")]
    assert anchored_c["candidate_regional_anchor_multiplier"].iloc[0] == pytest.approx(2.0)
    assert anchored_c["candidate_regionalism_signal"].iloc[0] > unanchored_c[
        "candidate_regionalism_signal"
    ].iloc[0]
    assert anchored.groupby(["election_id", "region_id"])["candidate_regionalism_signal"].sum().tolist() == pytest.approx(
        [0.0, 0.0]
    )


def test_candidate_regionalism_adjustment_preserves_region_composition(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {"election_id": "pres_x", "region_id": "r1", "slot": "A", "candidate_regionalism_signal": 0.2},
            {"election_id": "pres_x", "region_id": "r1", "slot": "B", "candidate_regionalism_signal": -0.2},
        ]
    )
    monkeypatch.setattr(
        issue_vote_engine,
        "THROUGH_2022_REDERIVED_LAYER_CONFIG",
        {
            "regionalism_scale": 0.1,
            "regional_anchor_strength": 1.0,
            "third_competitiveness_gate_enabled": False,
            "third_character_multiplier_enabled": False,
        },
    )

    adjusted = issue_vote_engine.apply_candidate_regionalism_adjustment(frame, [0.5, 0.5])

    assert adjusted.tolist() == pytest.approx([0.52, 0.48])
    assert adjusted.sum() == pytest.approx(1.0)


def test_district_terrain_signal_and_adjustment_are_region_zero_sum(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "region_id": region,
                "slot": slot,
                "landscape_inferred_district_prior": (
                    0.12 if region == "r1" and slot == "A" else -0.08
                ),
                "landscape_inferred_district_evidence": 4.0,
            }
            for region in ["r1", "r2"]
            for slot in ["A", "B"]
        ]
    )
    terrain = issue_vote_engine._finalize_district_terrain_features(frame)
    assert terrain.groupby(["election_id", "slot"])["district_terrain_signal"].sum().tolist() == pytest.approx(
        [0.0, 0.0]
    )
    assert terrain.groupby(["election_id", "region_id"])["district_terrain_signal"].sum().tolist() == pytest.approx(
        [0.0, 0.0]
    )

    monkeypatch.setattr(
        issue_vote_engine,
        "THROUGH_2022_REDERIVED_LAYER_CONFIG",
        {"district_terrain_scale": 0.2},
    )
    adjusted = issue_vote_engine.apply_district_terrain_adjustment(
        terrain,
        [0.5, 0.5, 0.5, 0.5],
    )
    sums = terrain.assign(pred=adjusted).groupby(["election_id", "region_id"])["pred"].sum()
    assert sums.tolist() == pytest.approx([1.0, 1.0])


def _within_bloc_transfer_fixture(independent_pole: float = 0.1) -> pd.DataFrame:
    rows = []
    for region in ["r1", "r2"]:
        for slot, bloc in [("A", "국민의힘"), ("B", "더불어민주당"), ("C", "무소속")]:
            rows.append(
                {
                    "election_id": "pres_x",
                    "region_id": region,
                    "slot": slot,
                    "candidate_name": slot,
                    "bloc": bloc,
                    "third_regime_competitiveness": 0.6,
                    "third_regime_bloc_split_score": 0.5,
                    "third_regime_independent_pole_score": independent_pole,
                    "candidate_regionalism_signal": (
                        0.4 if slot == "C" and region == "r1" else -0.4 if slot == "C" else 0.0
                    ),
                    "district_terrain_signal": (
                        0.2 if slot == "C" and region == "r1" else -0.2 if slot == "C" else 0.0
                    ),
                    "district_terrain_reliability": 1.0,
                    "candidate_regional_base_gated": 0.0,
                    "partisan_prior": (
                        0.6
                        if slot == "A" and region == "r1"
                        else -0.6 if slot == "A" else 0.0
                    ),
                    "effective_election_count": 5.0,
                    "landscape_axis_conservative": 0.8 if slot in {"A", "C"} else 0.0,
                    "landscape_axis_liberal": 0.8 if slot == "B" else 0.0,
                    "landscape_axis_progressive": 0.0,
                    "landscape_axis_centrist": 0.0,
                    "landscape_axis_anti_establishment": 0.0,
                    "landscape_axis_reform": 0.0,
                    "landscape_axis_regionalist": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_within_bloc_transfer_is_candidate_and_region_zero_sum() -> None:
    out = issue_vote_engine._finalize_within_bloc_regional_transfer_features(
        _within_bloc_transfer_fixture()
    )

    assert out.groupby(["election_id", "region_id"])[
        "within_bloc_regional_transfer_signal"
    ].sum().tolist() == pytest.approx([0.0, 0.0])
    assert out.groupby(["election_id", "slot"])[
        "within_bloc_regional_transfer_signal"
    ].sum().tolist() == pytest.approx([0.0, 0.0, 0.0])
    signal = out.pivot(index="region_id", columns="slot", values="within_bloc_regional_transfer_signal")
    assert signal.loc["r1", "C"] > 0.0
    assert signal.loc["r1", "A"] == pytest.approx(-signal.loc["r1", "C"])
    assert signal.loc["r1", "B"] == pytest.approx(0.0)
    reservoir = out.loc[out["region_id"].eq("r1"), "within_bloc_same_lane_reservoir"]
    assert (reservoir > 0.0).all()


def test_within_bloc_transfer_skips_independent_pole() -> None:
    out = issue_vote_engine._finalize_within_bloc_regional_transfer_features(
        _within_bloc_transfer_fixture(independent_pole=0.8)
    )

    assert out["within_bloc_regional_transfer_signal"].tolist() == pytest.approx([0.0] * len(out))


def test_within_bloc_transfer_adjustment_preserves_region_composition(monkeypatch) -> None:
    frame = issue_vote_engine._finalize_within_bloc_regional_transfer_features(
        _within_bloc_transfer_fixture()
    )
    monkeypatch.setattr(
        issue_vote_engine,
        "THROUGH_2022_REDERIVED_LAYER_CONFIG",
        {"within_bloc_transfer_scale": 0.5},
    )
    baseline = np.tile([0.4, 0.35, 0.25], 2)

    adjusted = issue_vote_engine.apply_within_bloc_regional_transfer_adjustment(
        frame,
        baseline,
    )

    sums = frame.assign(pred=adjusted).groupby(["election_id", "region_id"])["pred"].sum()
    assert sums.tolist() == pytest.approx([1.0, 1.0])
    r1 = frame.assign(pred=adjusted).loc[frame["region_id"].eq("r1")].set_index("slot")
    assert r1.loc["C", "pred"] > 0.25
    assert r1.loc["A", "pred"] < 0.4
    assert r1.loc["B", "pred"] == pytest.approx(0.35)


def test_within_bloc_stronghold_gain_reinforces_existing_personal_base(
    monkeypatch,
) -> None:
    source = _within_bloc_transfer_fixture()
    source.loc[
        source["slot"].eq("C") & source["region_id"].eq("r1"),
        "candidate_regional_base_gated",
    ] = 0.8
    source.loc[
        source["slot"].eq("C") & source["region_id"].eq("r2"),
        "candidate_regional_base_gated",
    ] = 0.1
    frame = issue_vote_engine._finalize_within_bloc_regional_transfer_features(source)
    baseline = np.tile([0.4, 0.35, 0.25], 2)

    monkeypatch.setattr(
        issue_vote_engine,
        "THROUGH_2022_REDERIVED_LAYER_CONFIG",
        {
            "within_bloc_transfer_scale": 0.5,
            "within_bloc_reservoir_gain": 1.0,
            "within_bloc_stronghold_gain": 0.0,
        },
    )
    without_reinforcement = (
        issue_vote_engine.apply_within_bloc_regional_transfer_adjustment(
            frame, baseline
        )
    )
    issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG[
        "within_bloc_stronghold_gain"
    ] = 0.25
    with_reinforcement = issue_vote_engine.apply_within_bloc_regional_transfer_adjustment(
        frame, baseline
    )

    r1_c = frame.index[frame["region_id"].eq("r1") & frame["slot"].eq("C")][0]
    assert with_reinforcement[r1_c] > without_reinforcement[r1_c]
    sums = frame.assign(pred=with_reinforcement).groupby(
        ["election_id", "region_id"]
    )["pred"].sum()
    assert sums.tolist() == pytest.approx([1.0, 1.0])


def test_future_holdout_skips_residual_calibration_with_five_prior_elections(monkeypatch) -> None:
    train_rows = []
    for election_id in ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"]:
        train_rows.extend(
            [
                {"election_id": election_id, "region_id": "r1", "slot": "A", "vote_share": 0.7},
                {"election_id": election_id, "region_id": "r1", "slot": "B", "vote_share": 0.3},
            ]
        )
    train = pd.DataFrame(train_rows)
    test = pd.DataFrame(
        [
            {"election_id": "pres_future", "region_id": "r1", "slot": "A", "vote_share": 0.5},
            {"election_id": "pres_future", "region_id": "r1", "slot": "B", "vote_share": 0.5},
        ]
    )
    test_pred = np.array([0.5, 0.5])

    adjusted = issue_vote_engine.apply_region_residual_calibration(
        train,
        test,
        train_pred=np.tile([0.6, 0.4], 5),
        test_pred=test_pred,
    )

    assert adjusted.tolist() == pytest.approx(test_pred.tolist())
