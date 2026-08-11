from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine


def test_third_candidate_structure_expands_profile_and_pressure(tmp_path, monkeypatch) -> None:
    profile = tmp_path / "third_candidate_profile.csv"
    profile.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,viability,centrist_appeal,anti_major_party_appeal,regional_base_overlap,available_date,confidence,notes",
                "pres_2017,C,Third,0.8,0.5,0.25,0.2,2017-04-01,0.75,test",
            ]
        ),
        encoding="utf-8",
    )
    pressure = tmp_path / "third_candidate_pressure.csv"
    pressure.write_text(
        "\n".join(
            [
                "election_id,slot,source_slot,transfer_pressure,available_date,confidence,notes",
                "pres_2017,C,A,0.4,2017-04-01,0.6,test",
                "pres_2017,C,B,0.1,2017-04-01,0.5,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "THIRD_CANDIDATE_PROFILE", str(profile))
    monkeypatch.setattr(issue_vote_engine, "THIRD_CANDIDATE_PRESSURE", str(pressure))
    base = pd.DataFrame(
        [
            {"election_id": "pres_2017", "region_id": "r1", "slot": "A"},
            {"election_id": "pres_2017", "region_id": "r1", "slot": "B"},
            {"election_id": "pres_2017", "region_id": "r1", "slot": "C"},
        ]
    )

    out = issue_vote_engine._third_candidate_structure_features(base)
    values = dict(zip(out["slot"], out["third_candidate_structure"]))

    assert values["A"] == pytest.approx(-0.4 * 0.6 * 0.8)
    assert values["B"] == pytest.approx(-0.1 * 0.5 * 0.8)
    attention = 0.8 * (0.55 + 0.25 * 0.5 + 0.20 * 0.25) * 0.75
    conversion_capacity = 0.20 + 0.35 * 0.8 + 0.25 * 0.2 + 0.20 * 0.5
    assert values["C"] == pytest.approx(attention * conversion_capacity)
    assert out.loc[out["slot"] == "C", "third_attention_score"].iloc[0] == pytest.approx(attention)
    assert out.loc[out["slot"] == "C", "third_conversion_capacity"].iloc[0] == pytest.approx(
        conversion_capacity
    )
    assert out.loc[out["slot"] == "C", "third_attention_overhang"].iloc[0] == pytest.approx(
        attention * (1.0 - conversion_capacity)
    )
    assert out.loc[out["slot"] == "A", "slotA_third_pressure"].iloc[0] == pytest.approx(0.192)
    assert out.loc[out["slot"] == "B", "slotB_third_pressure"].iloc[0] == pytest.approx(0.04)


def test_third_candidate_structure_excludes_future_rows(tmp_path, monkeypatch) -> None:
    profile = tmp_path / "third_candidate_profile.csv"
    profile.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,viability,centrist_appeal,anti_major_party_appeal,regional_base_overlap,available_date,confidence,notes",
                "pres_2017,C,Third,0.8,0.5,0.25,0.2,2018-01-01,0.75,future",
            ]
        ),
        encoding="utf-8",
    )
    pressure = tmp_path / "third_candidate_pressure.csv"
    pressure.write_text(
        "\n".join(
            [
                "election_id,slot,source_slot,transfer_pressure,available_date,confidence,notes",
                "pres_2017,C,B,0.5,2018-01-01,0.5,future",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "THIRD_CANDIDATE_PROFILE", str(profile))
    monkeypatch.setattr(issue_vote_engine, "THIRD_CANDIDATE_PRESSURE", str(pressure))
    base = pd.DataFrame([{"election_id": "pres_2017", "region_id": "r1", "slot": "C"}])

    out = issue_vote_engine._third_candidate_structure_features(base)

    assert out["third_candidate_structure"].iloc[0] == 0.0


def test_third_candidate_prediction_adjustment_is_bounded_and_optional(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {"third_candidate_structure": 0.5},
            {"third_candidate_structure": -0.2},
        ]
    )

    monkeypatch.setenv("POLL_PROJECT_THIRD_CANDIDATE_ADJUSTMENT_SCALE", "0.2")
    adjusted = issue_vote_engine.apply_third_candidate_prediction_adjustment(frame, [0.3, 0.3])

    assert adjusted.tolist() == pytest.approx([0.4, 0.26])

    monkeypatch.setenv("POLL_PROJECT_DISABLE_THIRD_CANDIDATE_ADJUSTMENT", "1")
    disabled = issue_vote_engine.apply_third_candidate_prediction_adjustment(frame, [0.3, 0.3])

    assert disabled.tolist() == pytest.approx([0.3, 0.3])


def test_withdrawn_candidate_transfer_targets_active_slot(tmp_path, monkeypatch) -> None:
    transfers = tmp_path / "withdrawn_candidate_transfers.csv"
    transfers.write_text(
        "\n".join(
            [
                "election_id,candidate_name,target_slot,viability,transfer_rate,voter_compliance,available_date,confidence,notes",
                "pres_2022,Ahn Cheol-soo,A,0.5,0.8,0.5,2022-03-03,0.75,test",
                "pres_2022,Future,B,1.0,1.0,1.0,2023-01-01,1.0,future",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "WITHDRAWN_CANDIDATE_TRANSFERS", str(transfers))
    base = pd.DataFrame(
        [
            {"election_id": "pres_2022", "region_id": "r1", "slot": "A"},
            {"election_id": "pres_2022", "region_id": "r1", "slot": "B"},
        ]
    )

    out = issue_vote_engine._withdrawn_candidate_transfer_features(base)

    assert out.loc[out["slot"] == "A", "withdrawn_candidate_transfer"].iloc[0] == pytest.approx(
        0.5 * 0.8 * 0.5 * 0.75
    )
    assert out.loc[out["slot"] == "B", "withdrawn_candidate_transfer"].iloc[0] == 0.0


def test_withdrawn_third_candidate_pressure_does_not_remain_on_final_slots(
    tmp_path,
    monkeypatch,
) -> None:
    profile = tmp_path / "third_candidate_profile.csv"
    profile.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,viability,centrist_appeal,anti_major_party_appeal,regional_base_overlap,available_date,confidence,notes",
                "pres_2022,C,Ahn Cheol-soo,0.5,0.8,0.7,0.3,2022-02-01,0.7,test",
            ]
        ),
        encoding="utf-8",
    )
    pressure = tmp_path / "third_candidate_pressure.csv"
    pressure.write_text(
        "\n".join(
            [
                "election_id,slot,source_slot,transfer_pressure,available_date,confidence,notes",
                "pres_2022,C,A,0.4,2022-02-01,0.6,test",
                "pres_2022,C,B,0.6,2022-02-01,0.6,test",
            ]
        ),
        encoding="utf-8",
    )
    events = tmp_path / "coalition_events.csv"
    events.write_text(
        "\n".join(
            [
                "event_id,election_id,event_date,available_date,event_type,source_slot,target_slot,transfer_rate,voter_compliance,source_viability_after_event,exclude_source_from_evaluation,notes",
                "pres_2022_c_to_a,pres_2022,2022-03-03,2022-03-03,coalition_withdrawal,C,A,0.9,0.8,0.0,true,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "THIRD_CANDIDATE_PROFILE", str(profile))
    monkeypatch.setattr(issue_vote_engine, "THIRD_CANDIDATE_PRESSURE", str(pressure))
    monkeypatch.setattr(issue_vote_engine, "COALITION_EVENTS", str(events))
    base = pd.DataFrame(
        [
            {"election_id": "pres_2022", "region_id": "r1", "slot": "A"},
            {"election_id": "pres_2022", "region_id": "r1", "slot": "B"},
        ]
    )

    out = issue_vote_engine._third_candidate_structure_features(base)

    assert out["third_candidate_structure"].tolist() == pytest.approx([0.0, 0.0])
    assert out["slotA_third_pressure"].sum() == 0.0
    assert out["slotB_third_pressure"].sum() == 0.0


def test_withdrawn_candidate_prediction_adjustment_uses_conservative_default(monkeypatch) -> None:
    frame = pd.DataFrame([{"withdrawn_candidate_transfer": 0.4}])

    monkeypatch.delenv("POLL_PROJECT_WITHDRAWN_CANDIDATE_ADJUSTMENT_SCALE", raising=False)
    adjusted = issue_vote_engine.apply_withdrawn_candidate_prediction_adjustment(frame, [0.3])

    assert adjusted.tolist() == pytest.approx([0.36])

    monkeypatch.setenv("POLL_PROJECT_WITHDRAWN_CANDIDATE_ADJUSTMENT_SCALE", "0.25")
    adjusted = issue_vote_engine.apply_withdrawn_candidate_prediction_adjustment(frame, [0.3])

    assert adjusted.tolist() == pytest.approx([0.4])


def test_withdrawn_transfer_uses_candidate_political_landscape(tmp_path, monkeypatch) -> None:
    transfers = tmp_path / "withdrawn_candidate_transfers.csv"
    transfers.write_text(
        "\n".join(
            [
                "election_id,candidate_name,target_slot,viability,transfer_rate,voter_compliance,available_date,confidence,notes",
                "pres_2022,Third,A,1.0,0.5,1.0,2022-03-03,1.0,test",
                "pres_2022,Third,B,1.0,0.5,1.0,2022-03-03,1.0,test",
            ]
        ),
        encoding="utf-8",
    )
    landscape = tmp_path / "candidate_political_landscape.csv"
    landscape.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,candidate_role,conservative,liberal,progressive,centrist,anti_establishment,reform,regionalist,available_date,confidence,notes",
                "pres_2022,C,Third,withdrawn,0.0,1.0,0.0,0.0,0.0,0.0,0.0,2022-03-01,1.0,test",
                "pres_2022,A,Target A,final,1.0,0.0,0.0,0.0,0.0,0.0,0.0,2022-03-01,1.0,test",
                "pres_2022,B,Target B,final,0.0,1.0,0.0,0.0,0.0,0.0,0.0,2022-03-01,1.0,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "WITHDRAWN_CANDIDATE_TRANSFERS", str(transfers))
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_POLITICAL_LANDSCAPE", str(landscape))

    out = issue_vote_engine._load_withdrawn_candidate_transfers()
    values = dict(zip(out["slot"], out["withdrawn_candidate_transfer"]))

    assert values["B"] > values["A"]
    assert sum(values.values()) == pytest.approx(1.0)


def test_candidate_political_landscape_features_cover_final_three_slots(tmp_path, monkeypatch) -> None:
    landscape = tmp_path / "candidate_political_landscape.csv"
    landscape.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,candidate_role,conservative,liberal,progressive,centrist,anti_establishment,reform,regionalist,available_date,confidence,notes",
                "pres_2017,A,A,final,0.2,0.8,0.6,0.2,0.2,0.5,0.0,2017-04-01,1.0,test",
                "pres_2017,B,B,final,0.9,0.1,0.0,0.2,0.1,0.1,0.1,2017-04-01,1.0,test",
                "pres_2017,C,C,final,0.3,0.3,0.1,0.9,0.8,0.8,0.0,2017-04-01,1.0,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_POLITICAL_LANDSCAPE", str(landscape))
    base = pd.DataFrame(
        [
            {"election_id": "pres_2017", "region_id": "r1", "slot": "A", "bloc": "더불어민주당"},
            {"election_id": "pres_2017", "region_id": "r1", "slot": "B", "bloc": "국민의힘"},
            {"election_id": "pres_2017", "region_id": "r1", "slot": "C", "bloc": "제3지대"},
        ]
    )

    out = issue_vote_engine._candidate_political_landscape_features(base)
    c_row = out.loc[out["slot"] == "C"].iloc[0]

    assert "landscape_bloc_alignment" in out.columns
    assert "landscape_bloc_alignment" in issue_vote_engine.PREDICTORS
    assert c_row["landscape_centrist"] > 0
    assert c_row["landscape_reform_anti_establishment"] > 0
    assert out[["election_id", "region_id", "slot"]].to_dict("records") == base[
        ["election_id", "region_id", "slot"]
    ].to_dict("records")


def test_candidate_political_landscape_blends_assembly15_legacy(
    tmp_path,
    monkeypatch,
) -> None:
    landscape = tmp_path / "candidate_political_landscape.csv"
    landscape.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,candidate_role,conservative,liberal,progressive,centrist,anti_establishment,reform,regionalist,available_date,confidence,notes",
                "pres_2007,C,Legacy Candidate,final,0.2,0.6,0.5,0.2,0.1,0.2,0.0,2007-10-01,1.0,test",
            ]
        ),
        encoding="utf-8",
    )
    legacy = tmp_path / "assembly15_candidate_legacy_landscape.csv"
    legacy.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,conservative,liberal,progressive,centrist,anti_establishment,reform,regionalist,matched_rows,issue_count,available_date,confidence,source,notes",
                "pres_2007,C,Legacy Candidate,1.0,0.0,0.0,0.3,0.2,0.2,0.0,40,10,2000-01-15,0.5,test,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_POLITICAL_LANDSCAPE", str(landscape))
    monkeypatch.setattr(issue_vote_engine, "ASSEMBLY15_CANDIDATE_LEGACY_LANDSCAPE", str(legacy))
    monkeypatch.setenv("POLL_PROJECT_ASSEMBLY15_LEGACY_BLEND_SCALE", "1.0")
    base = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "region_id": "r1",
                "slot": "C",
                "candidate_name": "Legacy Candidate",
                "bloc": "무소속",
            }
        ]
    )

    out = issue_vote_engine._candidate_political_landscape_features(base)

    assert out["landscape_legacy_confidence"].iloc[0] == pytest.approx(0.5)
    assert out["landscape_legacy_blend"].iloc[0] == pytest.approx(0.5)
    assert out["landscape_axis_conservative"].iloc[0] > 0.2
    assert out["landscape_left_right"].iloc[0] > -0.35


def test_candidate_landscape_inferred_prior_uses_political_vector_not_formal_bloc() -> None:
    base = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "region_id": "r_conservative",
                "slot": "C",
                "landscape_axis_conservative": 1.0,
                "landscape_axis_liberal": 0.0,
                "landscape_axis_progressive": 0.0,
                "landscape_axis_centrist": 0.0,
                "landscape_axis_anti_establishment": 0.0,
                "landscape_axis_reform": 0.0,
                "landscape_axis_regionalist": 0.0,
            },
            {
                "election_id": "pres_2007",
                "region_id": "r_liberal",
                "slot": "C",
                "landscape_axis_conservative": 1.0,
                "landscape_axis_liberal": 0.0,
                "landscape_axis_progressive": 0.0,
                "landscape_axis_centrist": 0.0,
                "landscape_axis_anti_establishment": 0.0,
                "landscape_axis_reform": 0.0,
                "landscape_axis_regionalist": 0.0,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "election_type": "presidential",
                "region_id": "r_conservative",
                "bloc": "국민의힘",
                "vote_share": 0.80,
                "data_quality_weight": 1.0,
                "baseline_share": pd.NA,
            },
            {
                "election_id": "pres_2002",
                "election_type": "presidential",
                "region_id": "r_liberal",
                "bloc": "국민의힘",
                "vote_share": 0.20,
                "data_quality_weight": 1.0,
                "baseline_share": pd.NA,
            },
        ]
    )

    out = issue_vote_engine._candidate_landscape_inferred_prior_features(base, history)
    values = dict(zip(out["region_id"], out["landscape_inferred_prior"]))

    assert "landscape_inferred_prior" in issue_vote_engine.PREDICTORS
    assert values["r_conservative"] > 0
    assert values["r_liberal"] < 0


def test_assemble_excludes_inactive_final_slots() -> None:
    frame = issue_vote_engine.assemble()

    assert frame.loc[frame["election_id"] == "pres_2022", "slot"].unique().tolist() == ["A", "B"]
    assert "C" in set(frame.loc[frame["election_id"] == "pres_2017", "slot"])


def test_assemble_keeps_2022_ahn_transfer_separate_from_final_c_slot() -> None:
    frame = issue_vote_engine.assemble()
    pres_2022 = frame.loc[frame["election_id"] == "pres_2022"]

    assert set(pres_2022["slot"]) == {"A", "B"}
    transfer_a = pres_2022.loc[pres_2022["slot"] == "A", "withdrawn_candidate_transfer"].mean()
    transfer_b = pres_2022.loc[pres_2022["slot"] == "B", "withdrawn_candidate_transfer"].mean()
    assert transfer_a > 0
    assert transfer_b > 0
    assert transfer_a > transfer_b
