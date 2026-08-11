from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import build_candidate_vote_conversion_context as builder
from presidential_issue_engine import issue_vote_engine


def test_candidate_vote_conversion_builder_distinguishes_viable_and_weak_third(
    tmp_path,
    monkeypatch,
) -> None:
    public = tmp_path / "candidate_public_treatment.csv"
    public.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,serious_contender_score,legitimacy_score,negative_treatment_score,alternative_score,protest_vote_score,ridicule_or_gaffe_score,public_treatment_support,available_date,confidence,notes",
                "pres_2017,C,Viable Third,0.85,0.80,0.10,0.60,0.45,0.05,0.70,2017-04-01,0.90,test",
                "pres_future,C,Weak Third,0.20,0.15,0.25,0.70,0.65,0.20,0.55,2025-05-01,0.70,test",
                "pres_future,A,Major A,0.80,0.75,0.20,0.30,0.20,0.05,0.60,2025-05-01,0.90,test",
                "pres_future,B,Major B,0.75,0.70,0.20,0.30,0.20,0.05,0.55,2025-05-01,0.90,test",
            ]
        ),
        encoding="utf-8",
    )
    party = tmp_path / "candidate_party_speech_context.csv"
    party.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,party_elite_support_score,party_elite_fragmentation_score,party_context_support,organization_strength,outsider_status,available_date,confidence,notes",
                "pres_2017,C,Viable Third,0.60,0.15,0.45,0.70,0.20,2017-04-01,0.80,test",
                "pres_future,C,Weak Third,0.10,0.45,-0.10,0.30,0.70,2025-05-01,0.70,test",
                "pres_future,A,Major A,0.90,0.10,0.80,0.95,0.05,2025-05-01,0.90,test",
                "pres_future,B,Major B,0.85,0.10,0.80,0.95,0.05,2025-05-01,0.90,test",
            ]
        ),
        encoding="utf-8",
    )
    tone = tmp_path / "candidate_party_tone_gap.csv"
    tone.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,same_party_supportive_tone,cross_party_positive_tone,cross_party_adverse_tone,party_tone_contrast,available_date,confidence,notes",
                "pres_2017,C,Viable Third,0.70,0.20,0.20,0.50,2017-04-01,0.80,test",
                "pres_future,C,Weak Third,0.15,0.40,0.30,-0.15,2025-05-01,0.70,test",
                "pres_future,A,Major A,0.85,0.20,0.20,0.60,2025-05-01,0.90,test",
                "pres_future,B,Major B,0.85,0.20,0.20,0.60,2025-05-01,0.90,test",
            ]
        ),
        encoding="utf-8",
    )
    third = tmp_path / "third_candidate_profile.csv"
    third.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,viability,centrist_appeal,anti_major_party_appeal,regional_base_overlap,available_date,confidence,notes",
                "pres_2017,C,Viable Third,0.90,0.80,0.70,0.45,2017-04-01,0.80,test",
                "pres_future,C,Weak Third,0.35,0.45,0.65,0.15,2025-05-01,0.60,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "CANDIDATE_PUBLIC_TREATMENT", public)
    monkeypatch.setattr(builder, "CANDIDATE_PARTY_SPEECH_CONTEXT", party)
    monkeypatch.setattr(builder, "CANDIDATE_PARTY_TONE_GAP", tone)
    monkeypatch.setattr(builder, "THIRD_CANDIDATE_PROFILE", third)

    out = builder.build()
    viable = out.loc[out["candidate_name"] == "Viable Third"].iloc[0]
    weak = out.loc[out["candidate_name"] == "Weak Third"].iloc[0]

    assert viable["candidate_weight"] > weak["candidate_weight"]
    assert viable["wasted_vote_resistance"] > weak["wasted_vote_resistance"]
    assert weak["major_party_gravity"] > viable["major_party_gravity"]
    assert weak["third_candidate_overexposure_risk"] > 0


def test_candidate_conversion_context_loader_excludes_future_rows(tmp_path, monkeypatch) -> None:
    context = tmp_path / "candidate_vote_conversion_context.csv"
    context.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,candidate_weight,coalition_cohesion,wasted_vote_resistance,major_party_gravity,third_candidate_overexposure_risk,attention_to_support_gap,conversion_capacity,available_date,confidence,notes",
                "pres_2017,C,Third,0.5,0.4,0.5,0.2,0.1,0.1,0.4,2018-01-01,0.8,future",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_VOTE_CONVERSION_CONTEXT", str(context))

    loaded = issue_vote_engine._load_candidate_vote_conversion_context()

    assert loaded.empty


def test_candidate_conversion_context_adjustment_penalizes_low_resistance_third(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_future",
                "region_id": "r1",
                "slot": "A",
                "conversion_capacity_centered": 0.10,
                "wasted_vote_resistance": 0.80,
                "major_party_gravity": 0.0,
                "third_candidate_overexposure_risk": 0.0,
                "candidate_conversion_confidence": 1.0,
            },
            {
                "election_id": "pres_future",
                "region_id": "r1",
                "slot": "B",
                "conversion_capacity_centered": 0.05,
                "wasted_vote_resistance": 0.75,
                "major_party_gravity": 0.0,
                "third_candidate_overexposure_risk": 0.0,
                "candidate_conversion_confidence": 1.0,
            },
            {
                "election_id": "pres_future",
                "region_id": "r1",
                "slot": "C",
                "conversion_capacity_centered": -0.15,
                "wasted_vote_resistance": 0.20,
                "major_party_gravity": 0.50,
                "third_candidate_overexposure_risk": 0.30,
                "candidate_conversion_confidence": 1.0,
            },
        ]
    )

    monkeypatch.setattr(
        issue_vote_engine,
        "THROUGH_2022_REDERIVED_LAYER_CONFIG",
        {
            "conversion_scale": 0.1,
            "third_competitiveness_gate_enabled": False,
            "third_character_multiplier_enabled": False,
        },
    )
    adjusted = issue_vote_engine.apply_candidate_conversion_context_adjustment(
        frame,
        [0.45, 0.40, 0.15],
    )

    assert adjusted[2] < 0.15
    assert adjusted.sum() == pytest.approx(1.0)
