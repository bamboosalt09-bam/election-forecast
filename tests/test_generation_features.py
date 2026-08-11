from __future__ import annotations

import pandas as pd

from presidential_issue_engine.build_candidate_generation_profile_from_assembly import build_generation_profile
from presidential_issue_engine import issue_vote_engine


def test_candidate_generation_features_are_election_centered(tmp_path, monkeypatch) -> None:
    profile = tmp_path / "candidate_generation_profile.csv"
    weights = tmp_path / "election_generation_weights.csv"
    profile.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,young_affinity,middle_affinity,senior_affinity,available_date,confidence,notes",
                "pres_2007,A,A,0.4,0.7,0.6,2007-12-01,1.0,broad",
                "pres_2007,C,C,0.7,0.3,0.3,2007-12-01,1.0,youth niche",
            ]
        ),
        encoding="utf-8",
    )
    weights.write_text(
        "\n".join(
            [
                "election_id,young_weight,middle_weight,senior_weight,available_date,notes",
                "pres_2007,0.2,0.5,0.3,2007-12-01,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_GENERATION_PROFILE", str(profile))
    monkeypatch.setattr(issue_vote_engine, "ELECTION_GENERATION_WEIGHTS", str(weights))

    base = pd.DataFrame(
        [
            {"election_id": "pres_2007", "region_id": "r1", "slot": "A"},
            {"election_id": "pres_2007", "region_id": "r1", "slot": "C"},
        ]
    )

    out = issue_vote_engine._candidate_generation_features(base)

    a = out.loc[out["slot"] == "A"].iloc[0]
    c = out.loc[out["slot"] == "C"].iloc[0]
    assert a["generation_support_score"] > c["generation_support_score"]
    assert a["generation_alignment"] > 0
    assert c["generation_alignment"] < 0
    assert c["generation_youth_niche"] > a["generation_youth_niche"]


def test_candidate_generation_features_filter_future_rows(tmp_path, monkeypatch) -> None:
    profile = tmp_path / "candidate_generation_profile.csv"
    weights = tmp_path / "election_generation_weights.csv"
    profile.write_text(
        "\n".join(
            [
                "election_id,slot,candidate_name,young_affinity,middle_affinity,senior_affinity,available_date,confidence,notes",
                "pres_2007,A,A,1.0,1.0,1.0,2008-01-01,1.0,future",
            ]
        ),
        encoding="utf-8",
    )
    weights.write_text(
        "election_id,young_weight,middle_weight,senior_weight,available_date,notes\n"
        "pres_2007,0.2,0.5,0.3,2007-12-01,test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_GENERATION_PROFILE", str(profile))
    monkeypatch.setattr(issue_vote_engine, "ELECTION_GENERATION_WEIGHTS", str(weights))

    base = pd.DataFrame([{"election_id": "pres_2007", "region_id": "r1", "slot": "A"}])

    out = issue_vote_engine._candidate_generation_features(base)

    assert out["generation_alignment"].iloc[0] == 0.0
    assert out["generation_confidence"].iloc[0] == 0.0


def test_generation_prediction_adjustment_preserves_group_sum_and_penalizes_youth_niche() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "region_id": "r1",
                "slot": "A",
                "generation_alignment": 0.09,
                "generation_youth_niche": -0.01,
                "generation_confidence": 0.65,
            },
            {
                "election_id": "pres_2007",
                "region_id": "r1",
                "slot": "C",
                "generation_alignment": -0.05,
                "generation_youth_niche": 0.02,
                "generation_confidence": 0.65,
            },
        ]
    )
    pred = pd.Series([0.50, 0.50])

    out = issue_vote_engine.apply_generation_prediction_adjustment(frame, pred)

    assert out[0] > 0.50
    assert out[1] < 0.50
    assert out.sum() == 1.0


def test_build_generation_profile_from_assembly_issue_links(tmp_path) -> None:
    link = tmp_path / "candidate_issue_link.csv"
    sensitivity = tmp_path / "generation_issue_sensitivity.csv"
    results = tmp_path / "presidential_results_standardized.csv"
    link.write_text(
        "\n".join(
            [
                "election_id,slot,issue_name,mentions,emphasis_volume,emphasis_within,available_date",
                "pres_2007,A,jobs_labor,10,1,0.8,2007-12-01",
                "pres_2007,A,security_nk,1,1,0.2,2007-12-01",
                "pres_2007,C,jobs_labor,1,1,0.2,2007-12-01",
                "pres_2007,C,security_nk,10,1,0.8,2007-12-01",
            ]
        ),
        encoding="utf-8",
    )
    sensitivity.write_text(
        "\n".join(
            [
                "issue_name,youth_lean,notes",
                "jobs_labor,0.5,youth",
                "security_nk,-0.7,senior",
            ]
        ),
        encoding="utf-8",
    )
    results.write_text(
        "\n".join(
            [
                "election_id,region_id,slot,candidate_name,party_name,votes,vote_share,is_active_slot",
                "pres_2007,r1,A,A,party,1,0.5,true",
                "pres_2007,r1,C,C,party,1,0.5,true",
            ]
        ),
        encoding="utf-8",
    )

    out = build_generation_profile(link, sensitivity, results, affinity_scale=3.0)

    a = out.loc[out["slot"] == "A"].iloc[0]
    c = out.loc[out["slot"] == "C"].iloc[0]
    assert a["young_affinity"] > c["young_affinity"]
    assert c["senior_affinity"] > a["senior_affinity"]
    assert a["candidate_name"] == "A"
