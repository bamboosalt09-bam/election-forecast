from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from presidential_issue_engine.speech_derived_issue_profiles import (
    build_candidate_profile,
    build_outputs,
)


DATES = {"pres_test": "2024-12-20"}
ELECTIONS = ("pres_test",)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "slot": "A",
                "candidate_name": "Candidate A",
                "party_name": "Party A",
                "is_active_slot": True,
                "votes": 999,
                "vote_share": 0.99,
            },
            {
                "election_id": "pres_test",
                "slot": "B",
                "candidate_name": "Candidate B",
                "party_name": "Party B",
                "is_active_slot": True,
                "votes": 1,
                "vote_share": 0.01,
            },
        ]
    )
    links = pd.DataFrame(
        [
            ["pres_test", "A", "economy_growth", 90.0, 1.0, 0.75, "2024-12-10"],
            ["pres_test", "A", "housing", 30.0, 0.4, 0.25, "2024-12-10"],
            ["pres_test", "B", "economy_growth", 20.0, 0.3, 0.20, "2024-12-10"],
            ["pres_test", "B", "housing", 80.0, 1.0, 0.80, "2024-12-10"],
        ],
        columns=[
            "election_id",
            "slot",
            "issue_name",
            "mentions",
            "emphasis_volume",
            "emphasis_within",
            "available_date",
        ],
    )
    salience = pd.DataFrame(
        [
            ["pres_test", "economy_growth", "2024-12-01", 8.0, 0.8, "assembly", "2024-12-08"],
            ["pres_test", "housing", "2024-12-01", 4.0, 0.4, "assembly", "2024-12-08"],
        ],
        columns=[
            "election_id",
            "issue_name",
            "period",
            "raw_value",
            "salience_score",
            "instrument",
            "available_date",
        ],
    )
    rows: list[dict[str, object]] = []
    for slot in ["A", "B"]:
        for issue, evidence in [("economy_growth", 12), ("housing", 4)]:
            explicit = slot == "A" and issue == "economy_growth"
            rows.append(
                {
                    "election_id": "pres_test",
                    "slot": slot,
                    "issue_name": issue,
                    "available_date": "2024-12-09",
                    "issue_confidence_quality": 0.8 if issue == "economy_growth" else 0.4,
                    "issue_evidence_count": evidence,
                    "issue_speaker_count": evidence,
                    "issue_committee_count": 2,
                    "link_evidence_count": 2 if explicit else 0,
                    "link_reliability": 0.8 if explicit else 0.0,
                    "target_signed_evidence": -1.2 if explicit else 0.0,
                    "target_absolute_evidence": 1.5 if explicit else 0.0,
                    "target_directional_balance": -0.8 if explicit else 0.0,
                    "target_attribution_confidence": 0.75 if explicit else 0.0,
                    "target_source_types": "person" if explicit else "",
                    "accountability_score": 0.2,
                    "polarized_score": 0.1,
                    "character_intensity": 0.4,
                }
            )
    return links, salience, pd.DataFrame(rows), candidates


def test_profile_uses_explicit_target_for_direction_only() -> None:
    links, salience, character, candidates = _inputs()
    profile = build_candidate_profile(
        links, salience, character, candidates, DATES, ELECTIONS
    )
    target = profile.loc[
        profile["slot"].eq("A") & profile["issue_name"].eq("economy_growth")
    ].iloc[0]
    assert target["direction"] == pytest.approx(-0.8)
    assert target["direction_confidence"] == pytest.approx(0.6)
    assert target["association_strength"] >= target["unsigned_association"]
    assert profile.loc[
        ~(
            profile["slot"].eq("A")
            & profile["issue_name"].eq("economy_growth")
        ),
        "direction",
    ].eq(0.0).all()
    assert profile["association_strength"].between(0.0, 1.0).all()
    assert profile["confidence"].between(0.0, 1.0).all()


def test_candidate_outcome_columns_cannot_change_profile() -> None:
    links, salience, character, candidates = _inputs()
    first = build_candidate_profile(
        links, salience, character, candidates, DATES, ELECTIONS
    )
    mutated = candidates.copy()
    mutated["votes"] = [0, 10_000_000]
    mutated["vote_share"] = [0.0, 1.0]
    second = build_candidate_profile(
        links, salience, character, mutated, DATES, ELECTIONS
    )
    assert_frame_equal(first, second)


def test_outputs_keep_only_explicit_signed_mega_attribution() -> None:
    links, salience, character, candidates = _inputs()
    outputs = build_outputs(
        links, salience, character, candidates, DATES, ELECTIONS
    )
    profile = outputs["candidate_issue_profile.csv"]
    axis = outputs["mega_issue_axis.csv"]
    attribution = outputs["mega_issue_attribution.csv"]
    assert len(profile) == 4
    assert set(axis["primary_issue"]) == {"economy_growth", "housing"}
    assert len(attribution) == 1
    row = attribution.iloc[0]
    assert row["target"] == "A"
    assert row["polarity"] == -1.0
    assert row["confidence"] == pytest.approx(0.6)
