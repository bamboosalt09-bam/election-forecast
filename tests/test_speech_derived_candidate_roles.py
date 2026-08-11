from __future__ import annotations

import pandas as pd

from presidential_issue_engine.speech_derived_candidate_roles import (
    build_automatic_third_candidate_profile,
)


def _speech() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "slot": slot,
                "candidate_name": slot,
                "bloc": bloc,
                "party_elite_support_score": support,
                "party_elite_fragmentation_score": fragmentation,
                "organization_strength": organization,
                "outsider_status": outsider,
                "available_date": "2020-01-01",
                "confidence": 0.8,
            }
            for slot, bloc, support, fragmentation, organization, outsider in [
                ("A", "더불어민주당", 0.8, 0.1, 0.9, 0.0),
                ("B", "국민의힘", 0.7, 0.2, 0.9, 0.0),
                ("C", "제3지대", 0.6, 0.3, 0.6, 0.4),
                ("D", "제3지대", 0.2, 0.8, 0.2, 0.8),
            ]
        ]
    )


def _treatment() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "slot": slot,
                "candidate_name": slot,
                "serious_contender_score": serious,
                "legitimacy_score": legitimacy,
                "alternative_score": alternative,
                "protest_vote_score": protest,
                "available_date": "2020-01-01",
                "confidence": 0.75,
            }
            for slot, serious, legitimacy, alternative, protest in [
                ("A", 0.9, 0.8, 0.2, 0.1),
                ("B", 0.8, 0.7, 0.2, 0.1),
                ("C", 0.7, 0.6, 0.7, 0.6),
                ("D", 0.2, 0.2, 0.5, 0.4),
            ]
        ]
    )


def _landscape() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "slot": slot,
                "candidate_name": slot,
                "centrist": centrist,
                "anti_establishment": anti,
                "available_date": "2020-01-01",
                "confidence": 0.7,
            }
            for slot, centrist, anti in [
                ("A", 0.3, 0.2),
                ("B", 0.3, 0.2),
                ("C", 0.8, 0.7),
                ("D", 0.6, 0.6),
            ]
        ]
    )


def test_profile_excludes_major_candidates_and_orders_nonmajor_viability() -> None:
    result = build_automatic_third_candidate_profile(
        _speech(), _treatment(), _landscape(), {"pres_x": "2020-01-02"}
    ).set_index("slot")
    assert set(result.index) == {"C", "D"}
    assert result.loc["C", "viability"] > result.loc["D", "viability"]
    assert result.loc["C", "centrist_appeal"] > result.loc["D", "centrist_appeal"]
    assert result["major_party_core_eligible"].eq(False).all()


def test_future_evidence_is_not_used() -> None:
    treatment = _treatment()
    treatment.loc[treatment["slot"].eq("C"), "available_date"] = "2020-01-03"
    result = build_automatic_third_candidate_profile(
        _speech(), treatment, _landscape(), {"pres_x": "2020-01-02"}
    )
    assert "C" not in set(result["slot"])
