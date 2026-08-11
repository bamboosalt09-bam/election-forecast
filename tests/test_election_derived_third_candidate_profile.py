from __future__ import annotations

import pandas as pd

from election_forecast.features.region_bloc_prior import THIRD_BLOC
from presidential_issue_engine.election_derived_third_candidate_profile import (
    build_election_derived_third_profile,
)


def _inputs() -> tuple[pd.DataFrame, ...]:
    speech = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "slot": "C",
                "candidate_name": "candidate",
                "viability": 0.3,
                "centrist_appeal": 0.4,
                "anti_major_party_appeal": 0.5,
                "regional_base_overlap": 0.2,
                "available_date": "2007-12-18",
                "confidence": 0.7,
            }
        ]
    )
    context = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "slot": "C",
                "candidate_name": "candidate",
                "bloc": THIRD_BLOC,
                "organization_strength": 0.5,
                "available_date": "2007-12-18",
                "confidence": 0.8,
            }
        ]
    )
    candidate_history = pd.DataFrame(
        columns=[
            "target_election_id",
            "target_candidate_name",
            "source_election_date",
            "source_sg_typecode",
            "prior_election_won",
        ]
    )
    results = pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "slot": "A",
                "candidate_name": "candidate",
                "votes": 40.0,
            },
            {
                "election_id": "pres_2002",
                "slot": "B",
                "candidate_name": "winner",
                "votes": 60.0,
            },
        ]
    )
    history = pd.DataFrame(
        [
            {
                "election_id": "assembly_2004_pr",
                "election_type": "assembly_pr",
                "region_id": "sido_11",
                "bloc": THIRD_BLOC,
                "vote_share": 0.30,
            },
            {
                "election_id": "assembly_2004_pr",
                "election_type": "assembly_pr",
                "region_id": "sido_11",
                "bloc": "major",
                "vote_share": 0.50,
            },
        ]
    )
    return speech, context, candidate_history, results, history


def test_target_result_is_excluded_from_election_stature() -> None:
    inputs = _inputs()
    base, _ = build_election_derived_third_profile(*inputs)
    target = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "slot": "C",
                "candidate_name": "candidate",
                "votes": 999.0,
            }
        ]
    )
    changed_results = pd.concat([inputs[3], target], ignore_index=True)
    changed, _ = build_election_derived_third_profile(
        inputs[0], inputs[1], inputs[2], changed_results, inputs[4]
    )
    assert float(base.iloc[0]["viability"]) == float(changed.iloc[0]["viability"])
    assert 0.0 <= float(base.iloc[0]["viability"]) <= 1.0
