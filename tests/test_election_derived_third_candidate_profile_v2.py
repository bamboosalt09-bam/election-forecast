from __future__ import annotations

import pandas as pd
import pytest

from election_forecast.features.region_bloc_prior import THIRD_BLOC
from presidential_issue_engine.election_derived_third_candidate_profile_v2 import (
    build_election_derived_third_profile_v2,
    merge_automatic_viability,
)


def _inputs() -> tuple[pd.DataFrame, ...]:
    speech = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "slot": "C",
                "candidate_name": "candidate",
                "viability": 0.30,
                "centrist_appeal": 0.40,
                "anti_major_party_appeal": 0.50,
                "regional_base_overlap": 0.20,
                "available_date": "2007-12-18",
                "confidence": 0.70,
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
                "organization_strength": 0.50,
                "available_date": "2007-12-18",
                "confidence": 0.80,
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
                "region_id": "r1",
                "bloc": THIRD_BLOC,
                "vote_share": 0.25,
            },
            {
                "election_id": "assembly_2004_pr",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": THIRD_BLOC,
                "vote_share": 0.15,
            },
            {
                "election_id": "assembly_2004_pr",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "major",
                "vote_share": 0.50,
            },
            {
                "election_id": "assembly_2004_pr",
                "election_type": "assembly_pr",
                "region_id": "r2",
                "bloc": THIRD_BLOC,
                "vote_share": 0.20,
            },
            {
                "election_id": "assembly_2004_pr",
                "election_type": "assembly_pr",
                "region_id": "r2",
                "bloc": "major",
                "vote_share": 0.60,
            },
            {
                "election_id": "assembly_2004_district",
                "election_type": "assembly_district",
                "region_id": "r1",
                "bloc": THIRD_BLOC,
                "vote_share": 0.10,
            },
            {
                "election_id": "assembly_2004_district",
                "election_type": "assembly_district",
                "region_id": "r1",
                "bloc": "major",
                "vote_share": 0.50,
            },
        ]
    )
    return speech, context, candidate_history, results, history


def test_target_result_is_excluded_from_v2_stature() -> None:
    inputs = _inputs()
    base, _ = build_election_derived_third_profile_v2(*inputs)
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
    changed, _ = build_election_derived_third_profile_v2(
        inputs[0], inputs[1], inputs[2], changed_results, inputs[4]
    )
    assert float(base.iloc[0]["viability"]) == float(changed.iloc[0]["viability"])


def test_direct_party_rows_are_aggregated_before_competitiveness() -> None:
    _, audit = build_election_derived_third_profile_v2(*_inputs())
    # Third bloc mean is (0.40 + 0.20) / 2 and major mean is
    # (0.50 + 0.60) / 2. Splitting the r1 third vote into two rows must not
    # dilute its strength.
    expected = 0.30 / 0.55
    assert float(audit.iloc[0]["direct_party_competitiveness"]) == pytest.approx(
        expected
    )


def test_prior_presidential_stature_is_not_double_counted_with_speech() -> None:
    _, audit = build_election_derived_third_profile_v2(*_inputs())
    row = audit.iloc[0]
    assert row["conversion_mode"] == "prior_candidate_stature_no_double_count"
    assert float(row["viability"]) == pytest.approx(
        float(row["electoral_competitiveness"])
    )


def test_viability_merge_preserves_character_traits_and_unmatched_rows() -> None:
    base = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "slot": "C",
                "candidate_name": "base name",
                "viability": 0.65,
                "centrist_appeal": 0.35,
                "anti_major_party_appeal": 0.65,
                "regional_base_overlap": 0.25,
                "notes": "base",
            },
            {
                "election_id": "pres_2012",
                "slot": "C",
                "candidate_name": "unmatched",
                "viability": 0.02,
                "centrist_appeal": 0.05,
                "anti_major_party_appeal": 0.05,
                "regional_base_overlap": 0.0,
                "notes": "base",
            },
        ]
    )
    automatic = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "slot": "C",
                "candidate_name": "automatic name",
                "viability": 0.66,
                "available_date": "2007-12-18",
                "confidence": 0.6,
            }
        ]
    )
    merged, audit = merge_automatic_viability(base, automatic)
    assert float(merged.loc[0, "viability"]) == pytest.approx(0.66)
    assert float(merged.loc[0, "centrist_appeal"]) == pytest.approx(0.35)
    assert float(merged.loc[0, "anti_major_party_appeal"]) == pytest.approx(0.65)
    assert float(merged.loc[1, "viability"]) == pytest.approx(0.02)
    assert len(audit) == 1
