from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.speech_landscape_builder import (
    build_landscape_from_issue_links,
    candidate_metadata_from_results,
    merge_manual_rows,
)


def test_build_landscape_from_issue_links_weights_axes() -> None:
    issue_link = pd.DataFrame(
        [
            {"election_id": "pres_x", "slot": "A", "issue_name": "security_nk", "emphasis_within": 0.75},
            {"election_id": "pres_x", "slot": "A", "issue_name": "welfare_pension", "emphasis_within": 0.25},
            {"election_id": "pres_x", "slot": "alpha", "issue_name": "security_nk", "emphasis_within": 1.0},
        ]
    )
    axis_map = pd.DataFrame(
        [
            {
                "issue_name": "security_nk",
                "conservative": 1.0,
                "liberal": 0.0,
                "progressive": 0.0,
                "centrist": 0.0,
                "anti_establishment": 0.0,
                "reform": 0.0,
                "regionalist": 0.0,
            },
            {
                "issue_name": "welfare_pension",
                "conservative": 0.0,
                "liberal": 1.0,
                "progressive": 1.0,
                "centrist": 0.0,
                "anti_establishment": 0.0,
                "reform": 0.0,
                "regionalist": 0.0,
            },
        ]
    )
    metadata = pd.DataFrame(
        [{"election_id": "pres_x", "slot": "A", "candidate_name": "Candidate A"}]
    )

    out = build_landscape_from_issue_links(
        issue_link,
        metadata,
        axis_map,
        available_date_by_election={"pres_x": "2026-01-01"},
    )

    assert len(out) == 1
    assert out.loc[0, "candidate_name"] == "Candidate A"
    assert out.loc[0, "candidate_role"] == "final"
    assert out.loc[0, "available_date"] == "2026-01-01"
    assert out.loc[0, "conservative"] == pytest.approx(0.75)
    assert out.loc[0, "liberal"] == pytest.approx(0.25)
    assert out.loc[0, "progressive"] == pytest.approx(0.25)


def test_candidate_metadata_from_results_keeps_active_non_alpha_slots() -> None:
    results = pd.DataFrame(
        [
            {"election_id": "pres_x", "slot": "A", "candidate_name": "A", "is_active_slot": True},
            {"election_id": "pres_x", "slot": "C", "candidate_name": "C", "is_active_slot": False},
            {"election_id": "pres_x", "slot": "alpha", "candidate_name": "Other", "is_active_slot": True},
        ]
    )

    out = candidate_metadata_from_results(results)

    assert out.to_dict("records") == [
        {"election_id": "pres_x", "slot": "A", "candidate_name": "A"}
    ]


def test_merge_manual_rows_preserves_withdrawn_rows_but_replaces_final_rows() -> None:
    generated = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "slot": "A",
                "candidate_name": "Generated",
                "candidate_role": "final",
                "conservative": 0.7,
                "liberal": 0.1,
                "progressive": 0.0,
                "centrist": 0.3,
                "anti_establishment": 0.0,
                "reform": 0.2,
                "regionalist": 0.0,
                "available_date": "2026-01-01",
                "confidence": 0.5,
                "notes": "generated",
            }
        ]
    )
    manual = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "slot": "A",
                "candidate_name": "Manual",
                "candidate_role": "final",
                "conservative": 0.1,
                "liberal": 0.1,
                "progressive": 0.1,
                "centrist": 0.1,
                "anti_establishment": 0.1,
                "reform": 0.1,
                "regionalist": 0.1,
                "available_date": "2026-01-01",
                "confidence": 0.5,
                "notes": "manual final",
            },
            {
                "election_id": "pres_x",
                "slot": "C",
                "candidate_name": "Withdrawn",
                "candidate_role": "withdrawn",
                "conservative": 0.2,
                "liberal": 0.2,
                "progressive": 0.2,
                "centrist": 0.9,
                "anti_establishment": 0.8,
                "reform": 0.8,
                "regionalist": 0.0,
                "available_date": "2026-01-01",
                "confidence": 0.6,
                "notes": "manual withdrawn",
            },
        ]
    )

    out = merge_manual_rows(generated, manual)

    assert set(out["candidate_name"]) == {"Generated", "Withdrawn"}
    assert not (out["candidate_name"] == "Manual").any()
