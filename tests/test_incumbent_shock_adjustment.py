from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.incumbent_shock_adjustment import (
    apply_incumbent_shock_response,
    compile_government_burden_scores,
)


DATES = {"pres_test": "2020-01-10"}


def test_government_burden_compiler_is_directional_and_point_in_time() -> None:
    profile = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "slot": "B",
                "issue_name": "economy_growth",
                "direction": -1.0,
                "association_strength": 0.8,
                "confidence": 0.5,
                "target_absolute_evidence": 2.0,
                "target_attribution_confidence": 0.75,
                "target_source_types": "government",
                "available_date": "2020-01-01",
            },
            {
                "election_id": "pres_test",
                "slot": "B",
                "issue_name": "housing",
                "direction": 1.0,
                "association_strength": 1.0,
                "confidence": 1.0,
                "target_absolute_evidence": 5.0,
                "target_attribution_confidence": 1.0,
                "target_source_types": "government",
                "available_date": "2020-01-11",
            },
        ]
    )
    scores = compile_government_burden_scores(profile, DATES)
    assert len(scores) == 1
    assert scores.loc[0, "government_direction_score"] == pytest.approx(-0.4)
    assert scores.loc[0, "government_evidence_count"] == 1
    assert scores.loc[0, "government_negative_share"] == pytest.approx(1.0)
    assert scores.loc[0, "government_rejection_breadth"] == pytest.approx(0.25)
    assert scores.loc[0, "government_rejection_strength"] == pytest.approx(0.2)


def _response_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": ["pres_test"] * 3,
            "region_id": ["r1"] * 3,
            "source_slot": ["A", "B", "C"],
            "pred": [0.45, 0.40, 0.15],
            "direct_party_recent_base": [0.40, 0.30, 0.10],
            "direct_party_reliability": [0.8, 0.8, 0.8],
            "direct_mega_score": [0.0, -0.25, 0.0],
            "actual": [0.0, 1.0, 0.0],
            "contest_votes": [1.0, 1000.0, 1.0],
        }
    )


def test_response_penalizes_exposed_candidate_and_preserves_composition() -> None:
    scores = pd.DataFrame(
        {
            "election_id": ["pres_test"],
            "slot": ["B"],
            "government_direction_score": [-0.2],
            "government_evidence_weight": [1.0],
            "government_evidence_count": [2],
        }
    )
    intensity = pd.DataFrame(
        {
            "election_id": ["pres_test"],
            "mega_issue_intensity": [2.0],
            "available_date": ["2020-01-01"],
        }
    )
    out = apply_incumbent_shock_response(
        _response_frame(), scores, intensity, DATES, prediction_column="pred"
    )
    assert out["pred"].sum() == pytest.approx(1.0)
    assert out.loc[out["source_slot"].eq("B"), "pred"].iloc[0] < 0.40
    assert out.loc[out["source_slot"].eq("A"), "incumbent_burden_exposure"].iloc[0] == 0.0


def test_response_does_not_depend_on_actual_or_contest_vote_columns() -> None:
    scores = pd.DataFrame(
        {
            "election_id": ["pres_test"],
            "slot": ["B"],
            "government_direction_score": [-0.2],
            "government_evidence_weight": [1.0],
            "government_evidence_count": [2],
        }
    )
    intensity = pd.DataFrame(
        {
            "election_id": ["pres_test"],
            "mega_issue_intensity": [1.0],
            "available_date": ["2020-01-01"],
        }
    )
    first = apply_incumbent_shock_response(
        _response_frame(), scores, intensity, DATES, prediction_column="pred"
    )
    changed = _response_frame()
    changed["actual"] = [1.0, 0.0, 0.0]
    changed["contest_votes"] = [5000.0, 1.0, 1.0]
    second = apply_incumbent_shock_response(
        changed, scores, intensity, DATES, prediction_column="pred"
    )
    assert second["pred"].tolist() == pytest.approx(first["pred"].tolist())
