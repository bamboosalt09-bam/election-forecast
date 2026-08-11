from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine


def test_neutral_issue_context_loader_filters_future_rows(tmp_path, monkeypatch) -> None:
    context = tmp_path / "assembly_neutral_issue_context.csv"
    pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "A",
                "candidate_name": "Candidate A",
                "assembly_neutral_issue_signal": 0.20,
                "available_date": "2017-04-30",
                "confidence": 0.80,
            },
            {
                "election_id": "pres_2017",
                "slot": "B",
                "candidate_name": "Candidate B",
                "assembly_neutral_issue_signal": -0.20,
                "available_date": "2017-05-10",
                "confidence": 0.80,
            },
        ]
    ).to_csv(context, index=False)
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_NEUTRAL_ISSUE_CONTEXT", str(context))

    loaded = issue_vote_engine._load_candidate_neutral_issue_context()

    assert list(loaded["candidate_name"]) == ["Candidate A"]


def test_neutral_issue_context_features_preserve_signal_and_fallback(tmp_path, monkeypatch) -> None:
    context = tmp_path / "assembly_neutral_issue_context.csv"
    pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "A",
                "candidate_name": "Candidate A",
                "assembly_neutral_issue_signal": 0.20,
                "evidence_count": 4,
                "context_neutral_count": 20,
                "global_context_neutral_count": 100,
                "global_context_strength": 0.75,
                "available_date": "2017-04-30",
                "confidence": 0.50,
            },
            {
                "election_id": "pres_2017",
                "slot": "B",
                "candidate_name": "Candidate B",
                "assembly_neutral_issue_signal": -0.10,
                "evidence_count": 2,
                "context_neutral_count": 10,
                "global_context_neutral_count": 80,
                "global_context_strength": 0.60,
                "available_date": "2017-04-30",
                "confidence": 0.40,
            },
        ]
    ).to_csv(context, index=False)
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_NEUTRAL_ISSUE_CONTEXT", str(context))
    base = pd.DataFrame(
        [
            {"election_id": "pres_2017", "region_id": "r1", "slot": "A", "candidate_name": "Candidate A"},
            {"election_id": "pres_2017", "region_id": "r1", "slot": "B", "candidate_name": "Candidate B"},
        ]
    )

    features = issue_vote_engine._candidate_neutral_issue_context_features(base)

    assert features["assembly_neutral_issue_signal"].sum() == pytest.approx(0.10)
    assert features.loc[features["slot"].eq("A"), "assembly_neutral_issue_signal"].iloc[0] == pytest.approx(0.20)
    assert features.loc[features["slot"].eq("B"), "assembly_neutral_issue_signal"].iloc[0] == pytest.approx(-0.10)
    assert "assembly_neutral_issue_signal" not in issue_vote_engine.PREDICTORS

    monkeypatch.setattr(
        issue_vote_engine,
        "CANDIDATE_NEUTRAL_ISSUE_CONTEXT",
        str(tmp_path / "missing.csv"),
    )
    fallback = issue_vote_engine._candidate_neutral_issue_context_features(base)
    assert fallback["assembly_neutral_issue_signal"].eq(0.0).all()


def test_neutral_issue_context_adjustment_preserves_vote_share_composition() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "region_id": "r1",
                "slot": "A",
                "assembly_neutral_issue_signal": -0.10,
            },
            {
                "election_id": "pres_2022",
                "region_id": "r1",
                "slot": "B",
                "assembly_neutral_issue_signal": 0.10,
            },
        ]
    )

    adjusted = issue_vote_engine.apply_neutral_issue_context_adjustment(
        frame,
        pd.Series([0.60, 0.40]),
        0.60,
    )

    assert adjusted.sum() == pytest.approx(1.0)
    assert adjusted[0] < 0.60
    assert adjusted[1] > 0.40


def test_neutral_issue_context_adjustment_uses_fixed_default_and_can_be_disabled(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "region_id": "r1",
                "slot": "A",
                "assembly_neutral_issue_signal": -0.10,
            },
            {
                "election_id": "pres_2022",
                "region_id": "r1",
                "slot": "B",
                "assembly_neutral_issue_signal": 0.10,
            },
        ]
    )
    baseline = pd.Series([0.60, 0.40])

    adjusted = issue_vote_engine.apply_neutral_issue_context_adjustment(frame, baseline)
    monkeypatch.setitem(
        issue_vote_engine.THROUGH_2022_REDERIVED_LAYER_CONFIG,
        "neutral_context_scale",
        0.0,
    )
    disabled = issue_vote_engine.apply_neutral_issue_context_adjustment(frame, baseline)

    assert adjusted == pytest.approx([0.54, 0.46])
    assert disabled == pytest.approx([0.60, 0.40])
