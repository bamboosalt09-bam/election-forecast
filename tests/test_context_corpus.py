from __future__ import annotations

import pandas as pd
import pytest
import json
from pathlib import Path

from election_forecast.context_corpus import validate_context_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_context_corpus_accepts_election_specific_pre_cutoff_rows() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2002", "pres_2022"],
            "available_date": ["2002-12-18", "2022-03-08"],
            "text": ["문장 하나", "문장 둘"],
        }
    )
    result = validate_context_corpus(frame)
    assert result.rows == 2
    assert result.latest_available_date == "2022-03-08"


def test_context_corpus_rejects_post_election_text() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2017"],
            "available_date": ["2017-05-10"],
            "text": ["사후 문장"],
        }
    )
    with pytest.raises(ValueError, match="post-cutoff"):
        validate_context_corpus(frame)


def test_context_corpus_rejects_outcome_columns() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "available_date": ["2022-03-08"],
            "text": ["문장"],
            "actual_vote_share": [0.5],
        }
    )
    with pytest.raises(ValueError, match="outcome-like"):
        validate_context_corpus(frame)


def test_classifier_sign_margin_is_not_treated_as_vote_margin() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "available_date": ["2022-03-08"],
            "text": ["문장"],
            "ensemble_sign_margin": [0.5],
        }
    )
    assert validate_context_corpus(frame).rows == 1


def test_context_model_registry_pins_enabled_model_revisions() -> None:
    registry_path = ROOT / "data" / "shadow" / "context_model_sources.json"
    if not registry_path.exists():
        pytest.skip("external shadow model registry is not part of the public repository")
    registry = json.loads(
        registry_path.read_text(encoding="utf-8")
    )
    enabled = {
        row["id"]: row
        for row in registry["sources"]
        if row.get("enabled") is True
    }
    assert enabled["klue_roberta_small"]["revision"] == (
        "b6b4c36d827e0293ae2fcf04d527072f10a23064"
    )
    assert enabled["ko_sroberta_nli"]["active_forecast_integration"] is False
