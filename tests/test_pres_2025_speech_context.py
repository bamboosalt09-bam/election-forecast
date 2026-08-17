from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from presidential_issue_engine import build_candidate_vote_conversion_context
from scripts import build_speech_derived_candidate_context_v2 as candidate_context_v2
from scripts import build_speech_derived_issue_context as issue_context
from scripts import extract_assembly_speaker_issue_matches as extractor
from src.news_collector.sources.assembly_batch import ELECTION_WINDOWS


def _registry(path: Path, *, outcome: bool = False) -> Path:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2025"] * 4,
            "candidate_id": ["a", "b", "c", "minor"],
            "candidate_name": ["A", "B", "C", "minor"],
            "party_name": ["p1", "p2", "p3", "p4"],
            "ballot_number": [1, 2, 4, 5],
            "available_date": ["2025-05-20"] * 4,
        }
    )
    if outcome:
        frame["vote_share"] = 0.0
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_pres_2025_extraction_scope_ends_at_d_minus_one() -> None:
    assert extractor.ELECTION_DATES["pres_2025"] == pd.Timestamp("2025-06-03")
    assert extractor.ELECTION_CUTOFFS["pres_2025"] == (
        pd.Timestamp("2022-03-10"),
        pd.Timestamp("2025-06-02"),
    )
    assert ELECTION_WINDOWS["pres_2025"] == ("2025-03-05", "2025-06-02")
    assert extractor.election_for_date("2025-06-02") == "pres_2025"
    assert extractor.election_for_date("2025-06-03") == ""
    assert extractor.election_for_date("2025-03-04", window_mode="campaign") == ""
    assert extractor.election_for_date("2025-03-05", window_mode="campaign") == "pres_2025"
    assert extractor.assembly_allowed_for_election("21", "pres_2025")
    assert extractor.assembly_allowed_for_election("22", "pres_2025")
    assert not extractor.assembly_allowed_for_election("16", "pres_2007")
    assert extractor.ELECTION_TO_ASSEMBLY["pres_2012"] == "19"


def test_forecast_registry_uses_only_ballots_one_two_four(tmp_path: Path) -> None:
    raw, candidates = issue_context._forecast_candidate_registry(
        _registry(tmp_path / "registry.csv")
    )
    assert raw["candidate_id"].tolist() == ["a", "b", "c"]
    assert candidates["slot"].tolist() == ["A", "B", "C"]
    assert candidates["is_active_slot"].all()


def test_forecast_registry_rejects_outcome_columns(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="outcome columns"):
        issue_context._forecast_candidate_registry(
            _registry(tmp_path / "registry.csv", outcome=True)
        )


def test_history_prefix_is_preserved_byte_for_byte(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    history.write_bytes(b"election_id,slot,candidate_name,value\r\npres_2022,A,old,0.1\r\n")
    output = tmp_path / "combined.csv"
    target = pd.DataFrame(
        {
            "election_id": ["pres_2025"],
            "slot": ["A"],
            "candidate_name": ["new"],
            "value": [0.2],
        }
    )

    candidate_context_v2._prepend_frozen_history(history, target, output)

    assert output.read_bytes().startswith(history.read_bytes())
    combined = pd.read_csv(output)
    assert combined["election_id"].tolist() == ["pres_2022", "pres_2025"]


def test_conversion_builder_uses_central_2025_date() -> None:
    assert build_candidate_vote_conversion_context.ELECTION_DATES["pres_2025"] == "2025-06-03"


def test_official_supplement_conversion_enforces_campaign_and_availability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "official.csv"
    output = tmp_path / "matches.csv"
    base = {
        "election_id": "pres_2025",
        "assembly_daesu": "22",
        "source_id": "official_minutes_1",
        "source_file": "https://example.test/1",
        "source_row_id": "7",
        "sentence_index": "2",
        "period": "2025-03-03",
        "committee": "committee",
        "agenda": "agenda",
        "speaker": "speaker",
        "member_id": "10",
        "issue_name": "economy_growth",
        "issue_weight": "1.25",
        "text_excerpt": "a sufficiently informative sentence",
    }
    pd.DataFrame(
        [
            {**base, "meeting_date": "2025-03-05", "available_date": "2025-06-02"},
            {**base, "meeting_date": "2025-03-04", "available_date": "2025-06-02"},
            {**base, "meeting_date": "2025-03-06", "available_date": "2025-06-03"},
        ]
    ).to_csv(source, index=False, encoding="utf-8-sig")

    converted = extractor.convert_pres_2025_official_supplement(source, output)

    assert len(converted) == 1
    assert converted.loc[0, "source_row_id"] == "official_minutes_1:7:2"
    assert converted.loc[0, "meeting_date"] == "2025-03-05"
    assert converted.loc[0, "matched_term_count"] == 1


def test_official_supplement_rejects_outcome_columns(tmp_path: Path) -> None:
    source = tmp_path / "official.csv"
    pd.DataFrame({"actual_vote_share": [0.5]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="forbidden outcome columns"):
        extractor.convert_pres_2025_official_supplement(source, tmp_path / "out.csv")
