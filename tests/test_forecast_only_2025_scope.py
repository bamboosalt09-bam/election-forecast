from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from presidential_issue_engine import issue_vote_engine
from presidential_issue_engine.election_scope import (
    ELECTION_DATES,
    FORECAST_ONLY_ELECTIONS,
    SCORED_ELECTIONS,
)
from presidential_issue_engine.point_in_time import (
    filter_available_by_election,
    forecast_cutoff,
)
from presidential_issue_engine.forecast_only_inputs import (
    attach_preliminary_slots,
    load_forecast_only_assembly_inputs,
)
from scripts.build_pres_2025_assembly_context import (
    REQUIRED_COLUMNS,
    build_context,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "election_id": "pres_2025",
        "assembly_daesu": "22",
        "source_id": "fixture",
        "source_file": "fixture.xlsx",
        "meeting_date": "2025-06-02",
        "available_date": "2025-06-02",
        "period": "2025-W23",
        "issue_name": "economy_growth",
        "issue_weight": 1.0,
        "target_type": "person",
        "target_name": "candidate_x",
        "target_model_eligible": True,
        "stance_label": "negative",
        "stance_polarity": -1.0,
        "stance_confidence": 0.8,
        "text_sha256": "a" * 64,
    }
    row.update(overrides)
    return row


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _write_registry(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "election_id": "pres_2025",
                "candidate_id": "candidate_x_id",
                "candidate_name": "candidate_x",
                "party_name": "party_x",
                "ballot_number": "1",
                "available_date": "2025-05-12",
                "source_url": "https://example.invalid/official",
                "source_type": "official_fixture",
            },
            {
                "election_id": "pres_2025",
                "candidate_id": "candidate_y_id",
                "candidate_name": "candidate_y",
                "party_name": "party_y",
                "ballot_number": "2",
                "available_date": "2025-05-12",
                "source_url": "https://example.invalid/official",
                "source_type": "official_fixture",
            },
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_2025_is_forecast_only_and_not_a_weight_selection_election() -> None:
    assert FORECAST_ONLY_ELECTIONS == ("pres_2025",)
    assert "pres_2025" not in SCORED_ELECTIONS
    assert "pres_2025" not in issue_vote_engine.ORDER
    assert "pres_2025" not in issue_vote_engine.WEIGHT_SELECTION_ELECTIONS
    assert forecast_cutoff("pres_2025", ELECTION_DATES) == pd.Timestamp("2025-06-02")


def test_2025_point_in_time_filter_excludes_election_day() -> None:
    frame = pd.DataFrame(
        [
            {"election_id": "pres_2025", "available_date": "2025-06-02", "value": "keep"},
            {"election_id": "pres_2025", "available_date": "2025-06-03", "value": "drop"},
        ]
    )
    out = filter_available_by_election(frame, ELECTION_DATES, source_name="2025 fixture")
    assert out["value"].tolist() == ["keep"]


def test_2025_context_builder_keeps_all_pre_cutoff_rows_only(tmp_path: Path) -> None:
    source = tmp_path / "corpus.csv"
    output = tmp_path / "output"
    registry = tmp_path / "candidate_registry.csv"
    _write_registry(registry)
    _write_source(
        source,
        [
            _row(),
            _row(text_sha256="b" * 64, meeting_date="2025-06-03", available_date="2025-06-03"),
            _row(election_id="pres_2022", text_sha256="c" * 64),
        ],
    )

    manifest = build_context(source, output, registry)
    salience = pd.read_csv(output / "issue_salience_weekly.csv", encoding="utf-8-sig")
    targets = pd.read_csv(
        output / "explicit_target_context_weekly.csv", encoding="utf-8-sig"
    )
    candidate_targets = pd.read_csv(
        output / "candidate_target_context_weekly.csv", encoding="utf-8-sig"
    )
    model_salience = pd.read_csv(
        output / "model_issue_salience.csv", encoding="utf-8-sig"
    )
    model_link = pd.read_csv(
        output / "model_candidate_issue_link.csv", encoding="utf-8-sig"
    )

    assert manifest["target_rows_seen"] == 2
    assert manifest["target_rows_included"] == 1
    assert manifest["post_cutoff_rows_excluded"] == 1
    assert manifest["pres_2025_outcome_used"] is False
    assert manifest["source_path"] == "external://corpus.csv"
    assert manifest["candidate_registry_path"] == "external://candidate_registry.csv"
    assert salience["sentence_count"].sum() == 1
    assert pd.to_datetime(salience["available_date"]).max() == pd.Timestamp("2025-06-02")
    assert targets["target_model_eligible"].tolist() == [True]
    assert manifest["candidate_registry_rows"] == 2
    assert candidate_targets["candidate_id"].tolist() == ["candidate_x_id"]
    assert candidate_targets["candidate_link_eligible"].tolist() == [True]
    assert candidate_targets["source_observed_available_date"].tolist() == ["2025-06-02"]
    assert candidate_targets["available_date"].tolist() == ["2025-06-02"]
    assert model_salience["salience_score"].tolist() == [1.0]
    assert model_salience["instrument"].tolist() == ["assembly_speech_forecast_only"]
    assert model_link["candidate_id"].tolist() == ["candidate_x_id"]
    assert model_link["emphasis_within"].tolist() == [1.0]


def test_candidate_link_waits_for_roster_availability(tmp_path: Path) -> None:
    source = tmp_path / "corpus.csv"
    output = tmp_path / "output"
    registry = tmp_path / "candidate_registry.csv"
    _write_registry(registry)
    _write_source(
        source,
        [_row(meeting_date="2024-11-01", available_date="2024-11-01")],
    )

    build_context(source, output, registry)
    candidate_targets = pd.read_csv(
        output / "candidate_target_context_weekly.csv", encoding="utf-8-sig"
    )

    assert candidate_targets["source_observed_available_date"].tolist() == ["2024-11-01"]
    assert candidate_targets["candidate_registry_available_date"].tolist() == ["2025-05-12"]
    assert candidate_targets["available_date"].tolist() == ["2025-05-12"]

    salience, candidate_link, manifest = load_forecast_only_assembly_inputs(
        context_dir=output
    )
    assert manifest["target_election"] == "pres_2025"
    assert len(salience) == 1
    assert candidate_link["candidate_id"].tolist() == ["candidate_x_id"]

    slots = pd.DataFrame(
        [
            {
                "election_id": "pres_2025",
                "candidate_id": "candidate_x_id",
                "slot": "A",
                "available_date": "2025-05-12",
            }
        ]
    )
    attached = attach_preliminary_slots(candidate_link, slots)
    assert attached["slot"].tolist() == ["A"]
