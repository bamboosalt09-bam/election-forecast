from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine
from presidential_issue_engine.automatic_controls_v22 import (
    build_automatic_generation_weights,
)
from presidential_issue_engine.automatic_withdrawal_v23 import (
    build_unified_candidate_profiles,
    compile_withdrawal_transfer_registry,
)
from scripts import build_preliminary_slot_assignments as preliminary


DATES = {
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}


def _aliases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "ahn",
                "canonical_name": "Ahn Cheol-soo",
                "alias": "Ahn Cheol-soo",
            },
            {
                "candidate_id": "ahn",
                "canonical_name": "Ahn Cheol-soo",
                "alias": "안철수",
            },
        ]
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "withdrawal",
                "election_id": "pres_2022",
                "candidate_id": "ahn",
                "candidate_name": "Ahn Cheol-soo",
                "source_slot": "C",
                "target_slot": "A",
                "event_type": "coalition_withdrawal",
                "event_timestamp": "2022-03-03T00:00:00+09:00",
                "available_date": "2022-03-03",
                "formal_endorsement": True,
                "source_viability_after_event": 0.0,
                "exclude_source_from_evaluation": True,
            }
        ]
    )


def test_prior_same_person_traits_are_shrunk_toward_neutral() -> None:
    active = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "slot": "C",
                "candidate_name": "Ahn Cheol-soo",
                "viability": 0.55,
                "centrist_appeal": 0.99,
                "anti_major_party_appeal": 0.99,
                "regional_base_overlap": 0.35,
                "available_date": "2022-02-01",
                "confidence": 0.70,
                "notes": "manual values must not survive",
            }
        ]
    )
    derived = pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "C",
                "candidate_name": "안철수",
                "viability": 0.90,
                "centrist_appeal": 0.83,
                "anti_major_party_appeal": 0.73,
                "regional_base_overlap": 0.37,
                "available_date": "2017-05-08",
                "confidence": 0.65,
            }
        ]
    )
    preliminary = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "slot": "C",
                "candidate_name": "Ahn Cheol-soo",
                "viability": 0.52,
                "regional_base_overlap": 0.27,
                "available_date": "2022-03-03",
                "confidence": 0.75,
            }
        ]
    )
    profiles, audit = build_unified_candidate_profiles(
        active,
        derived,
        preliminary,
        pd.DataFrame(),
        _events(),
        _aliases(),
        DATES,
    )
    row = profiles.iloc[0]
    assert row["viability"] == pytest.approx(0.52)
    assert 0.50 < row["centrist_appeal"] < 0.83
    assert 0.50 < row["anti_major_party_appeal"] < 0.73
    assert row["centrist_appeal"] != pytest.approx(0.99)
    assert audit.iloc[0]["profile_source"].endswith("_shrunk")
    assert not audit["target_outcome_used"].any()


def test_no_evidence_candidate_uses_common_low_confidence_fallback() -> None:
    active = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "slot": "C",
                "candidate_name": "Ahn Cheol-soo",
                "viability": 0.95,
                "centrist_appeal": 0.95,
                "anti_major_party_appeal": 0.95,
                "regional_base_overlap": 0.95,
                "available_date": "2022-02-01",
                "confidence": 0.95,
                "notes": "must be replaced",
            }
        ]
    )
    profiles, _ = build_unified_candidate_profiles(
        active,
        pd.DataFrame(
            columns=[
                "election_id",
                "slot",
                "candidate_name",
                "available_date",
                "confidence",
                "viability",
                "centrist_appeal",
                "anti_major_party_appeal",
                "regional_base_overlap",
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
        _events(),
        _aliases(),
        DATES,
    )
    row = profiles.iloc[0]
    assert row["viability"] == pytest.approx(0.5)
    assert row["centrist_appeal"] == pytest.approx(0.5)
    assert row["anti_major_party_appeal"] == pytest.approx(0.5)
    assert row["regional_base_overlap"] == pytest.approx(0.0)
    assert row["confidence"] == pytest.approx(0.25)


def test_transfer_registry_uses_common_scenario_and_event_facts_only() -> None:
    profiles = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "slot": "C",
                "candidate_id": "ahn",
                "candidate_name": "Ahn Cheol-soo",
                "candidate_role": "withdrawn_preliminary",
                "viability": 0.52,
                "centrist_appeal": 0.64,
                "anti_major_party_appeal": 0.60,
                "regional_base_overlap": 0.27,
                "available_date": "2022-03-03",
                "confidence": 0.57,
            }
        ]
    )
    registry, audit = compile_withdrawal_transfer_registry(_events(), profiles, DATES)
    assert set(registry["target_slot"]) == {"A", "B"}
    assert set(registry["scenario_level"]) == {"medium"}
    assert registry["preliminary_transfer_rate"].sum() == pytest.approx(
        0.55 * 0.70 + 0.25 * 0.55
    )
    assert registry.loc[registry["target_slot"].eq("A"), "is_official_target"].all()
    assert registry.loc[
        registry["target_slot"].eq("A"), "use_in_coalition_layer"
    ].all()
    assert registry["use_in_withdrawn_feature_layer"].all()
    assert registry["use_in_preliminary_layer"].all()
    assert not registry["target_outcome_used"].any()
    assert not audit["target_outcome_used"].any()


def test_single_registry_bypasses_all_legacy_transfer_inputs(tmp_path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {
                "event_id": "event",
                "election_id": "pres_2022",
                "candidate_name": "Ahn Cheol-soo",
                "source_slot": "C",
                "target_slot": "A",
                "event_type": "coalition_withdrawal",
                "available_date": "2022-03-03",
                "is_official_target": True,
                "coalition_transfer_rate": 0.60,
                "transfer_rate": 0.42,
                "voter_compliance": 0.70,
                "viability": 0.52,
                "confidence": 0.75,
                "source_viability_after_event": 0.0,
                "exclude_source_from_evaluation": True,
            }
        ]
    ).to_csv(registry_path, index=False)
    missing = tmp_path / "must_not_be_read.csv"
    monkeypatch.setattr(issue_vote_engine, "WITHDRAWAL_TRANSFER_REGISTRY", str(registry_path))
    monkeypatch.setattr(issue_vote_engine, "COALITION_EVENTS", str(missing))
    monkeypatch.setattr(issue_vote_engine, "WITHDRAWN_CANDIDATE_TRANSFERS", str(missing))
    monkeypatch.setattr(issue_vote_engine, "CANDIDATE_POLITICAL_LANDSCAPE", str(missing))

    events = issue_vote_engine._load_coalition_events()
    transfers = issue_vote_engine._load_withdrawn_candidate_transfers()
    assert events.iloc[0]["transfer_rate"] == pytest.approx(0.60)
    assert transfers.iloc[0]["withdrawn_transfer_rate"] == pytest.approx(0.42)


def test_prior_assembly_history_can_profile_withdrawn_candidate() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "withdrawal",
                "election_id": "pres_2022",
                "candidate_id": "ahn",
                "candidate_name": "Ahn Cheol-soo",
                "source_slot": "C",
                "target_slot": "A",
                "event_type": "coalition_withdrawal",
                "event_timestamp": "2022-03-03T00:00:00+09:00",
                "available_date": "2022-03-03",
                "formal_endorsement": True,
                "source_viability_after_event": 0.0,
                "exclude_source_from_evaluation": True,
            }
        ]
    )
    assembly = pd.DataFrame(
        [
            {
                "candidate_name": "Ahn Cheol-soo",
                "available_date": "2020-04-16",
                "vote_share": 0.65,
                "won": True,
                "party_name": "Independent",
                "region_id": "sido_11",
            },
            {
                "candidate_name": "Ahn Cheol-soo",
                "available_date": "2016-04-14",
                "vote_share": 0.60,
                "won": True,
                "party_name": "Independent",
                "region_id": "sido_11",
            },
        ]
    )
    profiles, audit = build_unified_candidate_profiles(
        pd.DataFrame(columns=["election_id", "slot", "candidate_name"]),
        pd.DataFrame(
            columns=[
                "election_id",
                "slot",
                "candidate_name",
                "available_date",
                "confidence",
                "viability",
                "centrist_appeal",
                "anti_major_party_appeal",
                "regional_base_overlap",
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
        events,
        _aliases(),
        DATES,
        assembly,
    )
    row = profiles.iloc[0]
    assert row["viability"] > 0.5
    assert row["centrist_appeal"] > 0.5
    assert row["regional_base_overlap"] > 0.0
    assert audit.iloc[0]["profile_source"].endswith("_shrunk")


def test_pre_withdrawal_attention_estimates_stature_without_direction() -> None:
    attention = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "candidate_id": "major_a",
                "unique_sentence_count": 200,
                "last_evidence_date": "2022-03-02",
            },
            {
                "election_id": "pres_2022",
                "candidate_id": "ahn",
                "unique_sentence_count": 128,
                "last_evidence_date": "2022-03-02",
            },
        ]
    )
    profiles, audit = build_unified_candidate_profiles(
        pd.DataFrame(columns=["election_id", "slot", "candidate_name"]),
        pd.DataFrame(
            columns=[
                "election_id",
                "slot",
                "candidate_name",
                "available_date",
                "confidence",
                "viability",
                "centrist_appeal",
                "anti_major_party_appeal",
                "regional_base_overlap",
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
        _events(),
        _aliases(),
        DATES,
        pd.DataFrame(),
        attention,
    )
    row = profiles.iloc[0]
    assert row["viability"] > 0.5
    assert row["centrist_appeal"] == pytest.approx(0.5)
    assert row["anti_major_party_appeal"] == pytest.approx(0.5)
    assert row["regional_base_overlap"] == pytest.approx(0.0)
    assert "attention_stature" in audit.iloc[0]["profile_source"]


def test_real_generation_history_uses_2017_report_for_2022() -> None:
    history = pd.read_csv(
        "data/raw/official_sources/nec_age_turnout_composition_history.csv"
    )
    output, audit = build_automatic_generation_weights(
        history, {"pres_2022": "2022-03-09"}
    )
    row = output.iloc[0]
    assert row["young_weight"] == pytest.approx(0.173)
    assert row["middle_weight"] == pytest.approx(0.368)
    assert row["senior_weight"] == pytest.approx(0.459)
    assert audit.iloc[0]["source_election_id"] == "pres_2017"
    assert not audit["target_outcome_used"].any()


def test_preliminary_registry_path_does_not_read_legacy_transfer_values(
    tmp_path, monkeypatch
) -> None:
    results = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "slot": ["A"],
            "is_active_slot": [True],
        }
    ).to_csv(results, index=False)
    legacy_coalition = tmp_path / "legacy_coalition.csv"
    legacy_withdrawn = tmp_path / "legacy_withdrawn.csv"
    pd.DataFrame({"forbidden": [1]}).to_csv(legacy_coalition, index=False)
    pd.DataFrame({"forbidden": [1]}).to_csv(legacy_withdrawn, index=False)
    registry = tmp_path / "registry.csv"
    pd.DataFrame({"event_id": ["event"]}).to_csv(registry, index=False)

    monkeypatch.setattr(preliminary.engine, "RESULTS", str(results))
    monkeypatch.setattr(preliminary.engine, "COALITION_EVENTS", str(legacy_coalition))
    monkeypatch.setattr(
        preliminary.engine, "WITHDRAWN_CANDIDATE_TRANSFERS", str(legacy_withdrawn)
    )
    monkeypatch.setattr(
        preliminary.engine, "WITHDRAWAL_TRANSFER_REGISTRY", str(registry)
    )

    observed: dict[str, str] = {}

    def fake_assemble() -> pd.DataFrame:
        observed["coalition"] = preliminary.engine.COALITION_EVENTS
        observed["withdrawn"] = preliminary.engine.WITHDRAWN_CANDIDATE_TRANSFERS
        observed["registry"] = preliminary.engine.WITHDRAWAL_TRANSFER_REGISTRY
        return pd.DataFrame(
            {
                "election_id": ["pres_2022"],
                "region_id": ["seoul"],
                "slot": ["A"],
            }
        )

    monkeypatch.setattr(preliminary.engine, "assemble", fake_assemble)
    preliminary._all_ballot_rows(pd.DataFrame())

    assert observed["registry"] == ""
    assert observed["coalition"] != str(legacy_coalition)
    assert observed["withdrawn"] != str(legacy_withdrawn)
    assert preliminary.engine.COALITION_EVENTS == str(legacy_coalition)
    assert preliminary.engine.WITHDRAWN_CANDIDATE_TRANSFERS == str(legacy_withdrawn)
