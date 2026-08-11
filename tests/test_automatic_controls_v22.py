from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.automatic_controls_v22 import (
    build_automatic_generation_weights,
    build_automatic_mega_taxonomy,
    build_automatic_responsibility_alignments,
    build_behavioral_party_transitions,
)
from presidential_issue_engine.regional_policy_commitment import (
    compile_policy_alignment,
)


DATES = {"pres_test": "2020-01-10"}


def test_policy_commitment_strength_is_derived_and_future_rows_are_excluded() -> None:
    registry = pd.DataFrame(
        [
            {
                "event_id": "pledge_before",
                "election_id": "pres_test",
                "candidate_name": "Candidate A",
                "issue_name": "regional_dev",
                "region_scope": "test_region",
                "event_date": "2020-01-01",
                "available_date": "2020-01-01",
                "source_type": "official_manifesto",
                "source_url": "https://example.invalid/before",
            },
            {
                "event_id": "pledge_after",
                "election_id": "pres_test",
                "candidate_name": "Candidate A",
                "issue_name": "regional_dev",
                "region_scope": "future_region",
                "event_date": "2020-01-11",
                "available_date": "2020-01-11",
                "source_type": "official_manifesto",
                "source_url": "https://example.invalid/after",
            },
        ]
    )
    profile = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "candidate_name": "Candidate A",
                "issue_name": "regional_dev",
                "association_strength": 0.64,
                "available_date": "2020-01-02",
            }
        ]
    )
    importance = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "issue_name": "regional_dev",
                "importance_multiplier": 0.81,
                "confidence": 0.81,
                "available_date": "2020-01-03",
            }
        ]
    )
    output, audit = compile_policy_alignment(registry, profile, importance, DATES)
    assert output["region_scope"].tolist() == ["test_region"]
    assert output.loc[0, "affinity"] == pytest.approx(0.8)
    assert output.loc[0, "confidence"] == pytest.approx(0.9)
    assert "strength" not in registry.columns
    assert not audit["target_outcome_used"].any()


def test_mega_taxonomy_uses_universal_class_thresholds() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "election_id": "pres_crisis",
                "source_rows": 1000,
                "salience_component": 0.9,
                "severity_component": 0.8,
                "breadth_component": 0.8,
                "accountability_component": 0.9,
                "joint_evidence": 0.8,
                "available_date": "2020-01-01",
            },
            {
                "election_id": "pres_diffuse",
                "source_rows": 1000,
                "salience_component": 0.1,
                "severity_component": 0.1,
                "breadth_component": 0.2,
                "accountability_component": 0.1,
                "joint_evidence": 0.1,
                "available_date": "2020-01-01",
            },
        ]
    )
    taxonomy, intensity, audit = build_automatic_mega_taxonomy(diagnostics)
    classes = taxonomy.set_index("election_id")["shock_type"]
    scores = intensity.set_index("election_id")["mega_issue_intensity"]
    assert classes["pres_crisis"] == "institutional_crisis"
    assert classes["pres_diffuse"] == "diffuse_issue_environment"
    assert scores["pres_crisis"] == pytest.approx(2.0)
    assert scores["pres_diffuse"] == pytest.approx(0.5)
    assert not audit["target_outcome_used"].any()


def test_responsibility_combines_incumbency_and_dated_discourse() -> None:
    profile = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "slot": "B",
                "issue_name": "economy_growth",
                "direction": -1.0,
                "association_strength": 0.8,
                "confidence": 0.8,
                "target_absolute_evidence": 3.0,
                "target_attribution_confidence": 0.75,
                "target_source_types": "government",
                "available_date": "2020-01-05",
            },
            {
                "election_id": "pres_test",
                "slot": "B",
                "issue_name": "housing",
                "direction": -1.0,
                "association_strength": 0.8,
                "confidence": 0.8,
                "target_absolute_evidence": 9.0,
                "target_attribution_confidence": 1.0,
                "target_source_types": "government",
                "available_date": "2020-01-11",
            },
        ]
    )
    context = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "slot": "A",
                "organization_strength": 0.8,
                "available_date": "2020-01-06",
            },
            {
                "election_id": "pres_test",
                "slot": "B",
                "organization_strength": 1.0,
                "available_date": "2020-01-06",
            },
        ]
    )
    economic, housing, audit = build_automatic_responsibility_alignments(
        profile, context, DATES
    )
    scores = economic.set_index("slot")["economic_responsibility_score"]
    assert scores["B"] > 0.0
    assert scores["A"] < 0.0
    housing_scores = housing.set_index("slot")["housing_responsibility_score"]
    assert housing_scores["B"] == pytest.approx(0.75)
    assert housing_scores["A"] == pytest.approx(-0.6)
    assert not audit["target_outcome_used"].any()


def test_generation_weights_use_only_a_strictly_prior_official_report() -> None:
    history = pd.DataFrame(
        [
            {
                "source_election_id": "pres_prior",
                "event_date": "2018-01-01",
                "published_date": "2018-02-01",
                "young_weight": 0.2,
                "middle_weight": 0.5,
                "senior_weight": 0.3,
            },
            {
                "source_election_id": "pres_target",
                "event_date": "2020-01-10",
                "published_date": "2020-02-01",
                "young_weight": 0.9,
                "middle_weight": 0.05,
                "senior_weight": 0.05,
            },
        ]
    )
    output, audit = build_automatic_generation_weights(history, DATES)
    assert output.loc[0, "young_weight"] == pytest.approx(0.2)
    assert audit.loc[0, "source_election_id"] == "pres_prior"
    assert not audit["target_outcome_used"].any()


def test_party_retention_update_uses_first_later_direct_party_ballot() -> None:
    transitions = pd.DataFrame(
        [
            {
                "predecessor_party": "Old Party",
                "successor_party": "New Party",
                "relation_type": "merge",
                "effective_date": "2015-01-01",
                "continuity": 1.0,
                "confidence": 1.0,
                "notes": "factual transition",
            }
        ]
    )
    rows = []
    for region, before, after in [("r1", 0.4, 0.3), ("r2", 0.2, 0.15), ("r3", 0.1, 0.075)]:
        rows.extend(
            [
                {
                    "source_party_names": "Old Party",
                    "election_type": "assembly_pr",
                    "event_date": "2014-01-01",
                    "region_id": region,
                    "regional_share": before,
                },
                {
                    "source_party_names": "New Party",
                    "election_type": "assembly_pr",
                    "event_date": "2016-01-01",
                    "region_id": region,
                    "regional_share": after,
                },
            ]
        )
    output, audit = build_behavioral_party_transitions(
        transitions, pd.DataFrame(rows)
    )
    update = output.loc[output["effective_date"].eq("2016-01-01")].iloc[0]
    assert update["continuity"] == pytest.approx(0.75**0.5)
    assert audit.iloc[-1]["status"] == "behavioral_update"
    assert not audit["target_outcome_used"].any()
