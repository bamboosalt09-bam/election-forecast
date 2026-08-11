from __future__ import annotations

import pandas as pd

from presidential_issue_engine.preliminary_candidate_registry import (
    build_preliminary_candidate_registry,
    derive_prior_candidate_profile,
    merge_preliminary_profile,
)


def _registry_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.DataFrame(
        [
            {
                "event_id": "withdraw_2022",
                "election_id": "pres_2022",
                "event_date": "2022-03-03",
                "available_date": "2022-03-03",
                "event_type": "coalition_withdrawal",
                "source_slot": "C",
                "target_slot": "A",
            }
        ]
    )
    transfers = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "candidate_name": "Ahn Cheol-soo",
                "target_slot": "A",
            }
        ]
    )
    landscape = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "slot": "C",
                "candidate_name": "Ahn Cheol-soo",
            }
        ]
    )
    return events, transfers, landscape


def _profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "C",
                "candidate_name": "Ahn Cheol-soo",
                "viability": 0.9,
                "centrist_appeal": 0.8,
                "anti_major_party_appeal": 0.7,
                "regional_base_overlap": 0.4,
                "available_date": "2017-05-08",
                "confidence": 0.75,
            },
            {
                "election_id": "pres_2022",
                "slot": "C",
                "candidate_name": "Ahn Cheol-soo",
                "viability": 0.55,
                "centrist_appeal": 0.8,
                "anti_major_party_appeal": 0.7,
                "regional_base_overlap": 0.35,
                "available_date": "2022-02-01",
                "confidence": 0.7,
            },
        ]
    )


def _results(target_votes: float = 999999.0) -> pd.DataFrame:
    rows = []
    for region, a, b, c in [("r1", 45, 35, 20), ("r2", 40, 30, 30)]:
        for election_id, slot, votes in [
            ("pres_2017", "A", a),
            ("pres_2017", "B", b),
            ("pres_2017", "C", c),
            ("pres_2022", "A", target_votes),
            ("pres_2022", "B", target_votes),
        ]:
            rows.append(
                {
                    "election_id": election_id,
                    "region_id": region,
                    "slot": slot,
                    "votes": votes,
                    "vote_share": votes / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_target_election_outcome_does_not_change_preliminary_profile() -> None:
    registry = build_preliminary_candidate_registry(*_registry_inputs())
    before, _ = derive_prior_candidate_profile(registry, _profiles(), _results(1.0))
    after, _ = derive_prior_candidate_profile(registry, _profiles(), _results(1e9))
    pd.testing.assert_frame_equal(before, after)


def test_prior_profile_is_used_and_target_row_is_not() -> None:
    registry = build_preliminary_candidate_registry(*_registry_inputs())
    derived, audit = derive_prior_candidate_profile(registry, _profiles(), _results())
    assert len(derived) == 1
    assert derived.iloc[0]["prior_election_id"] == "pres_2017"
    assert derived.iloc[0]["viability"] < 1.0
    assert bool(audit.iloc[0]["prior_evidence_available"])


def test_merge_replaces_only_matching_preliminary_candidate() -> None:
    registry = build_preliminary_candidate_registry(*_registry_inputs())
    derived, _ = derive_prior_candidate_profile(registry, _profiles(), _results())
    merged, audit = merge_preliminary_profile(_profiles(), derived)
    row_2017 = merged.loc[merged["election_id"].eq("pres_2017")].iloc[0]
    row_2022 = merged.loc[merged["election_id"].eq("pres_2022")].iloc[0]
    assert row_2017["viability"] == 0.9
    assert row_2022["viability"] == derived.iloc[0]["viability"]
    assert len(audit) == 1

