from __future__ import annotations

import pandas as pd

from presidential_issue_engine.build_assembly_speaker_influence import build_speaker_influence


def test_build_speaker_influence_separates_mandate_and_signal_weights() -> None:
    matches = pd.DataFrame(
        [
            {
                "source_sheet": "plenary",
                "source_file": "fixture.xlsx",
                "meeting_date": "1996-07-18",
                "period": "1996-07-15",
                "committee": "plenary",
                "agenda": "economic question",
                "speaker": "District Member",
                "member_id": "101",
                "issue_name": "regional_dev",
                "issue_weight": "1.0",
                "matched_term_count": "2",
                "text_length": "100",
            },
            {
                "source_sheet": "plenary",
                "source_file": "fixture.xlsx",
                "meeting_date": "1996-07-18",
                "period": "1996-07-15",
                "committee": "plenary",
                "agenda": "national list speech",
                "speaker": "List Member",
                "member_id": "202",
                "issue_name": "regional_dev",
                "issue_weight": "1.0",
                "matched_term_count": "1",
                "text_length": "100",
            },
            {
                "source_sheet": "plenary",
                "source_file": "fixture.xlsx",
                "meeting_date": "1996-07-18",
                "period": "1996-07-15",
                "committee": "cabinet",
                "agenda": "minister report",
                "speaker": "Industry Minister",
                "member_id": "",
                "issue_name": "foreign_policy",
                "issue_weight": "1.0",
                "matched_term_count": "1",
                "text_length": "100",
            },
        ]
    )
    roster15 = pd.DataFrame(
        [
            {
                "daesu": "15",
                "name": "District Member",
                "term_member_id": "101",
                "party": "Party A",
                "bloc": "bloc_a",
                "district": "Seoul district",
                "member_id": "101",
            },
            {
                "daesu": "15",
                "name": "List Member",
                "term_member_id": "202",
                "party": "Party A",
                "bloc": "bloc_a",
                "district": "proportional list",
                "member_id": "202",
            },
        ]
    )

    profile, issue_summary, scope, conversion, diagnostics = build_speaker_influence(
        matches,
        roster15,
        pd.DataFrame(),
    )

    assert len(profile) == 3
    mandate_by_speaker = dict(zip(profile["speaker_clean"], profile["mandate_type"]))
    assert mandate_by_speaker["District Member"] == "district"
    assert mandate_by_speaker["List Member"] == "proportional"
    assert mandate_by_speaker["Industry Minister"] == "government"

    regional = scope.loc[scope["issue_name"] == "regional_dev"].iloc[0]
    assert regional["national_weight"] > 0
    assert regional["local_weight"] > 0
    assert set(conversion["election_id"]) == {"pres_2002"}
    assert {
        "issue_name",
        "weighted_total",
        "national_share",
        "local_share",
        "unique_speakers",
    }.issubset(issue_summary.columns)
    assert diagnostics.loc[diagnostics["metric"] == "match_rows", "value"].iloc[0] == 3


def test_seniority_uses_only_terms_known_at_speech_assembly() -> None:
    matches = pd.DataFrame(
        [
            {
                "assembly_daesu": "15",
                "meeting_date": "1996-07-18",
                "speaker": "Past Member",
                "member_id": "101",
                "committee": "plenary",
                "agenda": "economy",
                "source_sheet": "plenary",
                "issue_name": "economy_growth",
                "issue_weight": "1.0",
                "matched_term_count": "1",
            }
        ]
    )
    roster15 = pd.DataFrame(
        [
            {
                "daesu": "15",
                "name": "Past Member",
                "term_member_id": "101",
                "party": "Party A",
                "bloc": "bloc_a",
                "district": "Seoul district",
                "member_id": "101",
            }
        ]
    )
    roster_all = pd.DataFrame(
        [
            {"daesu": "15", "name": "Past Member", "party": "Party A", "bloc": "bloc_a"},
            {"daesu": "16", "name": "Past Member", "party": "Party A", "bloc": "bloc_a"},
            {"daesu": "17", "name": "Past Member", "party": "Party A", "bloc": "bloc_a"},
        ]
    )

    profile, *_ = build_speaker_influence(matches, roster15, roster_all)

    assert profile.loc[0, "term_count"] == 1
    assert profile.loc[0, "seniority_weight"] == 1.0
