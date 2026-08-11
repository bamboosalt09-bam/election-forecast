from __future__ import annotations

import pandas as pd

from presidential_issue_engine.regional_party_channels import (
    GENERIC_FALLBACK_SCALE,
    build_lineage_corroborated_identity_events,
    build_two_channel_identity_events,
)
from presidential_issue_engine.chungcheong_identity import fit_identity_profiles
from presidential_issue_engine.automatic_regional_party_alignment import (
    build_full_history_identity_events,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "assembly_1996_district",
                "election_type": "assembly_district",
                "region_id": "sido_44",
                "bloc": "제3지대",
                "vote_share": 0.40,
                "data_quality_weight": 0.65,
            },
            {
                "election_id": "assembly_2004_pr",
                "election_type": "assembly_pr",
                "region_id": "sido_44",
                "bloc": "자민련",
                "vote_share": 0.24,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "assembly_2004_pr",
                "election_type": "assembly_pr",
                "region_id": "sido_11",
                "bloc": "자민련",
                "vote_share": 0.02,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "metro_council_2006_pr",
                "election_type": "metro_council_pr",
                "region_id": "sido_44",
                "bloc": "제3지대",
                "vote_share": 0.30,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "metro_council_2006_pr",
                "election_type": "metro_council_pr",
                "region_id": "sido_11",
                "bloc": "제3지대",
                "vote_share": 0.10,
                "data_quality_weight": 1.0,
            },
        ]
    )


def _assembly() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "assembly_1996_district",
                "election_date": "1996-04-11",
                "region_id": "sido_44",
                "district_name": "A",
                "party_name": "자유민주연합",
                "candidate_votes": 400,
                "district_valid_votes": 1000,
            },
            {
                "election_id": "assembly_1996_district",
                "election_date": "1996-04-11",
                "region_id": "sido_44",
                "district_name": "A",
                "party_name": "신한국당",
                "candidate_votes": 600,
                "district_valid_votes": 1000,
            },
            {
                "election_id": "assembly_1996_district",
                "election_date": "1996-04-11",
                "region_id": "sido_11",
                "district_name": "B",
                "party_name": "자유민주연합",
                "candidate_votes": 50,
                "district_valid_votes": 1000,
            },
            {
                "election_id": "assembly_1996_district",
                "election_date": "1996-04-11",
                "region_id": "sido_11",
                "district_name": "B",
                "party_name": "신한국당",
                "candidate_votes": 950,
                "district_valid_votes": 1000,
            },
        ]
    )


def test_restores_liberal_democrats_as_lineage_specific_organization() -> None:
    events = build_two_channel_identity_events(_history(), _assembly())
    row = events.loc[
        events["election_id"].eq("assembly_1996_district")
        & events["region_id"].eq("sido_44")
    ].iloc[0]
    assert row["evidence_channel"] == "district_organization"
    assert bool(row["lineage_specific"])
    assert row["identity_share"] == 0.40
    assert row["source_detail"] == "nec_constituency_party_lineage"


def test_direct_party_and_generic_fallback_have_distinct_reliability() -> None:
    events = build_two_channel_identity_events(_history(), _assembly())
    named = events.loc[
        events["election_id"].eq("assembly_2004_pr")
        & events["region_id"].eq("sido_44")
    ].iloc[0]
    generic = events.loc[
        events["election_id"].eq("metro_council_2006_pr")
        & events["region_id"].eq("sido_44")
    ].iloc[0]
    assert named["evidence_channel"] == "direct_party_preference"
    assert named["fallback_scale"] == 1.0
    assert generic["fallback_scale"] == GENERIC_FALLBACK_SCALE
    assert generic["identity_excess"] == generic["identity_excess_raw"] * GENERIC_FALLBACK_SCALE


def test_future_election_mutation_does_not_change_prior_profile() -> None:
    history = _history()
    events = build_two_channel_identity_events(history, _assembly())
    before = fit_identity_profiles(events, cutoff=pd.Timestamp("2002-12-19"))
    changed = history.copy()
    changed.loc[
        changed["election_id"].eq("assembly_2004_pr"), "vote_share"
    ] = [0.90, 0.01]
    after_events = build_two_channel_identity_events(changed, _assembly())
    after = fit_identity_profiles(after_events, cutoff=pd.Timestamp("2002-12-19"))
    pd.testing.assert_frame_equal(before, after)


def test_corroboration_preserves_reservoir_and_only_boosts_named_evidence() -> None:
    base = build_full_history_identity_events(_history())
    corroborated = build_lineage_corroborated_identity_events(_history(), _assembly())
    keys = ["election_id", "election_type", "region_id"]
    comparison = base.merge(
        corroborated,
        on=keys,
        suffixes=("_base", "_corroborated"),
        validate="one_to_one",
    )
    pd.testing.assert_series_equal(
        comparison["identity_excess_base"],
        comparison["identity_excess_corroborated"],
        check_names=False,
    )
    named = comparison.loc[
        comparison["election_id"].eq("assembly_1996_district")
        & comparison["region_id"].eq("sido_44")
    ].iloc[0]
    assert named["type_weight_corroborated"] > named["type_weight_base"]
