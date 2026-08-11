from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine.unified_lineage_identity import (
    apply_unified_lineage_routing,
    attach_exact_lineage_prior,
    attach_lineage_projected_prior,
    build_exact_lineage_events,
    estimate_type_reliability,
    fit_lineage_profiles,
    party_genealogy_affinity,
    project_lineage_events_to_bloc_history,
)


def _assembly_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "assembly_1996_district",
                "election_date": "1996-04-11",
                "region_id": "sido_44",
                "party_name": "자유민주연합",
                "candidate_votes": 600,
            },
            {
                "election_id": "assembly_1996_district",
                "election_date": "1996-04-11",
                "region_id": "sido_44",
                "party_name": "신한국당",
                "candidate_votes": 400,
            },
            {
                "election_id": "assembly_1996_district",
                "election_date": "1996-04-11",
                "region_id": "sido_11",
                "party_name": "자유민주연합",
                "candidate_votes": 100,
            },
            {
                "election_id": "assembly_1996_district",
                "election_date": "1996-04-11",
                "region_id": "sido_11",
                "party_name": "신한국당",
                "candidate_votes": 900,
            },
        ]
    )


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "assembly_1996_district",
                "election_type": "assembly_district",
                "region_id": "sido_44",
                "bloc": "제3지대",
                "vote_share": 0.60,
                "data_quality_weight": 0.65,
            },
            {
                "election_id": "assembly_1996_district",
                "election_type": "assembly_district",
                "region_id": "sido_11",
                "bloc": "제3지대",
                "vote_share": 0.10,
                "data_quality_weight": 0.65,
            },
            {
                "election_id": "pres_2002",
                "election_type": "presidential",
                "region_id": "sido_44",
                "bloc": "제3지대",
                "vote_share": 0.90,
                "data_quality_weight": 1.0,
            },
        ]
    )


def test_exact_assembly_party_replaces_collapsed_third_bloc() -> None:
    events = build_exact_lineage_events(_history(), _assembly_rows())
    event = events.loc[events["election_id"].eq("assembly_1996_district")]
    assert "chungcheong_regionalist" in set(event["lineage_id"])
    assert "mainstream_conservative" in set(event["lineage_id"])
    assert "party:제3지대" not in set(event["lineage_id"])
    row = event.loc[
        event["region_id"].eq("sido_44")
        & event["lineage_id"].eq("chungcheong_regionalist")
    ].iloc[0]
    assert row["regional_share"] == 0.60
    centered = event.groupby("lineage_id")["lineage_gap"].mean()
    assert np.allclose(centered.to_numpy(float), 0.0, atol=1e-12)
    assert "자유민주연합" in row["source_party_names"]


def test_target_and_future_outcomes_do_not_change_prior_profile() -> None:
    events = build_exact_lineage_events(_history(), _assembly_rows())
    before = fit_lineage_profiles(events, cutoff=pd.Timestamp("2002-12-19"))
    changed_history = _history()
    changed_history.loc[
        changed_history["election_id"].eq("pres_2002"), "vote_share"
    ] = 0.01
    changed = build_exact_lineage_events(changed_history, _assembly_rows())
    after = fit_lineage_profiles(changed, cutoff=pd.Timestamp("2002-12-19"))
    pd.testing.assert_frame_equal(before.profiles, after.profiles)
    pd.testing.assert_frame_equal(before.type_reliability, after.type_reliability)


def test_candidate_ballot_reliability_uses_only_prior_paired_party_ballots() -> None:
    rows: list[dict[str, object]] = []
    for region, direct_gap, candidate_gap in (
        ("sido_11", -0.20, -0.18),
        ("sido_26", -0.10, -0.08),
        ("sido_30", 0.10, 0.09),
        ("sido_44", 0.20, 0.19),
    ):
        for election_type, gap, channel in (
            ("assembly_pr", direct_gap, "direct_party"),
            ("assembly_district", candidate_gap, "candidate_proxy"),
        ):
            rows.append(
                {
                    "election_id": f"assembly_2004_{'pr' if election_type == 'assembly_pr' else 'district'}",
                    "election_type": election_type,
                    "event_date": pd.Timestamp("2004-04-15"),
                    "region_id": region,
                    "lineage_id": "chungcheong_regionalist",
                    "lineage_gap": gap,
                    "ballot_channel": channel,
                }
            )
    events = pd.DataFrame(rows)
    reliability = estimate_type_reliability(
        events, cutoff=pd.Timestamp("2007-12-19")
    )
    district = reliability.loc[
        reliability["election_type"].eq("assembly_district")
    ].iloc[0]
    assert district["paired_observations"] == 4
    assert district["paired_correlation"] > 0.99
    assert district["type_reliability"] > 0.25


def test_unified_routing_applies_same_formula_to_all_regions_and_conserves_mass() -> None:
    events = build_exact_lineage_events(_history(), _assembly_rows())
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "region_id": region,
                "slot": slot,
                "candidate_name_x": candidate,
                "bloc": bloc,
                "layer_pred": share,
            }
            for region in ("sido_11", "sido_44")
            for slot, candidate, bloc, share in (
                ("A", "후보A", "국민의힘", 0.55),
                ("B", "후보B", "제3지대", 0.45),
            )
        ]
    )
    frame.index = np.arange(100, 100 + len(frame) * 3, 3)
    candidate_parties = pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "slot": "A",
                "candidate_name": "후보A",
                "party_name": "한나라당",
            },
            {
                "election_id": "pres_2002",
                "slot": "B",
                "candidate_name": "후보B",
                "party_name": "자유민주연합",
            },
        ]
    )
    adjusted, audit, _ = apply_unified_lineage_routing(
        frame,
        events,
        pd.DataFrame(),
        pd.DataFrame(),
        candidate_parties,
        prediction_column="layer_pred",
        gain=0.5,
        shift_cap=0.08,
    )
    assert set(audit["region_id"]) == {"sido_11", "sido_44"}
    totals = adjusted.groupby(["election_id", "region_id"])["layer_pred"].sum()
    assert np.allclose(totals.to_numpy(float), 1.0)
    chung = adjusted.loc[
        adjusted["region_id"].eq("sido_44") & adjusted["slot"].eq("B"),
        "layer_pred",
    ].iloc[0]
    seoul = adjusted.loc[
        adjusted["region_id"].eq("sido_11") & adjusted["slot"].eq("B"),
        "layer_pred",
    ].iloc[0]
    assert chung > 0.45
    assert seoul < 0.45


def _collapsed_lineage_history(
    *,
    generic_profile: tuple[float, float, float, float],
    include_future: bool = False,
) -> pd.DataFrame:
    regions = ("sido_11", "sido_30", "sido_43", "sido_44")
    exact_profile = (0.05, 0.35, 0.40, 0.50)
    rows: list[dict[str, object]] = []
    for region, regional_share in zip(regions, exact_profile, strict=True):
        rows.extend(
            [
                {
                    "election_id": "metro_council_1995_pr",
                    "election_type": "metro_council_pr",
                    "region_id": region,
                    "bloc": "\uc790\ubbfc\ub828",
                    "vote_share": regional_share,
                    "data_quality_weight": 1.0,
                },
                {
                    "election_id": "metro_council_1995_pr",
                    "election_type": "metro_council_pr",
                    "region_id": region,
                    "bloc": "\uc2e0\ud55c\uad6d\ub2f9",
                    "vote_share": 1.0 - regional_share,
                    "data_quality_weight": 1.0,
                },
            ]
        )
        rows.extend(
            [
                {
                    "election_id": "assembly_1996_pr",
                    "election_type": "assembly_pr",
                    "region_id": region,
                    "bloc": "\uc790\ubbfc\ub828",
                    "vote_share": regional_share,
                    "data_quality_weight": 1.0,
                },
                {
                    "election_id": "assembly_1996_pr",
                    "election_type": "assembly_pr",
                    "region_id": region,
                    "bloc": "\uc2e0\ud55c\uad6d\ub2f9",
                    "vote_share": 1.0 - regional_share,
                    "data_quality_weight": 1.0,
                },
            ]
        )
    for region, regional_share in zip(regions, generic_profile, strict=True):
        rows.extend(
            [
                {
                    "election_id": "metro_council_1998_pr",
                    "election_type": "metro_council_pr",
                    "region_id": region,
                    "bloc": "\uc81c3\uc9c0\ub300",
                    "vote_share": regional_share,
                    "data_quality_weight": 1.0,
                },
                {
                    "election_id": "metro_council_1998_pr",
                    "election_type": "metro_council_pr",
                    "region_id": region,
                    "bloc": "\uc2e0\ud55c\uad6d\ub2f9",
                    "vote_share": 1.0 - regional_share,
                    "data_quality_weight": 1.0,
                },
            ]
        )
    if include_future:
        for region, regional_share in zip(
            regions, reversed(exact_profile), strict=True
        ):
            rows.append(
                {
                    "election_id": "metro_council_2002_pr",
                    "election_type": "metro_council_pr",
                    "region_id": region,
                    "bloc": "\uc790\ubbfc\ub828",
                    "vote_share": regional_share,
                    "data_quality_weight": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_collapsed_third_is_resolved_from_prior_exact_spatial_lineage() -> None:
    history = _collapsed_lineage_history(
        generic_profile=(0.04, 0.34, 0.42, 0.51)
    )
    events = build_exact_lineage_events(history, _assembly_rows().iloc[0:0])
    later = events.loc[events["election_id"].eq("metro_council_1998_pr")]
    assert "unresolved_third" not in set(later["lineage_id"])
    resolved = later.loc[
        later["lineage_id"].eq("chungcheong_regionalist")
        & later["source_party_names"].str.contains("\uc81c3\uc9c0\ub300")
    ]
    assert not resolved.empty
    assert set(resolved["lineage_resolution"]) == {
        "prior_exact_spatial_profile"
    }
    assert resolved["lineage_resolution_confidence"].gt(0.25).all()


def test_unrelated_collapsed_third_remains_unresolved() -> None:
    history = _collapsed_lineage_history(
        generic_profile=(0.45, 0.05, 0.40, 0.10)
    )
    events = build_exact_lineage_events(history, _assembly_rows().iloc[0:0])
    later = events.loc[events["election_id"].eq("metro_council_1998_pr")]
    assert "unresolved_third" in set(later["lineage_id"])


def test_future_exact_profile_cannot_change_earlier_resolution() -> None:
    base = _collapsed_lineage_history(
        generic_profile=(0.04, 0.34, 0.42, 0.51), include_future=False
    )
    future = _collapsed_lineage_history(
        generic_profile=(0.04, 0.34, 0.42, 0.51), include_future=True
    )
    before = build_exact_lineage_events(base, _assembly_rows().iloc[0:0])
    after = build_exact_lineage_events(future, _assembly_rows().iloc[0:0])
    columns = [
        "region_id",
        "lineage_id",
        "regional_share",
        "lineage_resolution",
        "lineage_resolution_confidence",
    ]
    before_event = before.loc[
        before["election_id"].eq("metro_council_1998_pr"), columns
    ].reset_index(drop=True)
    after_event = after.loc[
        after["election_id"].eq("metro_council_1998_pr"), columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_event, after_event)


def test_exact_lineage_prior_uses_candidate_party_and_prior_events_only() -> None:
    events = build_exact_lineage_events(_history(), _assembly_rows())
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "region_id": region,
                "slot": slot,
                "candidate_name": candidate,
                "bloc": bloc,
            }
            for region in ("sido_11", "sido_44")
            for slot, candidate, bloc in (
                ("A", "candidate_a", "\uad6d\ubbfc\uc758\ud798"),
                ("B", "candidate_b", "\uc81c3\uc9c0\ub300"),
            )
        ]
    )
    frame.index = np.arange(200, 200 + len(frame) * 5, 5)
    candidate_parties = pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "slot": "A",
                "candidate_name": "candidate_a",
                "party_name": "\ud55c\ub098\ub77c\ub2f9",
            },
            {
                "election_id": "pres_2002",
                "slot": "B",
                "candidate_name": "candidate_b",
                "party_name": "\uc790\ubbfc\ub828",
            },
        ]
    )
    attached = attach_exact_lineage_prior(frame, events, candidate_parties)
    assert attached.index.equals(frame.index)
    regionalist = attached.loc[attached["slot"].eq("B")].set_index("region_id")
    assert regionalist.at["sido_44", "exact_lineage_id"] == (
        "chungcheong_regionalist"
    )
    assert regionalist.at["sido_44", "partisan_prior"] > 0.0
    assert regionalist.at["sido_11", "partisan_prior"] < 0.0
    assert regionalist["effective_election_count"].eq(1.0).all()


def test_same_date_exact_district_profile_resolves_collapsed_party_list() -> None:
    regions = ("sido_11", "sido_28", "sido_29", "sido_46")
    third = (0.10, 0.20, 0.55, 0.60)
    history_rows: list[dict[str, object]] = []
    assembly_rows: list[dict[str, object]] = []
    for region, share in zip(regions, third, strict=True):
        history_rows.extend(
            [
                {
                    "election_id": "assembly_2016_pr",
                    "election_type": "assembly_pr",
                    "region_id": region,
                    "bloc": "\uc81c3\uc9c0\ub300",
                    "vote_share": share,
                    "data_quality_weight": 1.0,
                },
                {
                    "election_id": "assembly_2016_pr",
                    "election_type": "assembly_pr",
                    "region_id": region,
                    "bloc": "\uc0c8\ub204\ub9ac\ub2f9",
                    "vote_share": 1.0 - share,
                    "data_quality_weight": 1.0,
                },
            ]
        )
        assembly_rows.extend(
            [
                {
                    "election_id": "assembly_2016_district",
                    "election_date": "2016-04-13",
                    "region_id": region,
                    "party_name": "\uad6d\ubbfc\uc758\ub2f9",
                    "candidate_votes": share * 1000,
                },
                {
                    "election_id": "assembly_2016_district",
                    "election_date": "2016-04-13",
                    "region_id": region,
                    "party_name": "\uc0c8\ub204\ub9ac\ub2f9",
                    "candidate_votes": (1.0 - share) * 1000,
                },
            ]
        )
    events = build_exact_lineage_events(
        pd.DataFrame(history_rows), pd.DataFrame(assembly_rows)
    )
    party_list = events.loc[events["election_id"].eq("assembly_2016_pr")]
    resolved = party_list.loc[
        party_list["lineage_id"].eq("party:\uad6d\ubbfc\uc758\ub2f9")
        & party_list["source_party_names"].str.contains("\uc81c3\uc9c0\ub300")
    ]
    assert not resolved.empty
    assert "unresolved_third" not in set(party_list["lineage_id"])


def test_projected_prior_is_derived_from_lineage_ledger_at_boundary() -> None:
    events = build_exact_lineage_events(_history(), _assembly_rows())
    projected = project_lineage_events_to_bloc_history(events)
    assert set(projected.columns) == {
        "election_id",
        "election_type",
        "region_id",
        "bloc",
        "vote_share",
        "data_quality_weight",
    }
    assert not projected["bloc"].astype(str).str.startswith("party:").any()
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "region_id": region,
                "slot": "A",
                "candidate_name": "candidate_a",
                "bloc": "\uad6d\ubbfc\uc758\ud798",
            }
            for region in ("sido_11", "sido_44")
        ]
    )
    parties = pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "slot": "A",
                "candidate_name": "candidate_a",
                "party_name": "\ud55c\ub098\ub77c\ub2f9",
            }
        ]
    )
    attached = attach_lineage_projected_prior(
        frame, events, parties, ["pres_1997", "pres_2002"]
    )
    assert attached["exact_lineage_id"].eq("mainstream_conservative").all()
    assert attached["partisan_prior"].notna().all()


def test_projection_keeps_observed_quality_when_exact_lineage_is_uncertain() -> None:
    history = pd.DataFrame(
        [
            {
                "election_id": "pres_1997",
                "election_type": "presidential",
                "region_id": region,
                "bloc": bloc,
                "vote_share": share,
                "data_quality_weight": 1.0,
            }
            for region, third_share in (("sido_11", 0.2), ("sido_44", 0.5))
            for bloc, share in (
                ("\uc81c3\uc9c0\ub300", third_share),
                ("\ud55c\ub098\ub77c\ub2f9", 1.0 - third_share),
            )
        ]
    )
    events = build_exact_lineage_events(history, _assembly_rows().iloc[0:0])
    unresolved = events.loc[events["lineage_id"].eq("unresolved_third")]
    assert unresolved["quality"].eq(0.25).all()
    assert unresolved["source_quality"].eq(1.0).all()

    projected = project_lineage_events_to_bloc_history(events)
    broad_third = projected.loc[projected["bloc"].eq("\uc81c3\uc9c0\ub300")]
    assert broad_third["data_quality_weight"].eq(1.0).all()


def test_party_genealogy_uses_only_completed_pre_cutoff_transitions() -> None:
    transitions = pd.DataFrame(
        [
            {
                "predecessor_party": "regional_a",
                "successor_party": "regional_b",
                "effective_date": "2012-05-29",
                "relation_type": "rename",
                "continuity": 1.0,
                "confidence": 1.0,
            },
            {
                "predecessor_party": "regional_b",
                "successor_party": "major_party",
                "effective_date": "2012-11-16",
                "relation_type": "merge",
                "continuity": 1.0,
                "confidence": 1.0,
            },
        ]
    )
    before_merger = party_genealogy_affinity(
        transitions,
        {"regional_a"},
        "major_party",
        cutoff=pd.Timestamp("2012-10-01"),
    )
    after_merger = party_genealogy_affinity(
        transitions,
        {"regional_a"},
        "major_party",
        cutoff=pd.Timestamp("2012-12-19"),
    )
    assert before_merger == 0.0
    assert after_merger == 1.0


def test_genealogy_routing_keeps_exact_party_after_preliminary_slot_reassignment() -> None:
    history = pd.DataFrame(
        [
            {
                "election_id": "assembly_2008_pr",
                "election_type": "assembly_pr",
                "region_id": region,
                "bloc": party,
                "vote_share": share,
                "data_quality_weight": 1.0,
            }
            for region, regional_share in (("sido_11", 0.1), ("sido_44", 0.6))
            for party, share in (
                ("regional_party", regional_share),
                ("other_party", 1.0 - regional_share),
            )
        ]
    )
    events = build_exact_lineage_events(history, _assembly_rows().iloc[0:0])
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_2012",
                "region_id": region,
                "slot": slot,
                "candidate_name": candidate,
                "bloc": bloc,
                "layer_pred": 0.5,
            }
            for region in ("sido_11", "sido_44")
            for slot, candidate, bloc in (
                ("B", "target_candidate", "major_bloc"),
                ("A", "other_candidate", "other_bloc"),
            )
        ]
    )
    candidate_parties = pd.DataFrame(
        [
            {
                "election_id": "pres_2012",
                "slot": "A",
                "candidate_name": "target_candidate",
                "party_name": "major_party",
            },
            {
                "election_id": "pres_2012",
                "slot": "B",
                "candidate_name": "other_candidate",
                "party_name": "other_party",
            },
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "predecessor_party": "regional_party",
                "successor_party": "major_party",
                "effective_date": "2011-10-17",
                "continuity": 1.0,
                "confidence": 1.0,
            }
        ]
    )
    adjusted, audit, _ = apply_unified_lineage_routing(
        frame,
        events,
        pd.DataFrame(),
        pd.DataFrame(),
        candidate_parties,
        transitions,
        prediction_column="layer_pred",
        gain=0.5,
        include_direct_lineage_score=False,
    )
    target = adjusted.loc[
        adjusted["region_id"].eq("sido_44")
        & adjusted["candidate_name"].eq("target_candidate")
    ].iloc[0]
    assert target["layer_pred"] > 0.5
    assert audit.loc[
        audit["region_id"].eq("sido_44"), "maximum_genealogy_affinity"
    ].iloc[0] == 1.0


def test_independent_assembly_votes_do_not_enter_party_terrain() -> None:
    assembly = _assembly_rows()
    independent = assembly.iloc[[0]].copy()
    independent["party_name"] = "\ubb34\uc18c\uc18d"
    independent["candidate_votes"] = 5000
    events = build_exact_lineage_events(
        _history(), pd.concat([assembly, independent], ignore_index=True)
    )
    district = events.loc[
        events["election_id"].eq("assembly_1996_district")
    ]
    assert "independent" not in set(district["lineage_id"])
    totals = district.groupby("region_id")["regional_share"].sum()
    assert np.allclose(totals.to_numpy(float), 1.0)
