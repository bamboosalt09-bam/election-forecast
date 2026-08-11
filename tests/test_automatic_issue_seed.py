from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd

from presidential_issue_engine import issue_vote_engine
from scripts.build_through2022_automatic_issue_seeds import (
    DEFAULT_ELECTIONS,
    SCHEMA_VERSION,
    build_attribution,
    build_candidate_profile,
    build_mega_axis,
    write_outputs,
)


def test_automatic_issue_seed_has_full_scope_and_bounded_signals() -> None:
    profile = build_candidate_profile()
    axis = build_mega_axis()
    attribution = build_attribution(profile, axis)

    expected = set(DEFAULT_ELECTIONS)
    assert set(profile["election_id"]) == expected
    assert set(axis["election_id"]) == expected
    assert set(attribution["election_id"]).issubset(expected)
    assert set(attribution["election_id"])
    assert profile["candidate_name"].notna().all()
    assert profile["direction"].between(-1.0, 1.0).all()
    assert profile["association_strength"].between(0.0, 1.0).all()
    assert profile["confidence"].between(0.0, 1.0).all()
    assert attribution["polarity"].isin([-1.0, 1.0]).all()
    assert "target_directional_balance" in profile.columns
    assert "directional_balance" not in profile.columns


def test_candidate_profile_is_complete_and_does_not_truncate_to_top_four() -> None:
    profile = build_candidate_profile()
    counts = profile.groupby(["election_id", "slot"])["issue_name"].nunique()
    for _, election_counts in counts.groupby(level=0):
        assert election_counts.nunique() == 1
        assert election_counts.iloc[0] > 4


def test_2017_target_attribution_penalizes_incumbent_camp_not_global_speakers() -> None:
    profile = build_candidate_profile(("pres_2017",))
    regime = profile.loc[profile["issue_name"].eq("regime_change")].set_index("slot")
    assert regime.loc["B", "direction"] < 0.0
    assert regime.loc["A", "direction"] == 0.0
    assert regime.loc["C", "direction"] == 0.0


def test_2017_high_evidence_political_shock_survives_mega_axis_selection() -> None:
    axis = build_mega_axis(("pres_2017",))
    assert "corruption_integrity" in set(axis["primary_issue"])


def test_seed_writer_records_input_fingerprints(tmp_path) -> None:
    write_outputs(tmp_path, ("pres_2017",))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["outcome_fields_used"] == []


def test_automatic_issue_seed_respects_election_day_cutoff() -> None:
    frames = [build_candidate_profile(), build_mega_axis()]
    profile = frames[0]
    frames.append(build_attribution(profile, frames[1]))
    for frame in frames:
        available = pd.to_datetime(frame["available_date"], errors="raise")
        cutoffs = frame["election_id"].map(
            {
                election_id: pd.Timestamp(issue_vote_engine.ELECTION_DATES[election_id])
                - timedelta(days=1)
                for election_id in DEFAULT_ELECTIONS
            }
        )
        assert available.le(cutoffs).all()


def test_automatic_issue_seed_can_target_a_selected_election() -> None:
    elections = ("pres_2017",)
    profile = build_candidate_profile(elections)
    axis = build_mega_axis(elections)
    attribution = build_attribution(profile, axis)

    for frame in (profile, axis):
        assert set(frame["election_id"]) == set(elections)
    # Conservative attribution may be empty when no explicit target direction
    # overlaps the automatically selected mega axes.
    assert set(attribution["election_id"]).issubset(set(elections))


def test_registered_seed_path_prefers_automatic_source(monkeypatch) -> None:
    monkeypatch.setattr(
        issue_vote_engine,
        "THROUGH_2022_REDERIVED_LAYER_CONFIG",
        {
            "automatic_issue_seed_enabled": True,
            "manual_issue_seed_enabled": False,
        },
    )
    assert issue_vote_engine._registered_issue_seed_path("manual.csv", "auto.csv") == "auto.csv"
