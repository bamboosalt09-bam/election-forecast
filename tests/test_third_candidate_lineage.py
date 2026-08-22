"""Scope and behaviour guards for the V24 third-candidate lineage ceiling."""

from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import third_candidate_defection_scale as scale
from presidential_issue_engine import third_candidate_lineage_constraint as constraint
from presidential_issue_engine.election_scope import FORECAST_ONLY_ELECTIONS, SCORED_ELECTIONS
from presidential_issue_engine.issue_vote_engine import ELECTION_DATES

OUTCOME_TOKENS = ("vote_share", "votes", "actual", "result", "winner", "margin", "rank")


def test_lineage_table_carries_no_outcome_column() -> None:
    frame = pd.read_csv(constraint.LINEAGE_TABLE, encoding="utf-8-sig")
    offending = [
        column
        for column in frame.columns
        if any(token in column.casefold() for token in OUTCOME_TOKENS)
    ]
    assert offending == [], f"lineage table exposes outcome columns: {offending}"


def test_forecast_only_rows_stay_before_their_cutoff() -> None:
    frame = constraint.load_lineage()
    for election_id in FORECAST_ONLY_ELECTIONS:
        rows = frame.loc[frame["election_id"].astype(str).eq(election_id)]
        if rows.empty:
            continue
        cutoff = pd.Timestamp(ELECTION_DATES[election_id]) - pd.Timedelta(days=1)
        available = pd.to_datetime(rows["available_date"], errors="raise")
        assert (available <= cutoff).all(), f"{election_id} lineage row is not point-in-time"


def test_floor_choice_is_settled_only_by_scored_anchors() -> None:
    """Every floor strictly inside the scored gap must select the same scored set."""

    frame = constraint.load_lineage()
    scored = frame.loc[frame["election_id"].astype(str).isin(SCORED_ELECTIONS)]
    shares = (
        pd.to_numeric(scored["defection_seats"], errors="coerce")
        / pd.to_numeric(scored["assembly_size"], errors="coerce")
    ).dropna()
    low, high = float(shares.min()), float(shares.max())
    assert low < constraint.DEFAULT_DEFECTION_FLOOR < high

    reference = None
    for floor in (low + 1e-6, constraint.DEFAULT_DEFECTION_FLOOR, high - 1e-6):
        selected = {
            election
            for election in constraint.self_founded_elections(frame, defection_floor=floor)
            if election in SCORED_ELECTIONS
        }
        if reference is None:
            reference = selected
        assert selected == reference, "scored selection is sensitive to the floor"


def test_independents_fall_back_to_the_documented_flag() -> None:
    frame = constraint.load_lineage()
    weak = constraint.self_founded_elections(frame)
    assert "pres_2007" not in weak, "이회창 2007 carries major-party lineage without a party"
    assert "pres_2012" in weak, "강지원 2012 carries no lineage"


def test_ceiling_is_one_sided_and_conserves_each_region() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2002"] * 3,
            "region_id": ["sido_11"] * 3,
            "slot": ["A", "B", "C"],
            "candidate_name": ["노무현", "이회창", "권영길"],
            "layer_pred": [0.45, 0.40, 0.15],
            "direct_party_recent_base": [0.50, 0.45, 0.05],
        }
    )
    adjusted, audit = constraint.apply_lineage_ceiling(frame)
    assert len(audit) == 1
    third = adjusted.loc[adjusted.slot.eq("C"), "layer_pred"].iloc[0]
    assert third == pytest.approx(0.05 / 1.0, abs=1e-9)
    assert adjusted["layer_pred"].sum() == pytest.approx(1.0)
    assert adjusted.loc[adjusted.slot.eq("A"), "layer_pred"].iloc[0] > 0.45


def test_ceiling_is_inert_when_the_third_candidate_is_already_low() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2022"] * 3,
            "region_id": ["sido_11"] * 3,
            "slot": ["A", "B", "C"],
            "candidate_name": ["윤석열", "이재명", "심상정"],
            "layer_pred": [0.49, 0.48, 0.03],
            "direct_party_recent_base": [0.45, 0.45, 0.10],
        }
    )
    adjusted, audit = constraint.apply_lineage_ceiling(frame)
    assert audit.empty
    assert adjusted["layer_pred"].to_list() == pytest.approx([0.49, 0.48, 0.03])


def test_defection_scale_reports_missing_rather_than_guessing() -> None:
    report = scale.coverage_report()
    unsourced = report.loc[report["defection_seats_source"].isin({"needs_source", "no_party"})]
    assert not unsourced.empty
    assert unsourced["defection_scale"].isna().all()


def test_unknown_recipient_mode_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2002"] * 3,
            "region_id": ["sido_11"] * 3,
            "slot": ["A", "B", "C"],
            "candidate_name": ["노무현", "이회창", "권영길"],
            "layer_pred": [0.45, 0.40, 0.15],
            "direct_party_recent_base": [0.50, 0.45, 0.05],
        }
    )
    with pytest.raises(ValueError, match="recipient mode"):
        constraint.apply_lineage_ceiling(frame, recipient_weight_mode="nonsense")


def test_reference_mode_splits_the_excess_at_the_untouched_column() -> None:
    """The whole point is that an earlier layer's tilt must not decide the split."""

    frame = pd.DataFrame(
        {
            "election_id": ["pres_2002"] * 3,
            "region_id": ["sido_11"] * 3,
            "slot": ["A", "B", "C"],
            "candidate_name": ["노무현", "이회창", "권영길"],
            # a previous layer has already tilted the two majors far apart
            "layer_pred": [0.20, 0.65, 0.15],
            # their own standing before any postprocess is even
            "anchored_pred": [0.40, 0.40, 0.20],
            "direct_party_recent_base": [0.50, 0.45, 0.05],
        }
    )
    live, _ = constraint.apply_lineage_ceiling(frame, recipient_weight_mode="live")
    reference, _ = constraint.apply_lineage_ceiling(
        frame, recipient_weight_mode="reference"
    )

    excess = 0.15 - 0.05
    # live inherits the tilt: B takes 0.65/0.85 of the excess
    assert live.loc[live.slot.eq("B"), "layer_pred"].iloc[0] == pytest.approx(
        0.65 + excess * (0.65 / 0.85)
    )
    # reference splits evenly, because anchored_pred is even
    assert reference.loc[reference.slot.eq("A"), "layer_pred"].iloc[0] == pytest.approx(
        0.20 + excess / 2
    )
    assert reference.loc[reference.slot.eq("B"), "layer_pred"].iloc[0] == pytest.approx(
        0.65 + excess / 2
    )
    # both conserve the contest and cap the third candidate identically
    for adjusted in (live, reference):
        assert adjusted["layer_pred"].sum() == pytest.approx(1.0)
        assert adjusted.loc[adjusted.slot.eq("C"), "layer_pred"].iloc[0] == pytest.approx(0.05)


def test_reference_mode_falls_back_rather_than_skipping_the_cap() -> None:
    """A missing reference must not leave the third candidate above its ceiling."""

    frame = pd.DataFrame(
        {
            "election_id": ["pres_2002"] * 3,
            "region_id": ["sido_11"] * 3,
            "slot": ["A", "B", "C"],
            "candidate_name": ["노무현", "이회창", "권영길"],
            "layer_pred": [0.45, 0.40, 0.15],
            "direct_party_recent_base": [0.50, 0.45, 0.05],
        }
    )
    adjusted, audit = constraint.apply_lineage_ceiling(
        frame, recipient_weight_mode="reference"
    )
    assert adjusted.loc[adjusted.slot.eq("C"), "layer_pred"].iloc[0] == pytest.approx(0.05)
    assert audit.iloc[0]["recipient_weight_mode"] == "reference"


def test_the_shipped_default_is_still_the_live_split() -> None:
    assert constraint.DEFAULT_RECIPIENT_WEIGHT_MODE == "live"
