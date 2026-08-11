from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine import issue_vote_engine, strategic_lane_transfer


def _frame(resistance: float = 0.0) -> pd.DataFrame:
    rows = [
        ("A", True, 0.45, 0.0, 0.45, 0.8, 0.0),
        ("B", True, 0.40, 0.0, 0.40, 0.0, 0.8),
        ("C", False, 0.15, 0.10, 0.10, 0.8, 0.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "slot",
            "major_party_core_eligible",
            "layer_pred",
            "critical_voting_mass_effective",
            "preliminary_mean_share",
            "landscape_axis_conservative",
            "landscape_axis_liberal",
        ],
    )
    frame["election_id"] = "pres_test"
    frame["region_id"] = "region_1"
    frame["bloc"] = "minor"
    frame["wasted_vote_resistance"] = [0.0, 0.0, resistance]
    frame["major_party_gravity"] = [0.0, 0.0, 1.0]
    frame["strategic_transfer_confidence"] = [0.0, 0.0, 1.0]
    for axis in (
        "progressive",
        "centrist",
        "anti_establishment",
        "reform",
        "regionalist",
    ):
        frame[f"landscape_axis_{axis}"] = 0.0
    return frame


def test_transfer_is_zero_sum_and_goes_to_aligned_major_candidate() -> None:
    frame = _frame()
    result = strategic_lane_transfer.apply_strategic_lane_transfer(frame)
    assert np.isclose(result["layer_pred"].sum(), frame["layer_pred"].sum())
    assert result.loc[result["slot"].eq("C"), "strategic_lane_transfer_out"].iloc[0] > 0
    assert result.loc[result["slot"].eq("A"), "strategic_lane_transfer_in"].iloc[0] > 0
    assert result.loc[result["slot"].eq("B"), "strategic_lane_transfer_in"].iloc[0] == 0
    assert result.loc[result["slot"].eq("C"), "strategic_lane_transfer_out"].iloc[0] <= 0.10


def test_full_wasted_vote_resistance_blocks_transfer() -> None:
    frame = _frame(resistance=1.0)
    result = strategic_lane_transfer.apply_strategic_lane_transfer(frame)
    assert np.allclose(result["layer_pred"], frame["layer_pred"])


def test_progressive_minor_support_can_move_to_liberal_major() -> None:
    frame = _frame()
    frame.loc[frame["slot"].eq("C"), "landscape_axis_conservative"] = 0.0
    frame.loc[frame["slot"].eq("C"), "landscape_axis_progressive"] = 0.8
    result = strategic_lane_transfer.apply_strategic_lane_transfer(frame)
    assert result.loc[result["slot"].eq("A"), "strategic_lane_transfer_in"].iloc[0] == 0
    assert result.loc[result["slot"].eq("B"), "strategic_lane_transfer_in"].iloc[0] > 0


def test_context_join_uses_candidate_not_changed_slot_and_filters_future() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_test", "pres_test"],
            "candidate_name_x": ["Candidate One", "Candidate Two"],
            "slot": ["B", "A"],
        }
    )
    context = pd.DataFrame(
        {
            "election_id": ["pres_test", "pres_test"],
            "candidate_name": ["Candidate One", "Candidate Two"],
            "slot": ["A", "B"],
            "wasted_vote_resistance": [0.2, 0.9],
            "major_party_gravity": [0.4, 0.9],
            "available_date": ["2020-01-01", "2020-01-03"],
            "confidence": [0.8, 0.9],
        }
    )
    result = strategic_lane_transfer.attach_conversion_context(
        frame, context, {"pres_test": pd.Timestamp("2020-01-02")}
    )
    assert result.loc[result["candidate_name_x"].eq("Candidate One"), "major_party_gravity"].iloc[0] == 0.4
    assert result.loc[result["candidate_name_x"].eq("Candidate Two"), "major_party_gravity"].iloc[0] == 0.0


def test_shared_orientation_affinity_rejects_conservative_liberal_centrist_pair() -> None:
    conservative = pd.Series(
        {"bloc": "minor", "landscape_axis_conservative": 0.8}
    )
    liberal_centrist = pd.Series(
        {
            "bloc": "minor",
            "landscape_axis_conservative": 0.0,
            "landscape_axis_liberal": 0.8,
        }
    )
    assert issue_vote_engine._orientation_affinity(conservative, liberal_centrist) == 0.0
