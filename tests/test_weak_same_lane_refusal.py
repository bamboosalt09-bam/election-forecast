from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine as engine
from presidential_issue_engine.weak_same_lane_refusal import (
    apply_weak_same_lane_refusal,
)


def _lineage(
    *, major_split: bool = False, origin_lane: str = ""
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "candidate_name": "third",
                "major_split_lineage": major_split,
                "available_date": "2000-01-01",
                "has_party": True,
                "defection_seats": 0 if not major_split else 30,
                "assembly_size": 300,
                "origin_lane": origin_lane,
            }
        ]
    )


def _candidate_row(
    region_id: str,
    slot: str,
    share: float,
    *,
    liberal: float = 0.0,
    conservative: float = 0.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "election_id": "pres_test",
        "region_id": region_id,
        "source_slot": slot,
        "layer_pred": share,
        "candidate_ballot_recent_base": 0.05 if slot == "C" else 0.0,
    }
    for axis in engine.LANDSCAPE_VECTOR_COLUMNS:
        row[f"landscape_axis_{axis}"] = 0.0
    row["landscape_axis_liberal"] = liberal
    row["landscape_axis_conservative"] = conservative
    return row


def _frame(*, third_share: float = 0.09) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region_id in ("r1", "r2"):
        rows.extend(
            [
                _candidate_row(region_id, "A", 0.51, liberal=1.0),
                _candidate_row(
                    region_id,
                    "B",
                    1.0 - 0.51 - third_share,
                    conservative=1.0,
                ),
                _candidate_row(region_id, "C", third_share, liberal=1.0),
            ]
        )
    return pd.DataFrame(rows)


def test_candidate_ballot_mode_preserves_prior_ballot_base() -> None:
    frame = _frame()
    adjusted, audit = apply_weak_same_lane_refusal(
        frame,
        gain=0.25,
        floor_mode="candidate_ballot",
        recipient_weight_mode="affinity_only",
        lineage=_lineage(),
    )

    transfer = 0.25 * (0.09 - 0.05)
    for _, region in adjusted.groupby("region_id"):
        by_slot = region.set_index("source_slot")
        assert by_slot.at["A", "layer_pred"] == pytest.approx(0.51 + transfer)
        assert by_slot.at["B", "layer_pred"] == pytest.approx(0.40)
        assert by_slot.at["C", "layer_pred"] == pytest.approx(0.09 - transfer)
        assert by_slot.at["C", "layer_pred"] >= 0.05
        assert by_slot["layer_pred"].sum() == pytest.approx(1.0)
    assert len(audit) == 2
    assert set(audit["recipient_slots"]) == {"A"}


def test_default_moves_half_of_mass_above_one_percent_floor() -> None:
    frame = _frame()
    adjusted, audit = apply_weak_same_lane_refusal(frame, lineage=_lineage())

    transfer = 0.50 * (0.09 - 0.01)
    for _, region in adjusted.groupby("region_id"):
        by_slot = region.set_index("source_slot")
        assert by_slot.at["A", "layer_pred"] > 0.51
        assert by_slot.at["B", "layer_pred"] > 0.40
        assert by_slot.at["C", "layer_pred"] == pytest.approx(0.09 - transfer)
        assert by_slot.at["C", "layer_pred"] >= 0.01
    assert set(audit["floor_mode"]) == {"theoretical"}
    assert set(audit["protected_floor"]) == {0.01}
    assert set(audit["recipient_weight_mode"]) == {"prediction_tilted"}


def test_none_mode_has_no_absolute_floor() -> None:
    frame = _frame(third_share=0.04)
    adjusted, audit = apply_weak_same_lane_refusal(
        frame,
        gain=1.0,
        floor_mode="none",
        lineage=_lineage(),
    )

    assert set(adjusted.loc[adjusted["source_slot"].eq("C"), "layer_pred"]) == {0.0}
    assert set(audit["protected_floor"]) == {0.0}


def test_prediction_tilted_mode_keeps_both_major_recipients_nonzero() -> None:
    frame = _frame()
    adjusted, audit = apply_weak_same_lane_refusal(
        frame,
        recipient_weight_mode="prediction_tilted",
        lineage=_lineage(),
    )

    for _, region in adjusted.groupby("region_id"):
        before = frame.loc[frame["region_id"].eq(region["region_id"].iloc[0])]
        before = before.set_index("source_slot")
        after = region.set_index("source_slot")
        assert after.at["A", "layer_pred"] > before.at["A", "layer_pred"]
        assert after.at["B", "layer_pred"] > before.at["B", "layer_pred"]
        assert (
            after.at["A", "weak_lane_refusal_transfer_in"]
            > after.at["B", "weak_lane_refusal_transfer_in"]
        )
    assert set(audit["recipient_weight_mode"]) == {"prediction_tilted"}


def test_declared_origin_lane_overrides_noisy_speech_orientation() -> None:
    frame = _frame()
    adjusted, audit = apply_weak_same_lane_refusal(
        frame,
        recipient_weight_mode="prediction_tilted",
        lineage=_lineage(origin_lane="conservative_centrist"),
    )

    for _, region in adjusted.groupby("region_id"):
        by_slot = region.set_index("source_slot")
        assert (
            by_slot.at["B", "weak_lane_refusal_transfer_in"]
            > by_slot.at["A", "weak_lane_refusal_transfer_in"]
        )
    assert set(audit["declared_origin_lane"]) == {"conservative_centrist"}
    assert set(audit["resolved_origin_lane"]) == {"conservative_centrist"}


def test_is_inert_for_major_split_lineage() -> None:
    frame = _frame()
    adjusted, audit = apply_weak_same_lane_refusal(
        frame,
        lineage=_lineage(major_split=True),
    )

    assert adjusted["layer_pred"].equals(frame["layer_pred"])
    assert audit.empty


def test_is_inert_at_or_below_prior_ballot_base() -> None:
    frame = _frame(third_share=0.04)
    adjusted, audit = apply_weak_same_lane_refusal(
        frame,
        floor_mode="candidate_ballot",
        lineage=_lineage(),
    )

    assert adjusted["layer_pred"].equals(frame["layer_pred"])
    assert audit.empty


def test_requires_no_outcome_columns() -> None:
    frame = _frame()
    assert not {"actual", "votes", "vote_share"}.intersection(frame.columns)

    adjusted, audit = apply_weak_same_lane_refusal(frame, lineage=_lineage())

    assert not adjusted.empty
    assert not audit.empty
