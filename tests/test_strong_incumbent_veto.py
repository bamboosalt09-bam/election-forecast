from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import strong_incumbent_veto
from presidential_issue_engine.strong_incumbent_veto import (
    DEFAULT_GAIN,
    apply_strong_incumbent_veto,
)


def _frame(*, dominant_share: float = 0.50, runner_share: float = 0.35) -> pd.DataFrame:
    rows = []
    for region_id in ("r1", "r2"):
        rows.extend(
            [
                {
                    "election_id": "pres_test",
                    "region_id": region_id,
                    "source_slot": "A",
                    "layer_pred": dominant_share,
                    "government_direction_score": 0.0,
                    "government_rejection_strength": 0.0,
                    "dominant_slot": "A",
                    "runner_up_slot": "B",
                    "dominance_activation": 1.0,
                    "regime_certainty": 1.0,
                    "regime_core_floor": 0.30,
                },
                {
                    "election_id": "pres_test",
                    "region_id": region_id,
                    "source_slot": "B",
                    "layer_pred": runner_share,
                    "government_direction_score": -0.20,
                    "government_rejection_strength": 0.20,
                    "dominant_slot": "A",
                    "runner_up_slot": "B",
                    "dominance_activation": 1.0,
                    "regime_certainty": 1.0,
                    "regime_core_floor": 0.20,
                },
                {
                    "election_id": "pres_test",
                    "region_id": region_id,
                    "source_slot": "C",
                    "layer_pred": 1.0 - dominant_share - runner_share,
                    "government_direction_score": 0.0,
                    "government_rejection_strength": 0.0,
                    "dominant_slot": "A",
                    "runner_up_slot": "B",
                    "dominance_activation": 1.0,
                    "regime_certainty": 1.0,
                    "regime_core_floor": 0.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_moves_only_runner_flexible_mass_after_ten_point_gate() -> None:
    frame = _frame()
    adjusted, audit = apply_strong_incumbent_veto(frame)

    # Without rupture evidence the existing regime floor remains intact.
    transfer = DEFAULT_GAIN * 0.20 * 0.15
    for _, region in adjusted.groupby("region_id"):
        by_slot = region.set_index("source_slot")
        assert by_slot.at["A", "layer_pred"] == pytest.approx(0.50 + transfer)
        assert by_slot.at["B", "layer_pred"] == pytest.approx(0.35 - transfer)
        assert by_slot.at["C", "layer_pred"] == pytest.approx(0.15)
        assert by_slot["layer_pred"].sum() == pytest.approx(1.0)
        assert by_slot.at["B", "layer_pred"] >= by_slot.at["B", "regime_core_floor"]
    assert len(audit) == 2
    assert set(audit["beneficiary_slot"]) == {"A"}
    assert set(audit["burdened_slot"]) == {"B"}


def test_constitutional_rupture_continuously_erodes_core_floor_to_one_percent() -> None:
    frame = _frame()
    burdened = frame["source_slot"].eq("B")
    frame.loc[burdened, "mega_issue_intensity_response"] = 2.0
    frame.loc[burdened, "direct_mega_score"] = -0.25
    frame.loc[burdened, "government_negative_share"] = 1.0
    frame.loc[burdened, "government_rejection_breadth"] = 1.0

    adjusted, audit = apply_strong_incumbent_veto(frame)

    transfer = 0.20 * (0.35 - 0.01)
    for _, region in adjusted.groupby("region_id"):
        by_slot = region.set_index("source_slot")
        assert by_slot.at["A", "layer_pred"] == pytest.approx(0.50 + transfer)
        assert by_slot.at["B", "layer_pred"] == pytest.approx(0.35 - transfer)
        assert by_slot.at["B", "layer_pred"] >= 0.01
    assert set(audit["rupture_floor_activation"]) == {1.0}
    assert set(audit["effective_runner_floor"]) == {0.01}


def test_is_inert_below_projected_ten_point_margin() -> None:
    frame = _frame(dominant_share=0.46, runner_share=0.40)
    adjusted, audit = apply_strong_incumbent_veto(frame)

    assert adjusted["layer_pred"].equals(frame["layer_pred"])
    assert audit.empty


def test_is_inert_when_government_candidate_is_not_runner_up() -> None:
    frame = _frame()
    frame.loc[frame["source_slot"].eq("B"), "government_direction_score"] = 0.0
    frame.loc[frame["source_slot"].eq("A"), "government_direction_score"] = -0.20
    adjusted, audit = apply_strong_incumbent_veto(frame)

    assert adjusted["layer_pred"].equals(frame["layer_pred"])
    assert audit.empty


def test_requires_no_outcome_columns() -> None:
    frame = _frame()
    assert not {"actual", "votes", "vote_share"}.intersection(frame.columns)

    adjusted, audit = apply_strong_incumbent_veto(frame)

    assert not adjusted.empty
    assert not audit.empty


def test_unknown_floor_erosion_mode_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2017"] * 2,
            "region_id": ["sido_11"] * 2,
            "source_slot": ["A", "B"],
            "layer_pred": [0.5, 0.5],
        }
    )
    with pytest.raises(ValueError, match="floor erosion mode"):
        strong_incumbent_veto.apply_strong_incumbent_veto(
            frame, floor_erosion_mode="nonsense"
        )


def test_the_shipped_default_is_still_proportional_erosion() -> None:
    """The absolute mode is measured and unadopted; the default must not drift."""

    assert strong_incumbent_veto.DEFAULT_FLOOR_EROSION_MODE == "proportional"
    assert strong_incumbent_veto.FLOOR_EROSION_MODES == {"proportional", "absolute"}
