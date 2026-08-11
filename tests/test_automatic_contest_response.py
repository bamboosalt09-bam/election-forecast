from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.automatic_contest_response import (
    apply_prior_selected_contest_response,
)


def _fake_response(
    frame: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    prediction_column: str,
    slot_column: str,
    output_column: str,
    expansion_gain: float,
    **_: object,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    activation = regimes.set_index("election_id")["dominance_activation"]
    shift = out["election_id"].map(activation).fillna(0.0) * expansion_gain * 0.1
    out[output_column] = out[prediction_column]
    out.loc[out[slot_column].eq("A"), output_column] += shift
    out.loc[out[slot_column].eq("B"), output_column] -= shift
    return out


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for election_id in ["e1", "e2", "e3"]:
        rows.extend(
            [
                {
                    "election_id": election_id,
                    "region_id": "r1",
                    "source_slot": "A",
                    "pred": 0.5,
                    "actual": 0.6,
                    "contest_votes": 100.0,
                },
                {
                    "election_id": election_id,
                    "region_id": "r1",
                    "source_slot": "B",
                    "pred": 0.5,
                    "actual": 0.4,
                    "contest_votes": 100.0,
                },
            ]
        )
    regimes = pd.DataFrame(
        [
            {
                "election_id": election_id,
                "dominant_slot": "A",
                "runner_up_slot": "B",
                "dominance_activation": 1.0,
                "regime_rejection_activation": 0.0,
                "regime_certainty": 1.0,
                "cumulative_rejection_advantage": 0.0,
            }
            for election_id in ["e1", "e2", "e3"]
        ]
    )
    return pd.DataFrame(rows), regimes


def test_target_outcome_cannot_change_its_prior_selected_gain() -> None:
    frame, regimes = _fixture()
    _, audit = apply_prior_selected_contest_response(
        frame,
        regimes,
        prediction_column="pred",
        output_column="pred",
        apply_response=_fake_response,
        election_order=["e1", "e2", "e3"],
    )
    changed = frame.copy()
    changed.loc[changed["election_id"].eq("e3"), "actual"] = [0.1, 0.9]
    _, changed_audit = apply_prior_selected_contest_response(
        changed,
        regimes,
        prediction_column="pred",
        output_column="pred",
        apply_response=_fake_response,
        election_order=["e1", "e2", "e3"],
    )

    original_gain = audit.set_index("target_election").at["e3", "selected_gain"]
    changed_gain = changed_audit.set_index("target_election").at[
        "e3", "selected_gain"
    ]
    assert original_gain == pytest.approx(changed_gain)
    assert audit["target_excluded_from_selection"].all()
    assert audit.set_index("target_election").at["e1", "selected_gain"] == pytest.approx(
        0.5
    )
