from __future__ import annotations

import pandas as pd

from scripts.evaluate_nested_electorate_learning import select_preference_gain


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for election_id in ("pres_2002", "pres_2007"):
        for slot, actual, signal in (("A", 0.51, 1.0), ("B", 0.49, -1.0)):
            rows.append(
                {
                    "election_id": election_id,
                    "region_id": "r1",
                    "slot": slot,
                    "pred": 0.50,
                    "actual": actual,
                    "contest_votes": 100.0,
                    "core_voting_mass": 0.0,
                    "critical_voting_mass": 0.0,
                    "swing_voting_mass": 1.0,
                    "issue_pref_economy": signal,
                }
            )
    return pd.DataFrame(rows)


def test_nested_gain_learner_requires_two_prior_elections() -> None:
    gain, trace = select_preference_gain(
        _synthetic_frame(),
        ("pres_2002",),
        selection_label="test",
        gain_grid=(0.0, 0.01),
    )

    assert gain == 0.0
    assert trace == []


def test_nested_gain_learner_selects_consistently_improving_gain() -> None:
    gain, trace = select_preference_gain(
        _synthetic_frame(),
        ("pres_2002", "pres_2007"),
        selection_label="test",
        gain_grid=(0.0, 0.01),
    )

    assert gain == 0.01
    selected = [row for row in trace if row["gain"] == gain][0]
    assert selected["eligible"] is True
    assert selected["improved_elections"] == 2


def test_nested_gain_learner_expands_beyond_initial_search_range() -> None:
    frame = _synthetic_frame()
    frame.loc[frame["slot"].eq("A"), "actual"] = 0.60
    frame.loc[frame["slot"].eq("B"), "actual"] = 0.40

    gain, trace = select_preference_gain(
        frame,
        ("pres_2002", "pres_2007"),
        selection_label="adaptive_test",
    )

    assert gain > 0.04
    assert max(float(row["gain"]) for row in trace) > gain
    assert trace[-1]["search_stop_reason"] == "interior_optimum_confirmed"
    assert trace[-1]["search_converged"] is True
