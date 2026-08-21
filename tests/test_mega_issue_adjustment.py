from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.mega_issue_adjustment import (
    align_profile_to_event_class,
    apply_direct_mega_shift,
    compile_direct_mega_scores,
)


ELECTION_DATES = {
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "B",
                "issue_name": "regime_change",
                "direction": -0.9,
                "association_strength": 0.5,
                "confidence": 0.4,
                "target_absolute_evidence": 0.7,
                "target_attribution_confidence": 0.5,
                "available_date": "2017-05-08",
            },
            {
                "election_id": "pres_2017",
                "slot": "B",
                "issue_name": "external_shock",
                "direction": -0.8,
                "association_strength": 0.5,
                "confidence": 0.4,
                "target_absolute_evidence": 0.2,
                "target_attribution_confidence": 0.3,
                "available_date": "2017-05-08",
            },
            {
                "election_id": "pres_2022",
                "slot": "B",
                "issue_name": "regime_change",
                "direction": -0.9,
                "association_strength": 0.5,
                "confidence": 0.4,
                "target_absolute_evidence": 0.7,
                "target_attribution_confidence": 0.5,
                "available_date": "2022-03-08",
            },
        ]
    )


def test_compile_selects_only_strongest_issue_above_intensity_gate() -> None:
    intensity = pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "mega_issue_intensity": 2.0,
                "available_date": "2017-05-01",
            },
            {
                "election_id": "pres_2022",
                "mega_issue_intensity": 1.0,
                "available_date": "2022-03-01",
            },
        ]
    )

    scores = compile_direct_mega_scores(_profile(), intensity, ELECTION_DATES)

    assert scores[["election_id", "slot", "issue_name"]].to_dict("records") == [
        {"election_id": "pres_2017", "slot": "B", "issue_name": "regime_change"}
    ]
    assert scores.loc[0, "direct_mega_score"] == pytest.approx(-0.36)


def test_institutional_crisis_does_not_amplify_unrelated_withdrawal() -> None:
    profile = pd.concat(
        [
            _profile(),
            pd.DataFrame(
                [
                    {
                        "election_id": "pres_2017",
                        "slot": "A",
                        "issue_name": "withdrawal_event",
                        "direction": -1.0,
                        "association_strength": 1.0,
                        "confidence": 1.0,
                        "target_absolute_evidence": 100.0,
                        "target_attribution_confidence": 1.0,
                        "available_date": "2017-05-08",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    taxonomy = pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "shock_type": "institutional_crisis",
                "available_date": "2017-05-01",
            }
        ]
    )

    aligned = align_profile_to_event_class(profile, taxonomy, ELECTION_DATES)

    selected = aligned.loc[aligned["election_id"].eq("pres_2017"), "issue_name"]
    assert set(selected) == {"regime_change"}


def test_compile_excludes_future_evidence() -> None:
    profile = _profile()
    profile.loc[profile["election_id"].eq("pres_2017"), "available_date"] = "2017-05-09"
    intensity = pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "mega_issue_intensity": 2.0,
                "available_date": "2017-05-01",
            }
        ]
    )

    scores = compile_direct_mega_scores(profile, intensity, ELECTION_DATES)

    assert scores.empty


def test_compile_ramps_continuously_above_intensity_gate() -> None:
    intensity = pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "mega_issue_intensity": 1.05,
                "available_date": "2017-05-01",
            }
        ]
    )

    scores = compile_direct_mega_scores(_profile(), intensity, ELECTION_DATES)

    full_score = -0.9 * 0.5 * 0.4 * 1.05
    assert scores.loc[0, "direct_mega_score"] == pytest.approx(full_score * 0.05)
    assert abs(scores.loc[0, "direct_mega_score"]) < 0.02


def test_apply_shift_reduces_target_and_preserves_contest_sum() -> None:
    frame = pd.DataFrame(
        [
            {"election_id": "pres_2017", "region_id": "11", "source_slot": "A", "pred": 0.4},
            {"election_id": "pres_2017", "region_id": "11", "source_slot": "B", "pred": 0.4},
            {"election_id": "pres_2017", "region_id": "11", "source_slot": "C", "pred": 0.2},
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "B",
                "issue_name": "regime_change",
                "direct_mega_score": -0.5,
            }
        ]
    )

    out = apply_direct_mega_shift(frame, scores, prediction_column="pred", gain=0.4)

    assert out.loc[out["source_slot"].eq("B"), "pred"].iloc[0] < 0.4
    assert out.groupby(["election_id", "region_id"])["pred"].sum().iloc[0] == pytest.approx(1.0)
    assert out["direct_mega_log_shift"].min() == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# Event-class alignment: scope and shape guards (2026-08-21)
#
# ``align_profile_to_event_class`` is reached from the prospective forecast
# only; the retrospective panel calls ``compile_direct_mega_scores`` on the raw
# profile. The two guards below pin what that asymmetry does and does not cost.
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from presidential_issue_engine.automatic_controls_v22 import (  # noqa: E402
    SHOCK_CLASS_INTENSITY,
)
from presidential_issue_engine.issue_vote_engine import (  # noqa: E402
    ELECTION_DATES as PANEL_ELECTION_DATES,
)

_ROOT = Path(__file__).resolve().parents[1]
SCORED_TAXONOMY = _ROOT / "outputs" / "automatic_controls_v23" / "mega_issue_taxonomy.csv"
SCORED_PROFILE = _ROOT / "data" / "raw" / "auto_issue_seed" / "candidate_issue_profile.csv"


def _panel_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    profile = pd.read_csv(SCORED_PROFILE, encoding="utf-8-sig")
    taxonomy = pd.read_csv(SCORED_TAXONOMY, encoding="utf-8-sig")
    intensity = taxonomy.assign(
        mega_issue_intensity=taxonomy["shock_type"]
        .astype(str)
        .map(SHOCK_CLASS_INTENSITY)
        .fillna(1.0)
    )[["election_id", "mega_issue_intensity", "available_date"]]
    scored = set(taxonomy["election_id"].astype(str))
    dates = {
        election: date
        for election, date in PANEL_ELECTION_DATES.items()
        if str(election) in scored
    }
    return profile, taxonomy, intensity, dates


def test_event_class_alignment_is_inert_on_the_scored_panel() -> None:
    """The forecast-only alignment must not move a single scored number.

    Two independent reasons make it inert. The elections whose winning issue is
    off-class (2007, 2022) sit at intensity 1.00, where ``intensity_activation``
    is exactly zero; the one election above the gate (2017) already selects an
    on-class issue. The consequence is that the retrospective cannot measure
    this component at all - it is a declared assumption rather than a fitted
    one - so this guard exists to make any change to that status visible.
    """

    profile, taxonomy, intensity, dates = _panel_inputs()
    raw = compile_direct_mega_scores(profile, intensity, dates)
    aligned = compile_direct_mega_scores(
        align_profile_to_event_class(profile, taxonomy, dates), intensity, dates
    )
    pd.testing.assert_frame_equal(
        raw.sort_values(["election_id", "slot"]).reset_index(drop=True),
        aligned.sort_values(["election_id", "slot"]).reset_index(drop=True),
    )


def test_direct_mega_selection_stays_unique_per_election_and_slot() -> None:
    """``apply_direct_mega_shift`` merges with ``validate="many_to_one"``.

    A graded alignment weight that let a second issue survive the race would
    break that merge outright rather than blend the two shocks, so the
    winner-take-all selection is a downstream requirement and not a stylistic
    choice. The intensity is forced above the gate here so that every election
    reaches the selection step.
    """

    profile, _taxonomy, intensity, dates = _panel_inputs()
    intensity = intensity.assign(mega_issue_intensity=2.0)
    scores = compile_direct_mega_scores(profile, intensity, dates)
    assert not scores.empty
    assert not scores.duplicated(["election_id", "slot"]).any()
    assert scores.groupby("election_id")["issue_name"].nunique().le(1).all()
