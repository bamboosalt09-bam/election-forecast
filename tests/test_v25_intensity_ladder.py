"""Guards for the measured-but-unpromoted V25 intensity ladder candidate.

The ladder is not wired into the active model. These tests pin the properties
that make it a legitimate candidate - it introduces no constant of its own and
cannot move an election the classifier already calls a crisis - so that the
experiment record stays checkable against the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from presidential_issue_engine import automatic_controls_v22 as controls
from scripts import evaluate_v25_intensity_ladder as ladder


def _diagnostics(**components: float) -> pd.DataFrame:
    base = {
        "salience_component": 0.5,
        "severity_component": 0.5,
        "breadth_component": 0.5,
        "accountability_component": 0.5,
    }
    base.update(components)
    return pd.DataFrame({"election_id": ["pres_test"], **{k: [v] for k, v in base.items()}})


def test_the_ladder_reuses_the_classifier_thresholds_rather_than_its_own() -> None:
    """A shock class boundary must be described by one pair of numbers only."""

    assert controls.CRISIS_MIN_REGIME_EVIDENCE == 0.65
    assert controls.CRISIS_ACCOUNTABILITY == 0.75
    assert ladder.CRISIS_INTENSITY == controls.SHOCK_CLASS_INTENSITY[
        "institutional_crisis"
    ]


def test_classifier_still_selects_the_crisis_class_at_the_named_thresholds() -> None:
    """Naming the constants must not have moved the boundary."""

    frame = pd.DataFrame(
        {
            "election_id": ["at_gate", "below_regime", "below_accountability"],
            "source_rows": [10_000] * 3,
            "salience_component": [0.90, 0.90, 0.90],
            "severity_component": [
                controls.CRISIS_MIN_REGIME_EVIDENCE,
                controls.CRISIS_MIN_REGIME_EVIDENCE - 0.01,
                controls.CRISIS_MIN_REGIME_EVIDENCE,
            ],
            "breadth_component": [0.90, 0.90, 0.90],
            "accountability_component": [
                controls.CRISIS_ACCOUNTABILITY,
                controls.CRISIS_ACCOUNTABILITY,
                controls.CRISIS_ACCOUNTABILITY - 0.01,
            ],
            "joint_evidence": [0.80] * 3,
            "available_date": ["2020-01-01"] * 3,
        }
    )
    taxonomy, _intensity, _audit = controls.build_automatic_mega_taxonomy(frame)
    selected = dict(zip(taxonomy["election_id"], taxonomy["shock_type"]))
    assert selected["at_gate"] == "institutional_crisis"
    assert selected["below_regime"] != "institutional_crisis"
    assert selected["below_accountability"] != "institutional_crisis"


def test_proximity_saturates_at_and_above_both_gates() -> None:
    at_gate = ladder.crisis_proximity(
        _diagnostics(
            salience_component=1.0,
            severity_component=controls.CRISIS_MIN_REGIME_EVIDENCE,
            breadth_component=1.0,
            accountability_component=controls.CRISIS_ACCOUNTABILITY,
        )
    )
    assert at_gate.iloc[0] == pytest.approx(1.0)

    above = ladder.crisis_proximity(
        _diagnostics(
            salience_component=1.0,
            severity_component=1.0,
            breadth_component=1.0,
            accountability_component=1.0,
        )
    )
    assert above.iloc[0] == pytest.approx(1.0)


def test_proximity_is_bounded_and_rises_with_the_binding_component() -> None:
    weak = ladder.crisis_proximity(_diagnostics(severity_component=0.10)).iloc[0]
    strong = ladder.crisis_proximity(_diagnostics(severity_component=0.40)).iloc[0]
    assert 0.0 <= weak < strong <= 1.0


def test_an_election_already_at_the_ceiling_cannot_move() -> None:
    """2017 and 2025 both clear the gates, so the ladder must preserve them."""

    diagnostics = _diagnostics(
        salience_component=1.0,
        severity_component=1.0,
        breadth_component=1.0,
        accountability_component=1.0,
    )
    intensity = pd.DataFrame(
        {
            "election_id": ["pres_test"],
            "mega_issue_intensity": [ladder.CRISIS_INTENSITY],
            "available_date": ["2020-01-01"],
        }
    )
    result = ladder.ladder_intensity(intensity, diagnostics)
    assert result["mega_issue_intensity"].iloc[0] == pytest.approx(
        ladder.CRISIS_INTENSITY
    )


def test_the_ladder_is_one_sided_and_never_exceeds_the_crisis_ceiling() -> None:
    diagnostics = pd.concat(
        [
            _diagnostics(severity_component=value).assign(election_id=f"e{index}")
            for index, value in enumerate((0.05, 0.20, 0.40, 0.65, 0.90))
        ],
        ignore_index=True,
    )
    intensity = pd.DataFrame(
        {
            "election_id": [f"e{index}" for index in range(5)],
            "mega_issue_intensity": [0.50, 0.75, 1.00, 1.00, 0.50],
            "available_date": ["2020-01-01"] * 5,
        }
    )
    result = ladder.ladder_intensity(intensity, diagnostics)
    raised = pd.to_numeric(result["mega_issue_intensity"])
    floor = pd.to_numeric(intensity["mega_issue_intensity"])
    assert (raised >= floor - 1e-9).all()
    assert (raised <= ladder.CRISIS_INTENSITY + 1e-9).all()


def test_an_election_missing_from_the_diagnostics_keeps_its_floor() -> None:
    """An unmeasured election must not be silently raised toward the ceiling."""

    intensity = pd.DataFrame(
        {
            "election_id": ["pres_unmeasured"],
            "mega_issue_intensity": [0.75],
            "available_date": ["2020-01-01"],
        }
    )
    result = ladder.ladder_intensity(intensity, _diagnostics())
    assert result["mega_issue_intensity"].iloc[0] == pytest.approx(0.75)


def test_the_scored_panel_reproduces_the_recorded_ladder() -> None:
    """The experiment record's intensity table must match what the code produces."""

    if not ladder.DIAGNOSTICS.exists():
        pytest.skip(f"classifier diagnostics not present: {ladder.DIAGNOSTICS.name}")
    diagnostics = pd.read_csv(ladder.DIAGNOSTICS, encoding="utf-8-sig")
    intensity = pd.read_csv(
        ladder.AUTOMATIC_DIR / "mega_issue_intensity.csv", encoding="utf-8-sig"
    )
    result = ladder.ladder_intensity(intensity, diagnostics)
    recorded = {
        "pres_2002": 0.683736,
        "pres_2007": 1.590052,
        "pres_2012": 1.195804,
        "pres_2017": 2.000000,
        "pres_2022": 1.575327,
    }
    produced = dict(
        zip(result["election_id"].astype(str), result["mega_issue_intensity"].astype(float))
    )
    for election, expected in recorded.items():
        assert produced[election] == pytest.approx(expected, abs=1e-6), election
    # activation is zero below the gate and one only at the ceiling
    activation = {
        election: float(np.clip(value - 1.0, 0.0, 1.0))
        for election, value in produced.items()
    }
    assert activation["pres_2002"] == pytest.approx(0.0)
    assert activation["pres_2017"] == pytest.approx(1.0)
