"""A calibration that runs out of iterations must fail, not return quietly."""

from __future__ import annotations

import numpy as np
import pytest

from presidential_issue_engine import calibration_guard
from presidential_issue_engine.party_regionalism_dispersion import _calibrate


def _well_posed() -> dict[str, np.ndarray]:
    """Two candidates over three regions, targets the alternation can meet."""

    candidate_codes = np.array([0, 1, 0, 1, 0, 1])
    region_codes = np.array([0, 0, 1, 1, 2, 2])
    values = np.array([0.55, 0.45, 0.40, 0.60, 0.70, 0.30])
    weights = np.array([0.5, 0.5, 0.3, 0.3, 0.2, 0.2])
    targets = np.array([0.54, 0.46])
    return {
        "values": values,
        "candidate_codes": candidate_codes,
        "region_codes": region_codes,
        "weights": weights,
        "targets": targets,
    }


def test_a_converged_run_is_unchanged_by_the_guard() -> None:
    case = _well_posed()
    plain = _calibrate(
        case["values"], case["candidate_codes"], case["region_codes"],
        case["weights"], case["targets"],
    )
    guarded = calibration_guard.checked(_calibrate)(
        case["values"], case["candidate_codes"], case["region_codes"],
        case["weights"], case["targets"],
    )
    np.testing.assert_allclose(guarded, plain, rtol=0, atol=0)


def test_too_few_iterations_raises_instead_of_returning() -> None:
    """The original returns its last iterate; the guard refuses it."""

    case = _well_posed()
    unguarded = _calibrate(
        case["values"], case["candidate_codes"], case["region_codes"],
        case["weights"], case["targets"], iterations=1,
    )
    assert unguarded is not None, "the unguarded function returns regardless"

    with pytest.raises(calibration_guard.CalibrationDidNotConverge) as caught:
        calibration_guard.checked(_calibrate)(
            case["values"], case["candidate_codes"], case["region_codes"],
            case["weights"], case["targets"], iterations=1,
        )
    assert "acceptance tolerance" in str(caught.value)
    assert "residual" in str(caught.value)


def test_an_unreachable_target_raises_at_the_full_iteration_budget() -> None:
    """A pathological case: targets that do not sum to one cannot be met.

    Every region is renormalised to sum to one, so the candidate levels this
    produces must sum to one as well. Asking for targets that sum to 1.6 is
    unsatisfiable, and 200 rounds of alternation cannot make it otherwise.
    """

    case = _well_posed()
    case["targets"] = np.array([0.9, 0.7])
    with pytest.raises(calibration_guard.CalibrationDidNotConverge):
        calibration_guard.checked(_calibrate)(
            case["values"], case["candidate_codes"], case["region_codes"],
            case["weights"], case["targets"],
        )


def test_the_report_carries_what_the_audit_needs() -> None:
    case = _well_posed()
    seen: list[dict] = []
    calibration_guard.checked(_calibrate, record=seen.append)(
        case["values"], case["candidate_codes"], case["region_codes"],
        case["weights"], case["targets"],
    )
    assert len(seen) == 1
    report = seen[0]
    for key in (
        "iteration_budget", "max_candidate_residual", "max_region_sum_residual",
        "tolerance", "all_values_finite", "all_values_valid_shares",
        "converged", "numerical_impact_bound_pp",
    ):
        assert key in report, f"the audit record lacks {key}"
    assert report["converged"] is True
    assert report["tolerance"] == calibration_guard.CALIBRATION_ABS_TOL
    assert report["numerical_impact_bound_pp"] == 1e-6


def test_the_tolerance_is_an_accuracy_contract_not_a_fitted_plateau() -> None:
    """1e-8 share is 1e-6 percentage points, chosen for the bound it buys."""

    assert calibration_guard.CALIBRATION_ABS_TOL == 1e-8
    assert calibration_guard.NUMERICAL_IMPACT_BOUND_PP == 1e-6
    # the observed plateau sits inside it rather than defining it
    assert 3.8e-9 < calibration_guard.CALIBRATION_ABS_TOL


def test_a_broken_region_composition_is_refused_even_if_levels_hold() -> None:
    """Conserving candidate levels while breaking the regional sums is not success."""

    case = _well_posed()
    report = calibration_guard.inspect(
        np.array([0.55, 0.45, 0.40, 0.60, 0.70, 0.90]),
        case["candidate_codes"], case["region_codes"],
        case["weights"], case["targets"], iteration_budget=200,
    )
    assert report["max_region_sum_residual"] > calibration_guard.CALIBRATION_ABS_TOL
    assert report["converged"] is False
