"""An acceptance contract for the regional dispersion calibration.

``party_regionalism_dispersion._calibrate`` alternates two constraints and stops
early when the candidate residual falls under ``1e-11``::

    for _ in range(iterations):
        ...
        if np.max(np.abs(current - targets)) < 1e-11:
            break
    return out

If that tolerance is never met the loop runs out and returns its last iterate.
Nothing checks it and nothing records it, so a near-miss is indistinguishable
from convergence in the artifact and downstream.

What was measured
-----------------

On the scored panel, three of the five calibration calls never meet ``1e-11``.
They settle at a residual plateau between ``1.9e-9`` and ``3.8e-9``, and
raising the iteration budget from 200 to 20,000 does not reduce it by a single
digit - the same value comes back. So the ``1e-11`` termination condition was
far stricter than the numerical fixed point this implementation actually
reaches, and the loop was in practice always exhausting its budget.

This is deliberately *not* claimed to be a float64 floor. Machine epsilon is
about ``2.22e-16`` and summing 51 rows does not by itself produce a ``1e-9``
plateau; the cause is more likely the finite-precision fixed point of the
alternation, or a small incompatibility between how the two constraints are
applied. That distinction is not resolved here, and the tolerance below does
not depend on resolving it.

The tolerance
-------------

``1e-8`` in share units, which is ``1e-6`` percentage points on a published
figure. It is an accuracy contract, not a number fitted to the observed
plateau: a numerical reconciliation is accepted when it does not deform a
prediction by more than a millionth of a percentage point. The observed
plateau of ``3.7e-9`` sits comfortably inside it, but the tolerance would be
the same had the plateau been elsewhere.

There is one acceptance tolerance rather than a separate convergence criterion
and failure criterion, and **running the full iteration budget is not itself a
success condition**. Well-posed input meets the tolerance in a few rounds;
input that does not meet it after the budget fails.

Three invariants are checked together, because conserving candidate levels
while breaking the regional composition is not a success either:

- worst candidate-level residual within tolerance;
- worst region-sum residual within tolerance;
- every value finite and a valid share.

Why a wrapper
-------------

``party_regionalism_dispersion`` is pinned by hash in the V30 and V31
finalization manifests. Wrapping leaves those artifacts reproducing while the
active version gets the contract. A converged run is returned unchanged.

One limit of wrapping: the number of iterations actually used is internal to
the wrapped function and cannot be observed from outside, so the audit records
the budget it was given. Moving the tolerance into the loop itself - which
would also let it stop early - belongs to whichever version can edit that
module under its own manifest.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

#: Share units. 1e-8 share = 1e-6 percentage points on a published figure.
CALIBRATION_ABS_TOL = 1e-8
#: The bound the tolerance buys, stated in the unit a reader sees.
NUMERICAL_IMPACT_BOUND_PP = CALIBRATION_ABS_TOL * 100.0


class CalibrationDidNotConverge(RuntimeError):
    """The calibration did not meet the acceptance tolerance."""


def candidate_residual(
    values: np.ndarray,
    candidate_codes: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
) -> float:
    reached = np.bincount(
        candidate_codes, weights=values * weights, minlength=len(targets)
    )
    return float(np.max(np.abs(reached - targets)))


def region_sum_residual(values: np.ndarray, region_codes: np.ndarray) -> float:
    totals = np.bincount(region_codes, weights=values)
    present = np.bincount(region_codes) > 0
    if not present.any():
        return 0.0
    return float(np.max(np.abs(totals[present] - 1.0)))


def inspect(
    values: np.ndarray,
    candidate_codes: np.ndarray,
    region_codes: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    *,
    iteration_budget: int,
) -> dict[str, object]:
    """Everything the audit needs to explain an accept or a refusal."""

    candidate = candidate_residual(values, candidate_codes, weights, targets)
    regional = region_sum_residual(values, region_codes)
    finite = bool(np.all(np.isfinite(values)))
    valid_shares = bool(np.all(values >= 0.0) and np.all(values <= 1.0))
    converged = (
        finite
        and valid_shares
        and candidate <= CALIBRATION_ABS_TOL
        and regional <= CALIBRATION_ABS_TOL
    )
    return {
        "iteration_budget": int(iteration_budget),
        "max_candidate_residual": candidate,
        "max_region_sum_residual": regional,
        "tolerance": CALIBRATION_ABS_TOL,
        "all_values_finite": finite,
        "all_values_valid_shares": valid_shares,
        "converged": converged,
        "numerical_impact_bound_pp": NUMERICAL_IMPACT_BOUND_PP,
    }


def checked(
    original: Callable[..., np.ndarray],
    *,
    record: Callable[[dict[str, object]], None] | None = None,
) -> Callable[..., np.ndarray]:
    """Return ``original`` with the acceptance contract applied after it."""

    def calibrate(
        values: np.ndarray,
        candidate_codes: np.ndarray,
        region_codes: np.ndarray,
        weights: np.ndarray,
        targets: np.ndarray,
        iterations: int = 200,
    ) -> np.ndarray:
        out = original(values, candidate_codes, region_codes, weights, targets, iterations)
        report = inspect(
            out, candidate_codes, region_codes, weights, targets,
            iteration_budget=iterations,
        )
        if record is not None:
            record(report)
        if not report["converged"]:
            raise CalibrationDidNotConverge(
                "the regional dispersion calibration did not meet the acceptance "
                f"tolerance: candidate residual {report['max_candidate_residual']:.3e}, "
                f"region-sum residual {report['max_region_sum_residual']:.3e}, "
                f"tolerance {CALIBRATION_ABS_TOL:.3e} "
                f"(<= {NUMERICAL_IMPACT_BOUND_PP:.0e} percentage points), "
                f"finite={report['all_values_finite']}, "
                f"valid_shares={report['all_values_valid_shares']}, "
                f"iteration budget {iterations}. Running the budget is not a "
                "success condition; a reconciliation that deforms a prediction "
                "by more than the bound is refused."
            )
        return out

    calibrate.__wrapped__ = original  # type: ignore[attr-defined]
    return calibrate
