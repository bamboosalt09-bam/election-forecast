"""Guards for the error-cancellation diagnostic.

The national candidate metric offsets signed regional errors, so a compressed
prediction can be nationally exact while being regionally wrong everywhere.
These tests pin that arithmetic and the panel fact it exposes, because the
reading of the headline depends on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import diagnose_error_cancellation as cancel


def _frame(errors: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": ["pres_test"] * len(errors),
            "region_id": [f"r{i}" for i in range(len(errors))],
            "candidate_name": ["c"] * len(errors),
            "contest_votes": [100.0] * len(errors),
            "actual": [0.5] * len(errors),
            "layer_pred": [0.5 + e for e in errors],
        }
    )


def test_opposing_errors_cancel_completely() -> None:
    """Equal and opposite regional errors give a nationally exact prediction."""

    result = cancel.cancellation(_frame([0.10, -0.10]))
    row = result.iloc[0]
    assert row["national_error_pp"] == pytest.approx(0.0, abs=1e-9)
    assert row["regional_absolute_pp"] == pytest.approx(10.0)
    assert row["cancellation"] == pytest.approx(1.0)


def test_systematic_errors_do_not_cancel() -> None:
    result = cancel.cancellation(_frame([0.10, 0.10]))
    row = result.iloc[0]
    assert row["national_error_pp"] == pytest.approx(10.0)
    assert row["cancellation"] == pytest.approx(0.0, abs=1e-9)


def test_cancellation_explains_the_national_metric_better_than_accuracy() -> None:
    """On the panel, national error tracks cancellation, not regional accuracy."""

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    predictions = root / "outputs/active_presidential_nested_v26/nested_predictions.csv"
    if not predictions.exists():
        pytest.skip("V26 predictions are not present")
    frame = pd.read_csv(predictions, encoding="utf-8-sig", low_memory=False)
    detail = cancel.cancellation(frame)
    by_election = detail.groupby("election_id").agg(
        cancellation=("cancellation", "mean"),
        national=("national_error_pp", lambda s: s.abs().mean()),
        regional=("regional_absolute_pp", "mean"),
    )
    against_cancellation = abs(by_election["cancellation"].corr(by_election["national"]))
    against_regional = abs(by_election["regional"].corr(by_election["national"]))
    assert against_cancellation > against_regional, (
        "if regional accuracy ever explained the national metric better than "
        "cancellation does, the headline would read differently"
    )
    assert against_cancellation > 0.8


def test_the_worst_national_fold_is_the_one_that_cancels_least() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    predictions = root / "outputs/active_presidential_nested_v26/nested_predictions.csv"
    if not predictions.exists():
        pytest.skip("V26 predictions are not present")
    frame = pd.read_csv(predictions, encoding="utf-8-sig", low_memory=False)
    detail = cancel.cancellation(frame)
    by_election = detail.groupby("election_id").agg(
        cancellation=("cancellation", "mean"),
        national=("national_error_pp", lambda s: s.abs().mean()),
    )
    assert by_election["national"].idxmax() == by_election["cancellation"].idxmin()
