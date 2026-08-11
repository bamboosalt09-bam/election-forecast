from __future__ import annotations

import sys
from pathlib import Path

import pytest


ENGINE_DIR = Path(__file__).resolve().parents[1] / "presidential_issue_engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import issue_vote_engine as engine  # noqa: E402
import robustness_check as robustness  # noqa: E402


def test_rolling_report_rows_match_engine_metric() -> None:
    frame = robustness.competition_frame(engine.assemble())
    warmup = robustness.rolling_warmup_frame(frame)

    metric, _ = engine.rolling_origin_cv(
        frame,
        engine.PREDICTORS,
        alpha=engine.RIDGE_ALPHA,
        election_order=robustness.COMPETITION_ELECTIONS,
        warmup=warmup,
        warmup_order=robustness.ROLLING_WARMUP_ELECTIONS,
    )
    rows = robustness.rolling_origin_error_frame(
        frame,
        engine.PREDICTORS,
        alpha=engine.RIDGE_ALPHA,
        warmup=warmup,
        warmup_order=robustness.ROLLING_WARMUP_ELECTIONS,
    )

    assert rows["abs_err_pp"].mean() == pytest.approx(metric, abs=1e-10)
