"""Freeze the separate post-election evaluation without entering model selection."""

import math

from scripts import evaluate_pres_2025_v27 as evaluation


def test_post_election_evaluation_reproduces_published_metrics():
    summary = evaluation.evaluate()
    assert summary["status"] == "post_election_evaluation_not_model_selection"
    assert summary["rows"] == 51
    assert summary["regions"] == 17
    assert summary["contest_votes"] == 34_600_675
    assert summary["all_valid_votes"] == 34_980_616
    assert math.isclose(summary["regional_contest_vote_weighted_point_mae_pp"], 4.62809566565011)
    assert math.isclose(summary["regional_equal_region_point_mae_pp"], 4.696797153834369)
    assert math.isclose(summary["national_frozen_forecast_point_mae_pp"], 4.053940545061522)


def test_evaluator_is_separate_from_model_training_code():
    source = evaluation.Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "issue_vote_engine" not in source
    assert "ridge_fit" not in source
    assert "postprocess" not in source
    assert summary_outcome_text_is_explicit()


def summary_outcome_text_is_explicit() -> bool:
    return "no fitting, tuning, selection, or forecast mutation" in evaluation.evaluate()["outcome_use"]
