from __future__ import annotations

import pandas as pd

from presidential_issue_engine import fully_nested_policy as policy


ELECTIONS = ("e1", "e2", "e3", "e4")


def _losses() -> pd.DataFrame:
    rows = []
    for stage in policy.ORDERED_STAGES:
        for index, election in enumerate(ELECTIONS):
            rows.append(
                {
                    "variant": stage.name,
                    "election_id": election,
                    "regional_weighted_mae_pp": 10.0 - stage.complexity + index,
                }
            )
    return pd.DataFrame(rows)


def test_early_targets_use_simple_fallback() -> None:
    frame = _losses()
    selected, losses, prior = policy.select_stage_from_prior_folds(
        "e2", ELECTIONS, frame, minimum_selection_elections=2
    )
    assert selected == "strict_base"
    assert losses == {}
    assert prior == ("e1",)


def test_selection_uses_only_prior_outer_folds() -> None:
    frame = _losses()
    expected, _, prior = policy.select_stage_from_prior_folds("e4", ELECTIONS, frame)
    mutated = frame.copy()
    mutated.loc[mutated["election_id"].eq("e4"), "regional_weighted_mae_pp"] = -999.0
    observed, _, observed_prior = policy.select_stage_from_prior_folds(
        "e4", ELECTIONS, mutated
    )
    assert expected == observed == "structural_mega_shock_regime"
    assert prior == observed_prior == ("e1", "e2", "e3")


def test_deployment_selection_uses_all_completed_folds() -> None:
    selected, losses = policy.deployment_stage_from_completed_folds(
        _losses(), ELECTIONS
    )
    assert selected == "structural_mega_shock_regime"
    assert set(losses) == {stage.name for stage in policy.ORDERED_STAGES}
