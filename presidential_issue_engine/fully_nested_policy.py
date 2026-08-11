"""Outcome-safe selection for ordered presidential pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class PipelineStage:
    name: str
    complexity: int


ORDERED_STAGES = (
    PipelineStage("strict_base", 0),
    PipelineStage("structural", 1),
    PipelineStage("structural_mega", 2),
    PipelineStage("structural_mega_shock", 3),
    PipelineStage("structural_mega_shock_regime", 4),
)


def select_stage_from_prior_folds(
    target_election: str,
    election_order: Sequence[str],
    by_election: pd.DataFrame,
    *,
    minimum_selection_elections: int = 2,
    metric_column: str = "regional_weighted_mae_pp",
    stages: Sequence[PipelineStage] = ORDERED_STAGES,
) -> tuple[str, dict[str, float], tuple[str, ...]]:
    """Select a stage without reading the target election's loss."""

    if target_election not in election_order:
        raise ValueError(f"unknown target election: {target_election}")
    target_index = election_order.index(target_election)
    prior = tuple(election_order[:target_index])
    if len(prior) < minimum_selection_elections:
        return stages[0].name, {}, prior

    required = {"variant", "election_id", metric_column}
    missing = required - set(by_election.columns)
    if missing:
        raise ValueError(f"selection table missing columns: {sorted(missing)}")

    losses: dict[str, float] = {}
    for stage in stages:
        rows = by_election.loc[
            by_election["variant"].eq(stage.name)
            & by_election["election_id"].isin(prior),
            metric_column,
        ]
        if len(rows) != len(prior):
            raise ValueError(
                f"incomplete prior-fold losses for {target_election}/{stage.name}"
            )
        losses[stage.name] = float(pd.to_numeric(rows, errors="raise").mean())

    selected = min(stages, key=lambda stage: (losses[stage.name], stage.complexity))
    return selected.name, losses, prior


def deployment_stage_from_completed_folds(
    by_election: pd.DataFrame,
    election_order: Sequence[str],
    *,
    metric_column: str = "regional_weighted_mae_pp",
    stages: Sequence[PipelineStage] = ORDERED_STAGES,
) -> tuple[str, dict[str, float]]:
    """Choose the future deployment stage after every scored fold is frozen."""

    losses: dict[str, float] = {}
    for stage in stages:
        rows = by_election.loc[
            by_election["variant"].eq(stage.name)
            & by_election["election_id"].isin(election_order),
            metric_column,
        ]
        if len(rows) != len(election_order):
            raise ValueError(f"incomplete deployment losses for {stage.name}")
        losses[stage.name] = float(pd.to_numeric(rows, errors="raise").mean())
    selected = min(stages, key=lambda stage: (losses[stage.name], stage.complexity))
    return selected.name, losses
