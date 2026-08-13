"""V24-only alternatives for isolated defect ablations.

Nothing in this module is imported by the frozen V23 runner.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from presidential_issue_engine import automatic_contest_response
from presidential_issue_engine import rejection_beneficiary_routing


def apply_prior_selected_response_with_fixed_caps(
    frame: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    prediction_column: str,
    apply_response: Callable[..., pd.DataFrame],
    election_order: Sequence[str],
    slot_column: str = "source_slot",
    output_column: str | None = None,
    gain_grid: Sequence[float] = automatic_contest_response.DEFAULT_GAIN_GRID,
    default_gain: float = 0.50,
    prior_strength: float = 1.0,
    critical_elasticity: float = 0.75,
    swing_elasticity: float = 1.25,
    log_shift_cap: float = 0.40,
    swing_log_shift_cap: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select expansion gain from prior folds while enforcing fixed config caps."""

    output_column = output_column or prediction_column
    order = [str(value) for value in election_order]
    order_lookup = {election_id: index for index, election_id in enumerate(order)}
    work = frame.copy().reset_index(drop=True)
    work["_automatic_response_order"] = np.arange(len(work))
    grid = tuple(sorted({float(value) for value in gain_grid}))
    if not grid:
        raise ValueError("gain_grid must contain at least one value")

    regime_lookup = regimes.drop_duplicates("election_id").set_index("election_id")
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for target in order:
        target_index = order_lookup[target]
        prior_ids = order[:target_index]
        responsive_prior = [
            election_id
            for election_id in prior_ids
            if election_id in regime_lookup.index
            and float(regime_lookup.at[election_id, "dominance_activation"]) > 0.0
        ]
        losses: dict[float, float] = {}
        if responsive_prior:
            prior_frame = work.loc[work["election_id"].isin(prior_ids)].copy()
            prior_regimes = regimes.loc[regimes["election_id"].isin(prior_ids)].copy()
            for gain in grid:
                evaluated = apply_response(
                    prior_frame,
                    prior_regimes,
                    prediction_column=prediction_column,
                    slot_column=slot_column,
                    output_column=output_column,
                    expansion_gain=gain,
                    log_shift_cap=log_shift_cap,
                    critical_elasticity=critical_elasticity,
                    swing_elasticity=swing_elasticity,
                    swing_log_shift_cap=swing_log_shift_cap,
                )
                evaluated, _ = rejection_beneficiary_routing.apply_rejection_beneficiary_routing(
                    evaluated,
                    prior_regimes,
                    prediction_column=output_column,
                    slot_column=slot_column,
                )
                losses[gain] = automatic_contest_response._macro_regional_mae(
                    evaluated, output_column
                )
            best_gain = min(grid, key=lambda value: (losses[value], value))
        else:
            best_gain = float(default_gain)

        responsive_count = len(responsive_prior)
        reliability = responsive_count / (
            responsive_count + max(float(prior_strength), 1e-6)
        )
        selected_gain = float(
            np.clip(
                default_gain + reliability * (best_gain - default_gain),
                min(grid),
                max(grid),
            )
        )
        target_frame = work.loc[work["election_id"].eq(target)].copy()
        target_regimes = regimes.loc[regimes["election_id"].eq(target)].copy()
        evaluated_target = apply_response(
            target_frame,
            target_regimes,
            prediction_column=prediction_column,
            slot_column=slot_column,
            output_column=output_column,
            expansion_gain=selected_gain,
            log_shift_cap=log_shift_cap,
            critical_elasticity=critical_elasticity,
            swing_elasticity=swing_elasticity,
            swing_log_shift_cap=swing_log_shift_cap,
        )
        evaluated_target["automatic_contest_response_gain"] = selected_gain
        parts.append(evaluated_target)
        audit_rows.append(
            {
                "target_election": target,
                "selection_training_elections": "|".join(prior_ids),
                "responsive_training_elections": "|".join(responsive_prior),
                "responsive_training_count": responsive_count,
                "default_gain": float(default_gain),
                "raw_selected_gain": float(best_gain),
                "selection_reliability": float(reliability),
                "selected_gain": selected_gain,
                "fixed_log_shift_cap": float(log_shift_cap),
                "fixed_swing_log_shift_cap": float(swing_log_shift_cap),
                "target_excluded_from_selection": target not in prior_ids,
                **{
                    f"prior_macro_mae_gain_{gain:.2f}": losses.get(gain, np.nan)
                    for gain in grid
                },
            }
        )
    output = pd.concat(parts, ignore_index=True).sort_values(
        "_automatic_response_order"
    )
    return output.drop(columns="_automatic_response_order"), pd.DataFrame(audit_rows)
