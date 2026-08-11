"""Prior-only selection of contest-regime response strength."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from presidential_issue_engine import rejection_beneficiary_routing


DEFAULT_GAIN_GRID = (0.40, 0.50, 0.60, 0.70)


def _macro_regional_mae(frame: pd.DataFrame, prediction_column: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("election_id", sort=False):
        error = (
            pd.to_numeric(group[prediction_column], errors="coerce")
            - pd.to_numeric(group["actual"], errors="coerce")
        ).abs() * 100.0
        weights = pd.to_numeric(
            group["contest_votes"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        if float(weights.sum()) > 0.0:
            values.append(float(np.average(error, weights=weights)))
        else:
            values.append(float(error.mean()))
    return float(np.mean(values)) if values else float("nan")


def apply_prior_selected_contest_response(
    frame: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    prediction_column: str,
    apply_response: Callable[..., pd.DataFrame],
    election_order: Sequence[str],
    slot_column: str = "source_slot",
    output_column: str | None = None,
    gain_grid: Sequence[float] = DEFAULT_GAIN_GRID,
    default_gain: float = 0.50,
    prior_strength: float = 1.0,
    critical_elasticity: float = 0.75,
    swing_elasticity: float = 1.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select each target gain from earlier folds and shrink sparse selections."""

    output_column = output_column or prediction_column
    order = [str(value) for value in election_order]
    order_lookup = {election_id: index for index, election_id in enumerate(order)}
    work = frame.copy().reset_index(drop=True)
    work["_automatic_response_order"] = np.arange(len(work))
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    grid = tuple(sorted({float(value) for value in gain_grid}))
    if not grid:
        raise ValueError("gain_grid must contain at least one value")

    regime_lookup = regimes.drop_duplicates("election_id").set_index("election_id")
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
                    log_shift_cap=gain,
                    critical_elasticity=critical_elasticity,
                    swing_elasticity=swing_elasticity,
                    swing_log_shift_cap=1.25 * gain,
                )
                evaluated, _ = (
                    rejection_beneficiary_routing.apply_rejection_beneficiary_routing(
                        evaluated,
                        prior_regimes,
                        prediction_column=output_column,
                        slot_column=slot_column,
                    )
                )
                losses[gain] = _macro_regional_mae(evaluated, output_column)
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
        target_regime = regimes.loc[regimes["election_id"].eq(target)].copy()
        evaluated_target = apply_response(
            target_frame,
            target_regime,
            prediction_column=prediction_column,
            slot_column=slot_column,
            output_column=output_column,
            expansion_gain=selected_gain,
            log_shift_cap=selected_gain,
            critical_elasticity=critical_elasticity,
            swing_elasticity=swing_elasticity,
            swing_log_shift_cap=1.25 * selected_gain,
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
