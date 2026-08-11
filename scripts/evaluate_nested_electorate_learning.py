"""Strict chronological nested learning for the electorate preference gain.

Each outer target selects its gain using only already available outer-fold
predictions from earlier scored elections. The target election is never part of
its own gain selection. The learner is deliberately one-dimensional because
five scored presidential elections cannot support issue- or layer-specific
free parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import electorate_layers as layers  # noqa: E402
import evaluate_electorate_layers as base_eval  # noqa: E402


ELECTIONS = base_eval.ALLOWED_ELECTIONS
INITIAL_GAIN_MAX = 0.04
GAIN_STEP = 0.005
GAIN_EXPANSION_FACTOR = 2.0
MAX_NUMERICAL_EXPANSIONS = 12
SEARCH_TOLERANCE_PP = 1e-8
EXPANSION_POINTS = 8
DECLARED_GAIN_CAP = 0.04
CAPPED_GAIN_GRID = tuple(
    round(float(value), 9)
    for value in np.arange(0.0, DECLARED_GAIN_CAP + GAIN_STEP / 2.0, GAIN_STEP)
)
MIN_TUNING_ELECTIONS = 2
MAX_PRIOR_ELECTION_WORSENING_PP = 0.05
OUTPUT_DIR = ROOT / "outputs" / "electorate_nested_learning"


def _by_election(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    result = base_eval.election_weighted_mae(frame, prediction_column)
    return result.set_index("election_id")["weighted_row_mae_pp"]


def select_preference_gain(
    frame: pd.DataFrame,
    tuning_elections: tuple[str, ...],
    *,
    selection_label: str,
    gain_grid: tuple[float, ...] | None = None,
    metric_cache: dict[float, pd.Series] | None = None,
) -> tuple[float, list[dict[str, object]]]:
    """Select one gain from prior out-of-fold election predictions.

    With no explicit grid, the search starts near zero and doubles its range
    while the best eligible value remains on the search boundary. This avoids
    treating a hand-set maximum as a statistical constraint. The numerical
    expansion guard is only a termination safeguard; the response function
    normally reaches an interior optimum or its fixed log-shift saturation
    first.
    """

    if len(tuning_elections) < MIN_TUNING_ELECTIONS:
        return 0.0, []
    tuning_frame = frame.loc[frame["election_id"].isin(tuning_elections)].copy()
    baseline = _by_election(tuning_frame, "pred").loc[list(tuning_elections)]
    required_improved = math.ceil(len(tuning_elections) / 2.0)
    candidates: list[tuple[float, float]] = []
    trace: list[dict[str, object]] = []
    evaluated_gains: set[float] = set()
    explicit_grid = gain_grid is not None
    current_max = INITIAL_GAIN_MAX
    stop_reason = "explicit_grid"
    expansion = 0
    previous_max = 0.0
    while True:
        if explicit_grid:
            round_grid = tuple(sorted(set(float(value) for value in gain_grid or ())))
        elif expansion == 0:
            round_grid = tuple(
                round(float(value), 9)
                for value in np.arange(0.0, current_max + GAIN_STEP / 2.0, GAIN_STEP)
            )
        else:
            # Keep the number of evaluations bounded per expansion. A fixed
            # fine step over a geometrically growing range would turn an
            # uncapped search into tens of thousands of redundant fits.
            round_grid = tuple(
                round(float(value), 9)
                for value in np.linspace(
                    previous_max,
                    current_max,
                    EXPANSION_POINTS + 1,
                )[1:]
            )
        for gain in round_grid:
            if gain in evaluated_gains:
                continue
            evaluated_gains.add(gain)
            if metric_cache is not None and gain in metric_cache:
                candidate = metric_cache[gain].loc[list(tuning_elections)]
            else:
                config = layers.ElectorateLayerConfig(preference_gain=float(gain))
                evaluation_frame = frame if metric_cache is not None else tuning_frame
                evaluated = base_eval.apply_config(evaluation_frame, config)
                all_metrics = _by_election(evaluated, "layer_pred")
                if metric_cache is not None:
                    metric_cache[gain] = all_metrics
                candidate = all_metrics.loc[list(tuning_elections)]
            improvement = baseline - candidate
            improved_count = int((improvement > 1e-12).sum())
            max_worsening = float((-improvement).clip(lower=0.0).max())
            safe = bool(
                gain == 0.0
                or (
                    improved_count >= required_improved
                    and max_worsening <= MAX_PRIOR_ELECTION_WORSENING_PP
                )
            )
            macro = float(candidate.mean())
            trace.append(
                {
                    "selection_label": selection_label,
                    "tuning_elections": "|".join(tuning_elections),
                    "gain": float(gain),
                    "weighted_macro_mae_pp": macro,
                    "improvement_pp": float(baseline.mean() - macro),
                    "improved_elections": improved_count,
                    "required_improved_elections": required_improved,
                    "maximum_prior_worsening_pp": max_worsening,
                    "eligible": safe,
                    "search_expansion": expansion,
                }
            )
            if safe:
                candidates.append((macro, float(gain)))

        if explicit_grid or not candidates:
            stop_reason = "explicit_grid" if explicit_grid else "no_eligible_candidate"
            break
        _, selected_gain = min(candidates, key=lambda item: (item[0], item[1]))
        boundary = max(evaluated_gains)
        tail = sorted(
            (macro, gain) for macro, gain in candidates if gain > selected_gain
        )
        evaluated_above_selected = any(gain > selected_gain for gain in evaluated_gains)
        best_macro = min(macro for macro, _ in candidates)
        interior_confirmed = bool(
            selected_gain <= boundary - 2.0 * GAIN_STEP
            and tail
            and min(macro for macro, _ in tail) >= best_macro - SEARCH_TOLERANCE_PP
        )
        if interior_confirmed:
            stop_reason = "interior_optimum_confirmed"
            break
        if evaluated_above_selected and not tail:
            stop_reason = "constrained_optimum_confirmed"
            break
        expansion += 1
        if expansion >= MAX_NUMERICAL_EXPANSIONS:
            stop_reason = "numerical_expansion_guard"
            break
        previous_max = current_max
        current_max *= GAIN_EXPANSION_FACTOR
    if not candidates:
        return 0.0, trace
    _, selected_gain = min(candidates, key=lambda item: (item[0], item[1]))
    if trace:
        trace[-1]["search_stop_reason"] = stop_reason
        trace[-1]["search_converged"] = stop_reason != "numerical_expansion_guard"
    return selected_gain, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uncapped",
        action="store_true",
        help="Run the rejected adaptive-range diagnostic instead of the capped learner.",
    )
    args = parser.parse_args()
    output_dir = (
        ROOT / "outputs" / "electorate_nested_learning_uncapped"
        if args.uncapped
        else OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_grid = None if args.uncapped else CAPPED_GAIN_GRID
    frame = base_eval.prepare_frame()
    baseline_by = _by_election(frame, "pred")

    outer_parts: list[pd.DataFrame] = []
    configs: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    metric_cache: dict[float, pd.Series] = {}
    leakage_safe = True
    for index, target in enumerate(ELECTIONS):
        prior = ELECTIONS[:index]
        leakage_safe = leakage_safe and target not in prior
        gain, trace = select_preference_gain(
            frame,
            prior,
            selection_label=f"outer_{target}",
            gain_grid=selection_grid,
            metric_cache=metric_cache,
        )
        traces.extend(trace)
        target_frame = frame.loc[frame["election_id"].eq(target)].copy()
        adjusted = base_eval.apply_config(
            target_frame,
            layers.ElectorateLayerConfig(preference_gain=gain),
        )
        outer_parts.append(adjusted)
        target_mae = float(_by_election(adjusted, "layer_pred").loc[target])
        configs.append(
            {
                "target_election": target,
                "tuning_elections": "|".join(prior),
                "preference_gain": gain,
                "outer_weighted_row_mae_pp": target_mae,
                "target_excluded_from_tuning": target not in prior,
            }
        )

    nested = pd.concat(outer_parts, ignore_index=True)
    nested_by = _by_election(nested, "layer_pred")
    comparison = pd.DataFrame(
        {
            "election_id": list(ELECTIONS),
            "baseline_weighted_row_mae_pp": baseline_by.loc[list(ELECTIONS)].to_numpy(),
            "learned_weighted_row_mae_pp": nested_by.loc[list(ELECTIONS)].to_numpy(),
        }
    )
    comparison["improvement_pp"] = (
        comparison["baseline_weighted_row_mae_pp"]
        - comparison["learned_weighted_row_mae_pp"]
    )
    baseline_macro = float(comparison["baseline_weighted_row_mae_pp"].mean())
    nested_macro = float(comparison["learned_weighted_row_mae_pp"].mean())
    improvement = baseline_macro - nested_macro
    max_worsening = float((-comparison["improvement_pp"]).clip(lower=0.0).max())
    result_2022 = comparison.loc[comparison["election_id"].eq("pres_2022")].iloc[0]

    final_gain, final_trace = select_preference_gain(
        frame,
        ELECTIONS,
        selection_label="future_deployment_through2022",
        gain_grid=selection_grid,
        metric_cache=metric_cache,
    )
    traces.extend(final_trace)
    gates = {
        "strict_nested_macro_improvement_at_least_0_01pp": improvement >= 0.01,
        "no_outer_election_worsens_more_than_0_05pp": max_worsening <= 0.05,
        "pres_2022_worsening_at_most_0_05pp": (
            float(-result_2022["improvement_pp"]) <= 0.05
        ),
        "every_target_excluded_from_its_tuning_set": leakage_safe,
        "at_least_two_outer_folds_learn_nonzero_gain": (
            sum(float(row["preference_gain"]) > 0.0 for row in configs) >= 2
        ),
    }
    if args.uncapped:
        gates["future_deployment_search_converged_without_manual_cap"] = bool(
            final_trace and final_trace[-1].get("search_converged", False)
        )
    else:
        gates["future_deployment_gain_within_declared_cap"] = bool(
            0.0 <= final_gain <= DECLARED_GAIN_CAP
        )
    payload = {
        "scope": {
            "run_mode": "uncapped_diagnostic" if args.uncapped else "capped_candidate",
            "scored_elections": list(ELECTIONS),
            "post_2022_presidential_outcomes_loaded": False,
            "selection_unit": "earlier frozen outer-fold election predictions only",
            "metric": "contest-vote weighted row MAE within election, equal-election macro",
        },
        "learner": {
            "search": (
                "adaptive expansion from zero until an interior optimum is confirmed"
                if args.uncapped
                else "fixed fine grid within a declared conservative response cap"
            ),
            "declared_gain_cap": None if args.uncapped else DECLARED_GAIN_CAP,
            "initial_gain_max": INITIAL_GAIN_MAX,
            "gain_step": GAIN_STEP,
            "gain_expansion_factor": GAIN_EXPANSION_FACTOR,
            "points_per_expanded_range": EXPANSION_POINTS,
            "numerical_expansion_guard": MAX_NUMERICAL_EXPANSIONS,
            "minimum_tuning_elections": MIN_TUNING_ELECTIONS,
            "maximum_prior_election_worsening_pp": MAX_PRIOR_ELECTION_WORSENING_PP,
            "free_parameters": 1,
        },
        "baseline_weighted_macro_mae_pp": baseline_macro,
        "strict_nested_learned_weighted_macro_mae_pp": nested_macro,
        "strict_nested_improvement_pp": improvement,
        "outer_configs": configs,
        "future_deployment_through2022": asdict(
            layers.ElectorateLayerConfig(
                preference_gain=final_gain,
                mass_profile="direct_party_layers",
            )
        ),
        "adoption_gates": gates,
        "adopt_into_active_engine": bool(all(gates.values())),
        "caveat": (
            "The selection algorithm was designed after prior 2022 diagnostics; "
            "the computation is leakage-safe but 2022 is not an untouched holdout."
        ),
    }

    nested.to_csv(output_dir / "nested_predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "nested_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(configs).to_csv(
        output_dir / "outer_selected_gains.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(traces).to_csv(
        output_dir / "selection_trace.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"baseline weighted macro MAE: {baseline_macro:.6f}%p")
    print(f"strict nested learned macro MAE: {nested_macro:.6f}%p")
    print(f"strict nested improvement: {improvement:+.6f}%p")
    print(f"future deployment gain: {final_gain:.3f}")
    print(f"adoption gates: {gates}")
    print(f"adopt into active engine: {all(gates.values())}")


if __name__ == "__main__":
    main()
