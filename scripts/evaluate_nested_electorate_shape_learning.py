"""Strict nested experiment for core/critical/swing response separation.

The existing preference gain is selected first from prior outer predictions.
One additional bounded shape parameter is then selected from those same prior
elections. Target outcomes never participate in their own configuration.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import electorate_layers as layers  # noqa: E402
import evaluate_electorate_layers as base_eval  # noqa: E402
import evaluate_nested_electorate_learning as gain_eval  # noqa: E402


ELECTIONS = base_eval.ALLOWED_ELECTIONS
SEPARATION_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_TUNING_ELECTIONS = 2
MAX_PRIOR_WORSENING_PP = 0.05
OUTPUT_DIR = ROOT / "outputs" / "electorate_layer_shape_experiment"


def _by_election(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    return base_eval.election_weighted_mae(frame, prediction_column).set_index(
        "election_id"
    )["weighted_row_mae_pp"]


def select_layer_separation(
    frame: pd.DataFrame,
    tuning_elections: tuple[str, ...],
    preference_gain: float,
    *,
    selection_label: str,
    metric_cache: dict[tuple[float, float], pd.Series],
) -> tuple[float, list[dict[str, object]]]:
    if len(tuning_elections) < MIN_TUNING_ELECTIONS or preference_gain <= 0.0:
        return 0.0, []
    key0 = (float(preference_gain), 0.0)
    if key0 not in metric_cache:
        evaluated = base_eval.apply_config(
            frame,
            layers.ElectorateLayerConfig(preference_gain=preference_gain),
        )
        metric_cache[key0] = _by_election(evaluated, "layer_pred")
    baseline = metric_cache[key0].loc[list(tuning_elections)]
    required_improved = math.ceil(len(tuning_elections) / 2.0)
    candidates: list[tuple[float, float]] = []
    trace: list[dict[str, object]] = []
    for separation in SEPARATION_GRID:
        key = (float(preference_gain), float(separation))
        if key not in metric_cache:
            evaluated = base_eval.apply_config(
                frame,
                layers.ElectorateLayerConfig(
                    preference_gain=preference_gain,
                    layer_separation=separation,
                ),
            )
            metric_cache[key] = _by_election(evaluated, "layer_pred")
        candidate = metric_cache[key].loc[list(tuning_elections)]
        improvement = baseline - candidate
        improved_count = int((improvement > 1e-12).sum())
        max_worsening = float((-improvement).clip(lower=0.0).max())
        eligible = bool(
            separation == 0.0
            or (
                improved_count >= required_improved
                and max_worsening <= MAX_PRIOR_WORSENING_PP
            )
        )
        macro = float(candidate.mean())
        trace.append(
            {
                "selection_label": selection_label,
                "tuning_elections": "|".join(tuning_elections),
                "preference_gain": float(preference_gain),
                "layer_separation": float(separation),
                "weighted_macro_mae_pp": macro,
                "incremental_improvement_pp": float(baseline.mean() - macro),
                "improved_elections": improved_count,
                "required_improved_elections": required_improved,
                "maximum_prior_worsening_pp": max_worsening,
                "eligible": eligible,
            }
        )
        if eligible:
            candidates.append((macro, float(separation)))
    _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected, trace


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = base_eval.prepare_frame()
    gain_metric_cache: dict[float, pd.Series] = {}
    shape_metric_cache: dict[tuple[float, float], pd.Series] = {}
    baseline_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    configs: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []

    for index, target in enumerate(ELECTIONS):
        prior = ELECTIONS[:index]
        gain, _ = gain_eval.select_preference_gain(
            frame,
            prior,
            selection_label=f"gain_outer_{target}",
            gain_grid=gain_eval.CAPPED_GAIN_GRID,
            metric_cache=gain_metric_cache,
        )
        separation, trace = select_layer_separation(
            frame,
            prior,
            gain,
            selection_label=f"shape_outer_{target}",
            metric_cache=shape_metric_cache,
        )
        traces.extend(trace)
        target_frame = frame.loc[frame["election_id"].eq(target)].copy()
        baseline = base_eval.apply_config(
            target_frame,
            layers.ElectorateLayerConfig(preference_gain=gain),
        )
        candidate = base_eval.apply_config(
            target_frame,
            layers.ElectorateLayerConfig(
                preference_gain=gain,
                layer_separation=separation,
            ),
        )
        baseline_parts.append(baseline)
        candidate_parts.append(candidate)
        configs.append(
            {
                "target_election": target,
                "tuning_elections": "|".join(prior),
                "preference_gain": float(gain),
                "layer_separation": float(separation),
                "target_excluded_from_tuning": target not in prior,
            }
        )

    baseline_nested = pd.concat(baseline_parts, ignore_index=True)
    candidate_nested = pd.concat(candidate_parts, ignore_index=True)
    baseline_by = _by_election(baseline_nested, "layer_pred")
    candidate_by = _by_election(candidate_nested, "layer_pred")
    comparison = pd.DataFrame(
        {
            "election_id": list(ELECTIONS),
            "existing_nested_mae_pp": baseline_by.loc[list(ELECTIONS)].to_numpy(),
            "shape_nested_mae_pp": candidate_by.loc[list(ELECTIONS)].to_numpy(),
        }
    )
    comparison["improvement_pp"] = (
        comparison["existing_nested_mae_pp"] - comparison["shape_nested_mae_pp"]
    )
    baseline_macro = float(comparison["existing_nested_mae_pp"].mean())
    candidate_macro = float(comparison["shape_nested_mae_pp"].mean())
    improvement = baseline_macro - candidate_macro
    max_worsening = float((-comparison["improvement_pp"]).clip(lower=0.0).max())

    final_gain, _ = gain_eval.select_preference_gain(
        frame,
        ELECTIONS,
        selection_label="gain_future_through2022",
        gain_grid=gain_eval.CAPPED_GAIN_GRID,
        metric_cache=gain_metric_cache,
    )
    final_separation, final_trace = select_layer_separation(
        frame,
        ELECTIONS,
        final_gain,
        selection_label="shape_future_through2022",
        metric_cache=shape_metric_cache,
    )
    traces.extend(final_trace)
    result_2022 = comparison.loc[comparison["election_id"].eq("pres_2022")].iloc[0]
    gates = {
        "strict_nested_improvement_at_least_0_01pp": improvement >= 0.01,
        "no_outer_election_worsens_more_than_0_05pp": max_worsening <= 0.05,
        "pres_2022_worsening_at_most_0_05pp": (
            float(-result_2022["improvement_pp"]) <= 0.05
        ),
        "every_target_excluded_from_tuning": all(
            bool(row["target_excluded_from_tuning"]) for row in configs
        ),
        "at_least_two_outer_folds_select_nonzero_separation": sum(
            float(row["layer_separation"]) > 0.0 for row in configs
        )
        >= 2,
        "future_separation_within_declared_bound": 0.0 <= final_separation <= 1.0,
    }
    payload = {
        "scope": {
            "scored_elections": list(ELECTIONS),
            "post_2022_presidential_outcomes_loaded": False,
            "selection_order": "preference_gain_then_layer_separation",
            "selection_unit": "earlier frozen outer-fold election predictions only",
            "metric": "contest-vote weighted row MAE within election, equal-election macro",
        },
        "response_shape": {
            "free_parameters_added": 1,
            "layer_separation_grid": list(SEPARATION_GRID),
            "core_scale": "1 - 0.50 * separation",
            "critical_negative_scale": "1 + separation",
            "critical_positive_scale": "1 + 0.15 * separation",
            "swing_scale": "1 + 0.50 * separation",
        },
        "existing_nested_weighted_macro_mae_pp": baseline_macro,
        "shape_nested_weighted_macro_mae_pp": candidate_macro,
        "strict_nested_improvement_pp": improvement,
        "outer_configs": configs,
        "future_deployment_candidate": asdict(
            layers.ElectorateLayerConfig(
                preference_gain=final_gain,
                layer_separation=final_separation,
            )
        ),
        "adoption_gates": gates,
        "adopt_into_active_engine": bool(all(gates.values())),
        "caveat": (
            "The response shape is theory constrained but the scored sample contains only five "
            "presidential elections; adoption requires strict nested transfer."
        ),
    }
    baseline_nested.to_csv(
        OUTPUT_DIR / "existing_nested_predictions.csv", index=False, encoding="utf-8-sig"
    )
    candidate_nested.to_csv(
        OUTPUT_DIR / "shape_nested_predictions.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(OUTPUT_DIR / "nested_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(configs).to_csv(
        OUTPUT_DIR / "outer_selected_configs.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(traces).to_csv(
        OUTPUT_DIR / "selection_trace.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
