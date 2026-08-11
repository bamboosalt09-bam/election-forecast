"""Strict nested ablation of electorate-layer response mechanisms.

Each profile uses the same single bounded separation parameter. Profiles are
evaluated independently; the target election never selects its own gain or
separation. ``critical_defection`` is the predeclared primary hypothesis.
"""

from __future__ import annotations

import json
import math
import sys
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
PROFILES = (
    "critical_defection",
    "core_rigidity",
    "swing_mobility",
    "critical_swing",
    "combined",
)
PRIMARY_PROFILE = "critical_defection"
SEPARATION_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_TUNING_ELECTIONS = 2
MAX_PRIOR_WORSENING_PP = 0.05
OUTPUT_DIR = ROOT / "outputs" / "electorate_layer_profile_experiment"


def _by_election(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    return base_eval.election_weighted_mae(frame, prediction_column).set_index(
        "election_id"
    )["weighted_row_mae_pp"]


def _select_separation(
    frame: pd.DataFrame,
    tuning_elections: tuple[str, ...],
    preference_gain: float,
    profile: str,
    *,
    selection_label: str,
    metric_cache: dict[tuple[str, float, float], pd.Series],
) -> tuple[float, list[dict[str, object]]]:
    if len(tuning_elections) < MIN_TUNING_ELECTIONS or preference_gain <= 0.0:
        return 0.0, []
    baseline_key = (profile, float(preference_gain), 0.0)
    if baseline_key not in metric_cache:
        evaluated = base_eval.apply_config(
            frame,
            layers.ElectorateLayerConfig(preference_gain=preference_gain),
        )
        metric_cache[baseline_key] = _by_election(evaluated, "layer_pred")
    baseline = metric_cache[baseline_key].loc[list(tuning_elections)]
    required_improved = math.ceil(len(tuning_elections) / 2.0)
    candidates: list[tuple[float, float]] = []
    trace: list[dict[str, object]] = []
    for separation in SEPARATION_GRID:
        key = (profile, float(preference_gain), float(separation))
        if key not in metric_cache:
            evaluated = base_eval.apply_config(
                frame,
                layers.ElectorateLayerConfig(
                    preference_gain=preference_gain,
                    layer_separation=separation,
                    layer_response_profile=profile,
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
                "profile": profile,
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


def _evaluate_profile(
    frame: pd.DataFrame,
    profile: str,
    gain_metric_cache: dict[float, pd.Series],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shape_metric_cache: dict[tuple[str, float, float], pd.Series] = {}
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
        separation, trace = _select_separation(
            frame,
            prior,
            gain,
            profile,
            selection_label=f"{profile}_outer_{target}",
            metric_cache=shape_metric_cache,
        )
        traces.extend(trace)
        target_frame = frame.loc[frame["election_id"].eq(target)].copy()
        baseline_parts.append(
            base_eval.apply_config(
                target_frame,
                layers.ElectorateLayerConfig(preference_gain=gain),
            )
        )
        candidate_parts.append(
            base_eval.apply_config(
                target_frame,
                layers.ElectorateLayerConfig(
                    preference_gain=gain,
                    layer_separation=separation,
                    layer_response_profile=profile,
                ),
            )
        )
        configs.append(
            {
                "profile": profile,
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
            "profile": profile,
            "election_id": list(ELECTIONS),
            "existing_nested_mae_pp": baseline_by.loc[list(ELECTIONS)].to_numpy(),
            "profile_nested_mae_pp": candidate_by.loc[list(ELECTIONS)].to_numpy(),
        }
    )
    comparison["improvement_pp"] = (
        comparison["existing_nested_mae_pp"] - comparison["profile_nested_mae_pp"]
    )
    baseline_macro = float(comparison["existing_nested_mae_pp"].mean())
    candidate_macro = float(comparison["profile_nested_mae_pp"].mean())
    improvement = baseline_macro - candidate_macro
    max_worsening = float((-comparison["improvement_pp"]).clip(lower=0.0).max())

    final_gain, _ = gain_eval.select_preference_gain(
        frame,
        ELECTIONS,
        selection_label="gain_future_through2022",
        gain_grid=gain_eval.CAPPED_GAIN_GRID,
        metric_cache=gain_metric_cache,
    )
    final_separation, final_trace = _select_separation(
        frame,
        ELECTIONS,
        final_gain,
        profile,
        selection_label=f"{profile}_future_through2022",
        metric_cache=shape_metric_cache,
    )
    traces.extend(final_trace)
    result_2022 = comparison.loc[comparison["election_id"].eq("pres_2022")].iloc[0]
    gates = {
        "strict_nested_improvement_at_least_0_01pp": improvement >= 0.01,
        "no_outer_election_worsens_more_than_0_05pp": max_worsening <= 0.05,
        "pres_2022_worsening_at_most_0_05pp": float(-result_2022["improvement_pp"]) <= 0.05,
        "every_target_excluded_from_tuning": all(
            bool(row["target_excluded_from_tuning"]) for row in configs
        ),
        "at_least_two_outer_folds_select_nonzero_separation": sum(
            float(row["layer_separation"]) > 0.0 for row in configs
        ) >= 2,
    }
    result = {
        "profile": profile,
        "predeclared_primary": profile == PRIMARY_PROFILE,
        "existing_nested_weighted_macro_mae_pp": baseline_macro,
        "profile_nested_weighted_macro_mae_pp": candidate_macro,
        "strict_nested_improvement_pp": improvement,
        "maximum_outer_worsening_pp": max_worsening,
        "future_preference_gain": float(final_gain),
        "future_layer_separation": float(final_separation),
        "adoption_gates": gates,
        "passes_all_gates": bool(all(gates.values())),
    }
    return result, comparison, pd.DataFrame(configs), pd.DataFrame(traces)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = base_eval.prepare_frame()
    gain_metric_cache: dict[float, pd.Series] = {}
    results: list[dict[str, object]] = []
    comparisons: list[pd.DataFrame] = []
    configs: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    for profile in PROFILES:
        result, comparison, profile_configs, profile_traces = _evaluate_profile(
            frame, profile, gain_metric_cache
        )
        results.append(result)
        comparisons.append(comparison)
        configs.append(profile_configs)
        traces.append(profile_traces)

    primary = next(row for row in results if row["profile"] == PRIMARY_PROFILE)
    payload = {
        "scope": {
            "scored_elections": list(ELECTIONS),
            "post_2022_presidential_outcomes_loaded": False,
            "selection_unit": "earlier frozen outer-fold election predictions only",
            "metric": "contest-vote weighted row MAE within election, equal-election macro",
            "profiles_are_independent_ablations": True,
        },
        "predeclared_primary_profile": PRIMARY_PROFILE,
        "separation_grid": list(SEPARATION_GRID),
        "profile_results": results,
        "adopt_primary_into_active_engine": bool(primary["passes_all_gates"]),
        "interpretation_rule": (
            "Diagnostic profiles do not become active merely because the best one was observed; "
            "only the predeclared critical-defection hypothesis can pass the adoption gate."
        ),
    }
    pd.DataFrame(results).drop(columns=["adoption_gates"]).to_csv(
        OUTPUT_DIR / "profile_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(comparisons, ignore_index=True).to_csv(
        OUTPUT_DIR / "election_comparison.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(configs, ignore_index=True).to_csv(
        OUTPUT_DIR / "outer_selected_configs.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(traces, ignore_index=True).to_csv(
        OUTPUT_DIR / "selection_trace.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
