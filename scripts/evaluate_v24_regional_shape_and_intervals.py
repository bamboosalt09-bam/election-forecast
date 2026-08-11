"""Evaluate V24 regional shape and hierarchical predictive intervals.

The frozen V23 files are read-only inputs.  Every regional correction uses the
latest prior election's region volumes, and every interval uses only residuals
from elections earlier than its target fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine.v24_calibration import (  # noqa: E402
    apply_national_preserving_regional_shape,
    draw_region_weight_uncertainty,
    hierarchical_residual_draws,
    prior_region_weights,
)


INPUT = ROOT / "outputs" / "active_presidential_nested_v23" / "nested_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "experiments" / "v24_regional_shape_intervals"
FINALIZATION_MANIFEST = ROOT / "outputs" / "active_presidential_nested_v23" / "finalization_manifest.json"
ACTIVE_PREDICTION_COLUMN = "layer_pred"
ORDER = ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"]
GAIN_GRID = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
NESTED_GAIN_GRID = [0.0, 0.025, 0.05]
SCALE_GRID = [0.5, 0.55, 0.6, 0.65, 0.7, 0.71, 0.72, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
COHERENT_NATIONAL_SCALE = 0.75
INTERVAL_STRUCTURES = {
    "normal_common_only": ("normal", 1.0, 0.0, 0.0),
    "normal_common_regional": ("normal", 1.0, 1.0, 0.0),
    "normal_local_shrink_025": ("normal", 1.0, 1.0, 0.25),
    "normal_local_shrink_050": ("normal", 1.0, 1.0, 0.5),
    "normal_full_hierarchy": ("normal", 1.0, 1.0, 1.0),
    "empirical_common_regional": ("empirical", 1.0, 1.0, 0.0),
    "empirical_local_shrink_050": ("empirical", 1.0, 1.0, 0.5),
    "empirical_full_hierarchy": ("empirical", 1.0, 1.0, 1.0),
}


def _region_weighted_mae(frame: pd.DataFrame, prediction_column: str) -> float:
    rows = frame.copy()
    rows["_abs"] = (rows[prediction_column] - rows["actual"]).abs() * 100.0
    per_region = rows.groupby("region_id", as_index=False).agg(
        abs_error=("_abs", "mean"),
        weight=("contest_votes", "first"),
    )
    return float(np.average(per_region["abs_error"], weights=per_region["weight"]))


def _national_mae(frame: pd.DataFrame, prediction_column: str) -> float:
    rows = frame.copy()
    weights = rows.groupby("region_id")["contest_votes"].transform("first").astype(float)
    rows["_weighted_pred"] = rows[prediction_column].astype(float) * weights
    rows["_weighted_actual"] = rows["actual"].astype(float) * weights
    national = rows.groupby("slot", as_index=False).agg(
        pred=("_weighted_pred", "sum"),
        actual=("_weighted_actual", "sum"),
    )
    denominator = float(rows[["region_id", "contest_votes"]].drop_duplicates()["contest_votes"].sum())
    return float(((national["pred"] - national["actual"]).abs() / denominator).mean() * 100.0)


def _winner_hit(frame: pd.DataFrame, prediction_column: str) -> bool:
    weights = frame.groupby("region_id")["contest_votes"].transform("first").astype(float)
    national = frame.assign(
        _pred=frame[prediction_column].astype(float) * weights,
        _actual=frame["actual"].astype(float) * weights,
    ).groupby("slot")[["_pred", "_actual"]].sum()
    return bool(national["_pred"].idxmax() == national["_actual"].idxmax())


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_frozen_input() -> str:
    manifest = json.loads(FINALIZATION_MANIFEST.read_text(encoding="utf-8"))
    expected = next(
        item["sha256"]
        for item in manifest["artifacts"]
        if item["path"] == "outputs/active_presidential_nested_v23/nested_predictions.csv"
    )
    observed = _sha256(INPUT)
    if observed != expected:
        raise RuntimeError("V23 nested prediction hash differs from its finalization manifest")
    return observed


def run(n_sim: int, seed: int) -> dict[str, object]:
    input_hash = _verify_frozen_input()
    source = pd.read_csv(INPUT, encoding="utf-8-sig")
    source = source.loc[source["election_id"].isin(ORDER)].copy()
    if set(source["election_id"]) != set(ORDER):
        raise RuntimeError("V23 experiment input does not contain the complete 2002-2022 panel")

    adjusted_by_gain: dict[float, dict[str, pd.DataFrame]] = {gain: {} for gain in GAIN_GRID}
    metric_rows: list[dict[str, object]] = []
    prior: pd.DataFrame | None = None
    for election_id in ORDER:
        target = source.loc[source["election_id"].eq(election_id)].copy()
        weights, weight_source = prior_region_weights(target, prior)
        for gain in GAIN_GRID:
            adjusted = apply_national_preserving_regional_shape(
                target,
                weights,
                gain=gain,
                prediction_column=ACTIVE_PREDICTION_COLUMN,
            )
            adjusted["v24_region_weight_source"] = weight_source
            adjusted_by_gain[gain][election_id] = adjusted
            metric_rows.append(
                {
                    "gain": gain,
                    "election_id": election_id,
                    "region_weight_source": weight_source,
                    "regional_weighted_mae_pp": _region_weighted_mae(
                        adjusted, "v24_regional_shape_pred"
                    ),
                    "national_mae_pp": _national_mae(adjusted, "v24_regional_shape_pred"),
                    "winner_hit": _winner_hit(adjusted, "v24_regional_shape_pred"),
                    "forecast_national_total_max_drift": float(
                        adjusted["v24_national_total_max_drift"].max()
                    ),
                }
            )
        prior = target

    metrics = pd.DataFrame(metric_rows)
    selected_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    for order_index, election_id in enumerate(ORDER):
        history = ORDER[:order_index]
        if not history:
            selected_gain = 0.0
            selection_loss = np.nan
            reason = "no_prior_scored_election"
        else:
            candidates = metrics.loc[
                metrics["election_id"].isin(history)
                & metrics["gain"].isin(NESTED_GAIN_GRID)
            ].groupby("gain")["regional_weighted_mae_pp"].mean()
            selected_gain = float(candidates.sort_values(kind="stable").index[0])
            selection_loss = float(candidates.loc[selected_gain])
            reason = "minimum_mean_prior_election_regional_mae"
        selected = adjusted_by_gain[selected_gain][election_id].copy()
        selected["v24_nested_selected_gain"] = selected_gain
        selected_frames.append(selected)
        selection_rows.append(
            {
                "election_id": election_id,
                "selected_gain": selected_gain,
                "prior_elections": len(history),
                "prior_selection_loss_pp": selection_loss,
                "selection_reason": reason,
            }
        )
    selected = pd.concat(selected_frames, ignore_index=True)

    interval_rows: list[dict[str, object]] = []
    national_interval_rows: list[dict[str, object]] = []
    for target_index, election_id in enumerate(ORDER):
        if target_index == 0:
            continue
        train = selected.loc[selected["election_id"].isin(ORDER[:target_index])].copy()
        target = selected.loc[selected["election_id"].eq(election_id)].copy().reset_index(drop=True)
        weight_regions, region_weight_draws, weight_components = draw_region_weight_uncertainty(
            train,
            target,
            n_sim=n_sim,
            seed=seed + target_index * 1000 + 777,
        )
        weight_region_lookup = {region: index for index, region in enumerate(weight_regions)}
        evaluation_weight = target.groupby("region_id")["contest_votes"].transform("first").to_numpy(
            float, copy=True
        )
        evaluation_weight /= evaluation_weight.sum()
        for structure_index, (structure, specification) in enumerate(
            INTERVAL_STRUCTURES.items()
        ):
            distribution, common_multiplier, regional_multiplier, local_multiplier = specification
            for scale_index, scale in enumerate(SCALE_GRID):
                draw_seed = seed + target_index * 1000 + structure_index * 100 + scale_index
                if structure == "empirical_full_hierarchy":
                    draw_seed = seed + target_index * 1000 + 500
                draws, components = hierarchical_residual_draws(
                    train,
                    target,
                    n_sim=n_sim,
                    seed=draw_seed,
                    residual_scale=scale,
                    common_multiplier=common_multiplier,
                    regional_multiplier=regional_multiplier,
                    local_multiplier=local_multiplier,
                    distribution=distribution,
                )
                record: dict[str, object] = {
                    "election_id": election_id,
                    "interval_structure": structure,
                    "distribution": distribution,
                    "residual_scale": scale,
                    "common_multiplier": common_multiplier,
                    "regional_multiplier": regional_multiplier,
                    "local_multiplier": local_multiplier,
                    "training_elections": components.training_elections,
                    "training_rows": components.training_rows,
                    "common_sigma_log_share": components.common_sigma,
                    "regional_sigma_log_share": components.regional_sigma,
                    "local_sigma_log_share": components.local_sigma,
                    "region_weight_log_sigma": weight_components.log_weight_sigma,
                    "region_weight_training_transitions": weight_components.training_transitions,
                }
                actual = target["actual"].to_numpy(float)
                realized_region_volume = (
                    target[["region_id", "contest_votes"]]
                    .drop_duplicates("region_id")
                    .set_index("region_id")["contest_votes"]
                    .astype(float)
                )
                realized_region_weight = realized_region_volume / realized_region_volume.sum()
                for level, low_q, high_q in (
                    (90, 5.0, 95.0),
                    (95, 2.5, 97.5),
                    (99, 0.5, 99.5),
                ):
                    low = np.percentile(draws, low_q, axis=0)
                    high = np.percentile(draws, high_q, axis=0)
                    covered = (low <= actual) & (actual <= high)
                    widths = (high - low) * 100.0
                    record[f"coverage_{level}"] = float(np.sum(covered * evaluation_weight))
                    record[f"mean_width_{level}_pp"] = float(
                        np.sum(widths * evaluation_weight)
                    )
                    national_covered: list[bool] = []
                    national_widths: list[float] = []
                    structural_covered: list[bool] = []
                    structural_widths: list[float] = []
                    for slot, indices in target.groupby("slot", sort=False).indices.items():
                        idx = np.fromiter(indices, dtype=int)
                        group = target.iloc[idx]
                        forecast_weights = group["v24_forecast_region_weight"].to_numpy(
                            float, copy=True
                        )
                        forecast_weights /= forecast_weights.sum()
                        national_draws = draws[:, idx] @ forecast_weights
                        weight_indices = np.array(
                            [weight_region_lookup[str(region)] for region in group["region_id"]],
                            dtype=int,
                        )
                        structural_national_draws = np.sum(
                            draws[:, idx] * region_weight_draws[:, weight_indices],
                            axis=1,
                        )
                        national_low = float(np.percentile(national_draws, low_q))
                        national_high = float(np.percentile(national_draws, high_q))
                        structural_low = float(
                            np.percentile(structural_national_draws, low_q)
                        )
                        structural_high = float(
                            np.percentile(structural_national_draws, high_q)
                        )
                        national_point = float(
                            np.dot(
                                group["v24_regional_shape_pred"].to_numpy(float),
                                forecast_weights,
                            )
                        )
                        official_weights = group["region_id"].map(realized_region_weight).to_numpy(
                            float
                        )
                        national_actual = float(
                            np.dot(group["actual"].to_numpy(float), official_weights)
                        )
                        covered_national = national_low <= national_actual <= national_high
                        width_national = (national_high - national_low) * 100.0
                        covered_structural = (
                            structural_low <= national_actual <= structural_high
                        )
                        width_structural = (structural_high - structural_low) * 100.0
                        national_covered.append(covered_national)
                        national_widths.append(width_national)
                        structural_covered.append(covered_structural)
                        structural_widths.append(width_structural)
                        national_interval_rows.append(
                            {
                                "election_id": election_id,
                                "slot": slot,
                                "interval_structure": structure,
                                "distribution": distribution,
                                "residual_scale": scale,
                                "nominal_level": level,
                                "forecast_weighted_point": national_point,
                                "official_actual": national_actual,
                                "lower": national_low,
                                "upper": national_high,
                                "covered": covered_national,
                                "width_pp": width_national,
                                "structural_lower": structural_low,
                                "structural_upper": structural_high,
                                "structural_covered": covered_structural,
                                "structural_width_pp": width_structural,
                                "region_weight_log_sigma": weight_components.log_weight_sigma,
                                "region_weight_training_transitions": weight_components.training_transitions,
                                "forecast_region_weight_source": group[
                                    "v24_region_weight_source"
                                ].iloc[0],
                            }
                        )
                    record[f"national_coverage_{level}"] = float(np.mean(national_covered))
                    record[f"national_mean_width_{level}_pp"] = float(
                        np.mean(national_widths)
                    )
                    record[f"national_structural_coverage_{level}"] = float(
                        np.mean(structural_covered)
                    )
                    record[f"national_structural_mean_width_{level}_pp"] = float(
                        np.mean(structural_widths)
                    )
                interval_rows.append(record)

    intervals = pd.DataFrame(interval_rows)
    selected_election_metrics = pd.DataFrame(
        [
            {
                "election_id": election_id,
                "selected_gain": float(group["v24_nested_selected_gain"].iloc[0]),
                "baseline_regional_mae_pp": _region_weighted_mae(
                    group, ACTIVE_PREDICTION_COLUMN
                ),
                "v24_regional_mae_pp": _region_weighted_mae(group, "v24_regional_shape_pred"),
                "baseline_national_mae_pp": _national_mae(
                    group, ACTIVE_PREDICTION_COLUMN
                ),
                "v24_national_mae_pp": _national_mae(group, "v24_regional_shape_pred"),
                "winner_hit": _winner_hit(group, "v24_regional_shape_pred"),
                "forecast_national_total_max_drift": float(
                    group["v24_national_total_max_drift"].max()
                ),
            }
            for election_id, group in selected.groupby("election_id", sort=False)
        ]
    )

    scale_summary = intervals.groupby(
        ["interval_structure", "residual_scale"], as_index=False
    ).agg(
        elections=("election_id", "nunique"),
        coverage_90=("coverage_90", "mean"),
        mean_width_90_pp=("mean_width_90_pp", "mean"),
        coverage_95=("coverage_95", "mean"),
        mean_width_95_pp=("mean_width_95_pp", "mean"),
        coverage_99=("coverage_99", "mean"),
        mean_width_99_pp=("mean_width_99_pp", "mean"),
        national_coverage_90=("national_coverage_90", "mean"),
        national_mean_width_90_pp=("national_mean_width_90_pp", "mean"),
        national_coverage_95=("national_coverage_95", "mean"),
        national_mean_width_95_pp=("national_mean_width_95_pp", "mean"),
        national_coverage_99=("national_coverage_99", "mean"),
        national_mean_width_99_pp=("national_mean_width_99_pp", "mean"),
        national_structural_coverage_90=("national_structural_coverage_90", "mean"),
        national_structural_mean_width_90_pp=(
            "national_structural_mean_width_90_pp",
            "mean",
        ),
        national_structural_coverage_95=("national_structural_coverage_95", "mean"),
        national_structural_mean_width_95_pp=(
            "national_structural_mean_width_95_pp",
            "mean",
        ),
        national_structural_coverage_99=("national_structural_coverage_99", "mean"),
        national_structural_mean_width_99_pp=(
            "national_structural_mean_width_99_pp",
            "mean",
        ),
    )
    frontier_rows: list[dict[str, object]] = []
    for scope, coverage_prefix, width_prefix in (
        ("regional_row", "coverage", "mean_width"),
        ("national_candidate", "national_coverage", "national_mean_width"),
        (
            "national_candidate_with_region_weight_uncertainty",
            "national_structural_coverage",
            "national_structural_mean_width",
        ),
    ):
        for level, nominal in ((90, 0.90), (95, 0.95), (99, 0.99)):
            coverage_column = f"{coverage_prefix}_{level}"
            width_column = f"{width_prefix}_{level}_pp"
            eligible = scale_summary.loc[scale_summary[coverage_column].ge(nominal)].copy()
            if eligible.empty:
                continue
            best = eligible.sort_values(width_column, kind="stable").iloc[0]
            frontier_rows.append(
                {
                    "interval_scope": scope,
                    "nominal_level": level,
                    "interval_structure": best["interval_structure"],
                    "residual_scale": best["residual_scale"],
                    "observed_weighted_coverage": best[coverage_column],
                    "mean_width_pp": best[width_column],
                    "selection_warning": "development outcome aware; not a nested deployment selection",
                }
            )
    frontier = pd.DataFrame(frontier_rows)

    _atomic_csv(metrics, OUTPUT_DIR / "fixed_gain_ablation.csv")
    _atomic_csv(pd.DataFrame(selection_rows), OUTPUT_DIR / "nested_gain_selection.csv")
    _atomic_csv(selected, OUTPUT_DIR / "nested_predictions.csv")
    _atomic_csv(selected_election_metrics, OUTPUT_DIR / "nested_performance_by_election.csv")
    national_interval_detail = pd.DataFrame(national_interval_rows)
    coherent_national = national_interval_detail.loc[
        national_interval_detail["interval_structure"].eq("empirical_full_hierarchy")
        & national_interval_detail["residual_scale"].eq(COHERENT_NATIONAL_SCALE)
    ].copy()
    coherent_national["interval_scope"] = (
        "national_candidate_share_with_prior_region_weight_uncertainty"
    )
    coherent_by_election = (
        coherent_national.groupby(["nominal_level", "election_id"], as_index=False)
        .agg(
            candidates=("covered", "size"),
            fixed_weight_coverage=("covered", "mean"),
            fixed_weight_mean_width_pp=("width_pp", "mean"),
            structural_coverage=("structural_covered", "mean"),
            structural_mean_width_pp=("structural_width_pp", "mean"),
        )
    )
    coherent_summary = (
        coherent_by_election.groupby("nominal_level", as_index=False)
        .agg(
            elections=("election_id", "nunique"),
            candidates=("candidates", "sum"),
            fixed_weight_coverage=("fixed_weight_coverage", "mean"),
            fixed_weight_mean_width_pp=("fixed_weight_mean_width_pp", "mean"),
            structural_coverage=("structural_coverage", "mean"),
            structural_mean_width_pp=("structural_mean_width_pp", "mean"),
        )
    )
    _atomic_csv(intervals, OUTPUT_DIR / "hierarchical_interval_by_election.csv")
    _atomic_csv(
        national_interval_detail,
        OUTPUT_DIR / "hierarchical_national_interval_rows.csv",
    )
    _atomic_csv(
        coherent_national,
        OUTPUT_DIR / "coherent_national_candidate_intervals.csv",
    )
    _atomic_csv(
        coherent_summary,
        OUTPUT_DIR / "coherent_national_candidate_interval_summary.csv",
    )
    _atomic_csv(
        coherent_by_election,
        OUTPUT_DIR / "coherent_national_candidate_interval_by_election.csv",
    )
    _atomic_csv(scale_summary, OUTPUT_DIR / "hierarchical_interval_scale_summary.csv")
    _atomic_csv(frontier, OUTPUT_DIR / "hierarchical_interval_frontier.csv")

    summary: dict[str, object] = {
        "schema": "v24_regional_shape_interval_experiment_v1",
        "status": "shadow_experiment_not_promoted",
        "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": input_hash,
        "active_prediction_column": ACTIVE_PREDICTION_COLUMN,
        "frozen_v23_modified": False,
        "post_2022_outcomes_used": False,
        "regional_shape": {
            "selection": "strictly_prior_election_regional_mae",
            "nested_gain_grid": NESTED_GAIN_GRID,
            "wide_ablation_gain_grid": GAIN_GRID,
            "gain_cap_rationale": "maximum direct log-share tilt about 0.04 before national-total raking",
            "target_actual_region_weights_used_in_adjustment": False,
            "first_fold_weight_fallback": "equal_region",
            "forecast_national_totals_preserved": True,
            "baseline_regional_macro_mae_pp": float(
                selected_election_metrics["baseline_regional_mae_pp"].mean()
            ),
            "v24_regional_macro_mae_pp": float(
                selected_election_metrics["v24_regional_mae_pp"].mean()
            ),
            "baseline_national_macro_mae_pp": float(
                selected_election_metrics["baseline_national_mae_pp"].mean()
            ),
            "v24_national_macro_mae_pp": float(
                selected_election_metrics["v24_national_mae_pp"].mean()
            ),
            "winner_accuracy": float(selected_election_metrics["winner_hit"].mean()),
            "maximum_forecast_total_drift": float(
                selected_election_metrics["forecast_national_total_max_drift"].max()
            ),
            "promotion_recommendation": "keep_shadow",
            "promotion_reason": "macro gain is only about 0.024pp and two of five elections regress",
        },
        "intervals": {
            "first_scored_fold_excluded": "no strictly prior scored residuals",
            "target_outcome_used_for_interval_construction": False,
            "coefficient_uncertainty_included": False,
            "residual_hierarchy": ["candidate_common", "regional", "local"],
            "structures_evaluated": list(INTERVAL_STRUCTURES),
            "scales_evaluated": SCALE_GRID,
            "frontier_selection_is_development_outcome_aware": True,
            "regional_row_promotion_recommendation": "reject_for_production",
            "regional_row_promotion_reason": "nominal row coverage requires wide intervals",
            "national_candidate_shadow": {
                "structure": "empirical_full_hierarchy",
                "residual_scale": COHERENT_NATIONAL_SCALE,
                "region_weight_uncertainty": True,
                "selection_is_development_outcome_aware": True,
                "metrics": coherent_summary.to_dict(orient="records"),
                "promotion_recommendation": "keep_shadow",
                "promotion_reason": "national intervals are coherent and materially narrower, but only four evaluable elections are available",
            },
            "explicit_coefficient_covariance_draws_included": False,
            "coefficient_uncertainty_note": "strict out-of-sample residuals implicitly contain historical estimation error but no separate covariance draw is added",
        },
    }
    _atomic_json(summary, OUTPUT_DIR / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2402)
    args = parser.parse_args()
    print(json.dumps(run(args.n_sim, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
