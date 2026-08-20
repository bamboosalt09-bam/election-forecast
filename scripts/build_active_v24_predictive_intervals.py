"""Build chronological predictive intervals for the active V24 point model.

The interval model is downstream of the fixed V24 point predictions.  For a
target election it uses only residuals and regional vote-volume transitions
from strictly earlier presidential elections.  Target outcomes are read only
after the interval bounds have been fixed, and only to report historical
coverage.  No post-2022 outcome is read.

These are predictive intervals for national candidate vote share, not
confidence intervals for fitted Ridge coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine.v24_calibration import (  # noqa: E402
    draw_region_weight_uncertainty,
    hierarchical_residual_draws,
    prior_region_weights,
)


INPUT = ROOT / "outputs" / "active_presidential_nested_v24" / "nested_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "active_presidential_nested_v24"
ORDER = ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"]
PREDICTION_COLUMN = "layer_pred"
DEFAULT_LEVELS = (0.50, 0.80, 0.90, 0.95)
DEFAULT_RESIDUAL_SCALE = 1.0
DEFAULT_DISTRIBUTION = "empirical"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _candidate_name_column(frame: pd.DataFrame) -> str:
    for column in ("candidate_name", "candidate_name_x", "candidate_name_y"):
        if column in frame.columns:
            return column
    raise ValueError("V24 prediction panel has no candidate-name column")


def _manifest_path(path: Path) -> str:
    try:
        rendered = path.relative_to(ROOT)
    except ValueError:
        rendered = path.resolve()
    return str(rendered).replace("\\", "/")


def _attach_forecast_region_weights(
    target: pd.DataFrame,
    prior: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    weights, source = prior_region_weights(target, prior)
    out = target.copy().reset_index(drop=True)
    out["v24_forecast_region_weight"] = out["region_id"].astype(str).map(weights)
    if out["v24_forecast_region_weight"].isna().any():
        raise RuntimeError("forecast-time regional weights did not cover the target panel")
    return out, source


def build(
    *,
    n_sim: int = 50_000,
    seed: int = 24_820,
    residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    distribution: str = DEFAULT_DISTRIBUTION,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    if n_sim < 1_000:
        raise ValueError("n_sim must be at least 1,000")
    if residual_scale <= 0.0:
        raise ValueError("residual_scale must be positive")
    if not levels or any(level <= 0.0 or level >= 1.0 for level in levels):
        raise ValueError("interval levels must lie strictly between zero and one")

    panel = pd.read_csv(INPUT, encoding="utf-8-sig")
    panel = panel.loc[panel["election_id"].isin(ORDER)].copy()
    required = {
        "election_id",
        "region_id",
        "slot",
        "actual",
        "contest_votes",
        PREDICTION_COLUMN,
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"V24 prediction panel missing columns: {sorted(missing)}")
    if set(panel["election_id"].astype(str)) != set(ORDER):
        raise RuntimeError("V24 prediction panel does not contain the full 2002-2022 lineage")
    if panel["election_id"].astype(str).str.contains("2025").any():
        raise RuntimeError("post-2022 row found in V24 historical interval input")

    name_column = _candidate_name_column(panel)
    interval_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []

    for target_index, election_id in enumerate(ORDER[1:], start=1):
        prior_ids = ORDER[:target_index]
        train = panel.loc[panel["election_id"].isin(prior_ids)].copy()
        prior = panel.loc[panel["election_id"].eq(prior_ids[-1])].copy()
        target = panel.loc[panel["election_id"].eq(election_id)].copy()
        target, weight_source = _attach_forecast_region_weights(target, prior)

        target_ids = set(target["election_id"].astype(str))
        training_ids = set(train["election_id"].astype(str))
        if target_ids & training_ids:
            raise RuntimeError(f"target leakage in interval fold {election_id}")

        weight_regions, weight_draws, weight_components = draw_region_weight_uncertainty(
            train,
            target,
            n_sim=n_sim,
            seed=seed + target_index * 10_000 + 17,
        )
        draws, residual_components = hierarchical_residual_draws(
            train,
            target,
            n_sim=n_sim,
            seed=seed + target_index * 10_000 + 29,
            prediction_column=PREDICTION_COLUMN,
            residual_scale=residual_scale,
            distribution=distribution,
        )
        weight_lookup = {region: index for index, region in enumerate(weight_regions)}
        realized_volume = (
            target[["region_id", "contest_votes"]]
            .drop_duplicates("region_id")
            .set_index("region_id")["contest_votes"]
            .astype(float)
        )
        realized_weight = realized_volume / realized_volume.sum()

        component_rows.append(
            {
                "election_id": election_id,
                "training_elections": "|".join(prior_ids),
                "training_election_count": residual_components.training_elections,
                "training_rows": residual_components.training_rows,
                "common_sigma": residual_components.common_sigma,
                "regional_sigma": residual_components.regional_sigma,
                "local_sigma": residual_components.local_sigma,
                "region_weight_log_sigma": weight_components.log_weight_sigma,
                "region_weight_training_transitions": weight_components.training_transitions,
                "forecast_region_weight_source": weight_source,
                "target_outcome_used_to_construct_bounds": False,
            }
        )

        for slot, indices in target.groupby("slot", sort=False).indices.items():
            idx = np.fromiter(indices, dtype=int)
            group = target.iloc[idx]
            weight_idx = np.array(
                [weight_lookup[str(region)] for region in group["region_id"]],
                dtype=int,
            )
            national_draws = np.sum(draws[:, idx] * weight_draws[:, weight_idx], axis=1)
            forecast_weights = group["v24_forecast_region_weight"].to_numpy(
                float, copy=True
            )
            forecast_weights /= forecast_weights.sum()
            point = float(np.dot(group[PREDICTION_COLUMN].to_numpy(float), forecast_weights))
            official_weights = group["region_id"].map(realized_weight).to_numpy(float)
            actual = float(np.dot(group["actual"].to_numpy(float), official_weights))

            for level in sorted(set(levels)):
                tail = (1.0 - level) / 2.0
                lower = float(np.quantile(national_draws, tail))
                upper = float(np.quantile(national_draws, 1.0 - tail))
                interval_rows.append(
                    {
                        "election_id": election_id,
                        "slot": str(slot),
                        "candidate_name": str(group[name_column].iloc[0]),
                        "nominal_level": level,
                        "point_share": point,
                        "lower_share": lower,
                        "upper_share": upper,
                        "width_pp": (upper - lower) * 100.0,
                        "official_actual_share": actual,
                        "covered": lower <= actual <= upper,
                        "training_elections": "|".join(prior_ids),
                        "forecast_region_weight_source": weight_source,
                        "target_outcome_used_to_construct_bounds": False,
                    }
                )

    intervals = pd.DataFrame(interval_rows)
    components = pd.DataFrame(component_rows)
    by_election = intervals.groupby(
        ["nominal_level", "election_id"], as_index=False
    ).agg(
        candidates=("covered", "size"),
        coverage=("covered", "mean"),
        mean_width_pp=("width_pp", "mean"),
    )
    summary = by_election.groupby("nominal_level", as_index=False).agg(
        elections=("election_id", "nunique"),
        candidates=("candidates", "sum"),
        equal_election_coverage=("coverage", "mean"),
        candidate_weighted_coverage=(
            "coverage",
            lambda values: float(
                np.average(
                    values,
                    weights=by_election.loc[values.index, "candidates"],
                )
            ),
        ),
        equal_election_mean_width_pp=("mean_width_pp", "mean"),
    )

    interval_path = output_dir / "national_predictive_intervals.csv"
    summary_path = output_dir / "predictive_interval_summary.csv"
    components_path = output_dir / "predictive_interval_components.csv"
    manifest_path = output_dir / "predictive_interval_manifest.json"
    _atomic_csv(intervals, interval_path)
    _atomic_csv(summary, summary_path)
    _atomic_csv(components, components_path)

    payload: dict[str, object] = {
        "schema": "active_v24_national_predictive_intervals_v1",
        "status": "historical_chronological_calibration",
        "model_version": "v24",
        "interval_scope": "national_candidate_vote_share",
        "interval_type": "predictive_interval_not_coefficient_confidence_interval",
        "point_prediction_column": PREDICTION_COLUMN,
        "residual_structure": "empirical_full_hierarchy_with_region_weight_uncertainty",
        "residual_scale": residual_scale,
        "residual_scale_policy": "fixed_unscaled_not_selected_on_coverage",
        "distribution": distribution,
        "levels": list(sorted(set(levels))),
        "n_sim": n_sim,
        "seed": seed,
        "warmup_election": ORDER[0],
        "evaluated_elections": ORDER[1:],
        "candidate_outcomes": int(intervals.loc[intervals["nominal_level"].eq(levels[0])].shape[0]),
        "target_outcomes_used_to_construct_bounds": False,
        "target_outcomes_used_for_historical_coverage_only": True,
        "post_2022_outcomes_used": False,
        "development_outcome_warning": (
            "V24 point-model rules were developed on the through-2022 sample; "
            "coverage is historical calibration, not an untouched holdout guarantee."
        ),
        "input": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": _sha256(INPUT),
        "outputs": {
            "intervals": _manifest_path(interval_path),
            "summary": _manifest_path(summary_path),
            "components": _manifest_path(components_path),
        },
        "summary": summary.to_dict(orient="records"),
    }
    _atomic_json(payload, manifest_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=24_820)
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument(
        "--levels",
        default=",".join(str(level) for level in DEFAULT_LEVELS),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    levels = tuple(float(value) for value in args.levels.split(",") if value.strip())
    payload = build(
        n_sim=args.n_sim,
        seed=args.seed,
        residual_scale=args.residual_scale,
        levels=levels,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
