"""Refine the coherent V24 national predictive-interval scale near its boundary."""

from __future__ import annotations

import argparse
import json
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
)


INPUT = ROOT / "outputs" / "experiments" / "v24_regional_shape_intervals" / "nested_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "experiments" / "v24_national_interval_scale_refinement"
ORDER = ["pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"]
DEFAULT_SCALES = [0.55, 0.60, 0.65, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.80, 0.85]


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


def run(scales: list[float], n_sim: int, seeds: list[int]) -> dict[str, object]:
    panel = pd.read_csv(INPUT, encoding="utf-8-sig")
    details: list[dict[str, object]] = []
    for seed in seeds:
        for target_index, election_id in enumerate(ORDER[1:], start=1):
            train = panel.loc[panel["election_id"].isin(ORDER[:target_index])].copy()
            target = panel.loc[panel["election_id"].eq(election_id)].copy().reset_index(drop=True)
            weight_regions, weight_draws, weight_components = draw_region_weight_uncertainty(
                train,
                target,
                n_sim=n_sim,
                seed=seed + target_index * 1000 + 777,
            )
            weight_lookup = {region: index for index, region in enumerate(weight_regions)}
            realized_volume = (
                target[["region_id", "contest_votes"]]
                .drop_duplicates("region_id")
                .set_index("region_id")["contest_votes"]
                .astype(float)
            )
            realized_weight = realized_volume / realized_volume.sum()
            for scale in scales:
                draws, _ = hierarchical_residual_draws(
                    train,
                    target,
                    n_sim=n_sim,
                    seed=seed + target_index * 1000 + 500,
                    residual_scale=scale,
                    distribution="empirical",
                )
                candidate_rows: list[dict[str, object]] = []
                for slot, indices in target.groupby("slot", sort=False).indices.items():
                    idx = np.fromiter(indices, dtype=int)
                    group = target.iloc[idx]
                    weight_indices = np.array(
                        [weight_lookup[str(region)] for region in group["region_id"]],
                        dtype=int,
                    )
                    national_draws = np.sum(
                        draws[:, idx] * weight_draws[:, weight_indices], axis=1
                    )
                    official_weights = group["region_id"].map(realized_weight).to_numpy(float)
                    actual = float(np.dot(group["actual"].to_numpy(float), official_weights))
                    for level, lower_q, upper_q in (
                        (90, 5.0, 95.0),
                        (95, 2.5, 97.5),
                        (99, 0.5, 99.5),
                    ):
                        lower = float(np.percentile(national_draws, lower_q))
                        upper = float(np.percentile(national_draws, upper_q))
                        candidate_rows.append(
                            {
                                "seed": seed,
                                "election_id": election_id,
                                "slot": slot,
                                "scale": scale,
                                "nominal_level": level,
                                "actual": actual,
                                "lower": lower,
                                "upper": upper,
                                "covered": lower <= actual <= upper,
                                "width_pp": (upper - lower) * 100.0,
                                "region_weight_log_sigma": weight_components.log_weight_sigma,
                            }
                        )
                details.extend(candidate_rows)

    detail = pd.DataFrame(details)
    by_election = detail.groupby(
        ["seed", "scale", "nominal_level", "election_id"], as_index=False
    ).agg(
        candidates=("covered", "size"),
        coverage=("covered", "mean"),
        mean_width_pp=("width_pp", "mean"),
    )
    by_seed = by_election.groupby(
        ["seed", "scale", "nominal_level"], as_index=False
    ).agg(
        elections=("election_id", "nunique"),
        candidates=("candidates", "sum"),
        equal_election_coverage=("coverage", "mean"),
        equal_election_mean_width_pp=("mean_width_pp", "mean"),
    )
    summary = by_seed.groupby(["nominal_level", "scale"], as_index=False).agg(
        seeds=("seed", "nunique"),
        minimum_seed_coverage=("equal_election_coverage", "min"),
        mean_seed_coverage=("equal_election_coverage", "mean"),
        mean_width_pp=("equal_election_mean_width_pp", "mean"),
        maximum_seed_mean_width_pp=("equal_election_mean_width_pp", "max"),
    )
    selected_rows: list[dict[str, object]] = []
    for level, target_coverage in ((90, 0.90), (95, 0.95), (99, 0.99)):
        eligible = summary.loc[
            summary["nominal_level"].eq(level)
            & summary["minimum_seed_coverage"].ge(target_coverage)
        ].sort_values("scale")
        if not eligible.empty:
            selected_rows.append(eligible.iloc[0].to_dict())
    selected_scales = {
        int(row["nominal_level"]): float(row["scale"]) for row in selected_rows
    }
    selected_detail = pd.concat(
        [
            detail.loc[
                detail["nominal_level"].eq(level)
                & detail["scale"].eq(scale)
            ]
            for level, scale in selected_scales.items()
        ],
        ignore_index=True,
    )
    nested_rows: list[dict[str, object]] = []
    if set(selected_scales) == {90, 95, 99}:
        for keys, group in selected_detail.groupby(
            ["seed", "election_id", "slot"], sort=False
        ):
            by_level = group.set_index("nominal_level")
            lower_90 = float(by_level.loc[90, "lower"])
            upper_90 = float(by_level.loc[90, "upper"])
            lower_95 = min(float(by_level.loc[95, "lower"]), lower_90)
            upper_95 = max(float(by_level.loc[95, "upper"]), upper_90)
            lower_99 = min(float(by_level.loc[99, "lower"]), lower_95)
            upper_99 = max(float(by_level.loc[99, "upper"]), upper_95)
            actual = float(by_level.loc[90, "actual"])
            for level, lower, upper in (
                (90, lower_90, upper_90),
                (95, lower_95, upper_95),
                (99, lower_99, upper_99),
            ):
                nested_rows.append(
                    {
                        "seed": keys[0],
                        "election_id": keys[1],
                        "slot": keys[2],
                        "nominal_level": level,
                        "scale": selected_scales[level],
                        "actual": actual,
                        "lower": lower,
                        "upper": upper,
                        "covered": lower <= actual <= upper,
                        "width_pp": (upper - lower) * 100.0,
                        "nesting_enforced": True,
                    }
                )
    nested_detail = pd.DataFrame(nested_rows)
    if nested_detail.empty:
        nested_summary = pd.DataFrame()
    else:
        nested_by_election = nested_detail.groupby(
            ["seed", "nominal_level", "election_id"], as_index=False
        ).agg(
            coverage=("covered", "mean"),
            mean_width_pp=("width_pp", "mean"),
        )
        nested_by_seed = nested_by_election.groupby(
            ["seed", "nominal_level"], as_index=False
        ).agg(
            equal_election_coverage=("coverage", "mean"),
            equal_election_mean_width_pp=("mean_width_pp", "mean"),
        )
        nested_summary = nested_by_seed.groupby("nominal_level", as_index=False).agg(
            seeds=("seed", "nunique"),
            minimum_seed_coverage=("equal_election_coverage", "min"),
            mean_seed_coverage=("equal_election_coverage", "mean"),
            mean_width_pp=("equal_election_mean_width_pp", "mean"),
            maximum_seed_mean_width_pp=("equal_election_mean_width_pp", "max"),
        )
    payload: dict[str, object] = {
        "schema": "v24_national_interval_scale_refinement_v1",
        "status": "shadow_development_selection",
        "interval_scope": "national_candidate_predictive_interval",
        "structure": "empirical_full_hierarchy_with_region_weight_uncertainty",
        "n_sim": n_sim,
        "seeds": seeds,
        "scales": scales,
        "selected_minimum_scale_by_nominal_level": selected_rows,
        "nested_level_calibrated_metrics": nested_summary.to_dict(orient="records"),
        "nested_level_calibration_policy": "95 contains 90; 99 contains 95",
        "selection_is_development_outcome_aware": True,
        "post_2022_outcomes_used": False,
    }
    _atomic_csv(detail, OUTPUT_DIR / "candidate_intervals.csv")
    _atomic_csv(by_election, OUTPUT_DIR / "by_election.csv")
    _atomic_csv(by_seed, OUTPUT_DIR / "by_seed.csv")
    _atomic_csv(summary, OUTPUT_DIR / "scale_summary.csv")
    _atomic_csv(nested_detail, OUTPUT_DIR / "selected_nested_intervals_all_seeds.csv")
    _atomic_csv(
        nested_detail.loc[nested_detail["seed"].eq(seeds[0])],
        OUTPUT_DIR / "selected_nested_intervals_canonical_seed.csv",
    )
    _atomic_csv(nested_summary, OUTPUT_DIR / "selected_nested_interval_summary.csv")
    _atomic_json(payload, OUTPUT_DIR / "summary.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=20_000)
    parser.add_argument("--seeds", default="2402,3402,4402,5402,6402")
    parser.add_argument(
        "--scales",
        default=",".join(str(value) for value in DEFAULT_SCALES),
    )
    args = parser.parse_args()
    scales = [float(value) for value in args.scales.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    print(json.dumps(run(scales, args.n_sim, seeds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
