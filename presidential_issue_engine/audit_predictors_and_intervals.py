"""Audit predictor selection provenance and rolling interval calibration.

The production Monte Carlo table is fit on the complete scored panel. This
script instead estimates every interval from strictly earlier elections and
therefore measures historical rolling coverage without target-outcome access.
It is an experiment and does not change production defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from presidential_issue_engine import issue_vote_engine as engine
except ImportError:  # pragma: no cover - supports direct script execution
    import issue_vote_engine as engine  # type: ignore


OUTPUT_DIR = Path("outputs/predictor_interval_audit")
RESULTS_OUTPUT = OUTPUT_DIR / "rolling_interval_experiment.csv"
DETAIL_OUTPUT = OUTPUT_DIR / "rolling_interval_rows.csv"
PREDICTOR_OUTPUT = OUTPUT_DIR / "predictor_inventory.csv"


def _postprocess_after_candidate(
    frame: pd.DataFrame,
    pred: np.ndarray,
    *,
    electorate_layer: bool = True,
) -> np.ndarray:
    """Apply the rolling post-model sequence after candidate adjustments."""

    return engine.apply_prediction_postprocess(
        frame,
        engine.normalize_vote_share_predictions(frame, pred),
        electorate_layer=electorate_layer,
    )


def _common_residual_draws(
    rng: np.random.Generator,
    train: pd.DataFrame,
    residuals: np.ndarray,
    test: pd.DataFrame,
    n_sim: int,
) -> tuple[np.ndarray, float]:
    """Bootstrap election-slot residuals from training into a target contest."""

    residual_frame = train[["election_id", "region_id", "slot"]].copy()
    residual_frame["residual"] = np.asarray(residuals, dtype=float)
    if "votes" in train.columns:
        residual_frame["weight"] = (
            train.groupby(["election_id", "region_id"])["votes"]
            .transform("sum")
            .to_numpy(float)
        )
    else:
        residual_frame["weight"] = 1.0

    common = (
        residual_frame.groupby(["election_id", "slot"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "common_residual": engine._weighted_group_average(
                        group["residual"],
                        group["weight"],
                    )
                }
            ),
            include_groups=False,
        )
    )
    values = common["common_residual"].to_numpy(float)
    centered_values = values - float(values.mean()) if len(values) else values
    sigma = float(np.sqrt(np.mean(centered_values**2))) if len(centered_values) else 0.0

    target_keys = test[["election_id", "slot"]].drop_duplicates().reset_index(drop=True)
    if len(centered_values):
        key_draws = rng.choice(
            centered_values,
            size=(n_sim, len(target_keys)),
            replace=True,
        )
    else:
        key_draws = np.zeros((n_sim, len(target_keys)), dtype=float)
    for indices in target_keys.groupby("election_id").indices.values():
        index_array = np.fromiter(indices, dtype=int)
        if len(index_array) > 1:
            key_draws[:, index_array] -= key_draws[:, index_array].mean(
                axis=1,
                keepdims=True,
            )
    key_lookup = target_keys.copy()
    key_lookup["draw_index"] = np.arange(len(key_lookup))
    row_indices = (
        test[["election_id", "slot"]]
        .merge(key_lookup, on=["election_id", "slot"], how="left")["draw_index"]
        .astype(int)
        .to_numpy()
    )
    return key_draws[:, row_indices], sigma


def _rolling_fold_draws(
    context: pd.DataFrame,
    warmup: pd.DataFrame,
    election_id: str,
    alpha: float,
    n_sim: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, np.ndarray]:
    full_order = [*engine.ROLLING_WARMUP_ORDER, *engine.ORDER]
    order_lookup = {name: index for index, name in enumerate(full_order)}
    frame = pd.concat([warmup.copy(), context.copy()], ignore_index=True, sort=False)
    for predictor in engine.PREDICTORS:
        frame[predictor] = pd.to_numeric(frame[predictor], errors="coerce").fillna(0.0)
    frame = frame.copy()
    frame["_order"] = frame["election_id"].map(order_lookup)
    target_order = order_lookup[election_id]
    test = engine.scored_contest_rows(frame.loc[frame["election_id"].eq(election_id)]).copy()
    train = frame.loc[frame["_order"].lt(target_order)].copy()
    train["_rolling_target"] = engine.normalized_vote_share_target(train)
    train, residual_mask = engine.rolling_training_with_slot_backfill(
        train,
        test,
        set(engine.ROLLING_WARMUP_ORDER),
    )

    x_train = train[engine.PREDICTORS].to_numpy(float)
    y_train = train["_rolling_target"].to_numpy(float)
    x_test = test[engine.PREDICTORS].to_numpy(float)
    beta, _, covariance, _, means, scales = engine.ridge_fit(
        x_train,
        y_train,
        alpha=alpha,
        sample_weight=engine.election_epoch_sample_weight(train),
    )
    rng = np.random.default_rng(seed)
    beta_draws = rng.multivariate_normal(beta, covariance, size=n_sim)
    test_design = np.column_stack([np.ones(len(test)), (x_test - means) / scales])
    raw_point = engine.ridge_predict(beta, x_test, means, scales)
    raw_draws = beta_draws @ test_design.T

    train_raw = engine.ridge_predict(beta, x_train, means, scales)
    train_adjusted = engine.apply_withdrawn_candidate_prediction_adjustment(
        train,
        engine.apply_third_candidate_prediction_adjustment(train, train_raw),
    )
    test_adjusted = engine.apply_withdrawn_candidate_prediction_adjustment(
        test,
        engine.apply_third_candidate_prediction_adjustment(test, raw_point),
    )
    residual_train = train.loc[residual_mask].copy()
    calibrated = engine.apply_region_residual_calibration(
        residual_train,
        test,
        train_adjusted[residual_mask],
        test_adjusted,
    )
    calibration_delta = calibrated - test_adjusted

    point = _postprocess_after_candidate(test, calibrated)
    coefficient_draws = np.vstack(
        [
            _postprocess_after_candidate(
                test,
                engine.apply_withdrawn_candidate_prediction_adjustment(
                    test,
                    engine.apply_third_candidate_prediction_adjustment(test, draw),
                )
                + calibration_delta,
                electorate_layer=False,
            )
            for draw in raw_draws
        ]
    )
    if engine.ELECTORATE_LAYER_ENABLED:
        coefficient_draws = engine.apply_electorate_layer_response_draws(
            test,
            coefficient_draws,
            engine.ELECTORATE_LAYER_CONFIG,
        )

    train_final = _postprocess_after_candidate(train, train_adjusted)
    train_final_residual = y_train - train_final
    residual_noise, residual_sigma = _common_residual_draws(
        rng,
        residual_train,
        train_final_residual[residual_mask],
        test,
        n_sim,
    )
    return test, point, coefficient_draws, residual_sigma, residual_noise


def run_experiment(
    n_sim: int,
    seed: int,
    alphas: list[float],
    scales: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = engine.assemble()
    warmup = engine.historical_presidential_warmup_frame()
    warmup = warmup.loc[warmup["election_id"].isin(engine.ROLLING_WARMUP_ORDER)].copy()
    details: list[dict[str, object]] = []
    levels = {90: (5.0, 95.0), 95: (2.5, 97.5), 99: (0.5, 99.5)}

    for alpha_index, alpha in enumerate(alphas):
        for election_index, election_id in enumerate(engine.ORDER):
            test, point, coefficient_draws, residual_sigma, residual_noise = _rolling_fold_draws(
                context,
                warmup,
                election_id,
                alpha,
                n_sim,
                seed + alpha_index * 100 + election_index,
            )
            actual = engine.normalized_vote_share_target(test)
            for residual_scale in scales:
                interval_draws = np.vstack(
                    [
                        engine.normalize_vote_share_predictions(
                            test,
                            draw + residual_scale * noise,
                        )
                        for draw, noise in zip(coefficient_draws, residual_noise)
                    ]
                )
                interval_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for level, (low_q, high_q) in levels.items():
                    low = np.minimum(np.percentile(interval_draws, low_q, axis=0), point)
                    high = np.maximum(np.percentile(interval_draws, high_q, axis=0), point)
                    interval_values[level] = (low, high)
                for row_index, (_, row) in enumerate(test.iterrows()):
                    record: dict[str, object] = {
                        "alpha": alpha,
                        "residual_scale": residual_scale,
                        "election_id": election_id,
                        "region_id": row["region_id"],
                        "slot": row["slot"],
                        "actual": actual[row_index],
                        "point": point[row_index],
                        "abs_err_pp": abs(point[row_index] - actual[row_index]) * 100.0,
                        "training_common_residual_sigma_pp": residual_sigma * 100.0,
                    }
                    for level, (low, high) in interval_values.items():
                        record[f"lo{level}"] = low[row_index]
                        record[f"hi{level}"] = high[row_index]
                        record[f"width{level}_pp"] = (high[row_index] - low[row_index]) * 100.0
                        record[f"covered{level}"] = bool(
                            low[row_index] <= actual[row_index] <= high[row_index]
                        )
                    details.append(record)

    detail = pd.DataFrame(details)
    summary_rows: list[dict[str, object]] = []
    for (alpha, residual_scale), group in detail.groupby(["alpha", "residual_scale"]):
        record = {
            "alpha": alpha,
            "residual_scale": residual_scale,
            "rows": len(group),
            "rolling_point_mae_pp": group["abs_err_pp"].mean(),
            "mean_training_common_residual_sigma_pp": group[
                "training_common_residual_sigma_pp"
            ].mean(),
        }
        for level in levels:
            record[f"mean_width{level}_pp"] = group[f"width{level}_pp"].mean()
            record[f"coverage{level}"] = group[f"covered{level}"].mean()
        summary_rows.append(record)
    return pd.DataFrame(summary_rows), detail


def predictor_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    roles = {
        "slot_A": "structural slot indicator",
        "slot_B": "structural slot indicator",
        "issue_advantage": "assembly-derived issue signal",
        "rif": "region-issue interaction",
        "partisan_prior": "historical party-terrain prior",
        "slotA_prior": "slot-specific historical prior",
        "slotB_prior": "slot-specific historical prior",
        "landscape_bloc_alignment": "speech-landscape alignment",
        "landscape_centrist": "speech-landscape centrist axis",
        "landscape_inferred_prior": "landscape-derived prior",
    }
    historical_outcome_predictors = {
        "partisan_prior",
        "slotA_prior",
        "slotB_prior",
        "landscape_inferred_prior",
    }
    vifs = engine.vif(engine.scored_contest_rows(frame), engine.PREDICTORS)
    return pd.DataFrame(
        [
            {
                "predictor": predictor,
                "role": roles[predictor],
                "selection_mechanism": "manual_hard_coded",
                "uses_target_outcome_in_value_construction": False,
                "uses_only_prior_historical_outcomes": predictor
                in historical_outcome_predictors,
                "strictly_outcome_blind_selection_proven": False,
                "vif": vifs[predictor],
            }
            for predictor in engine.PREDICTORS
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--alphas", default="0.3")
    parser.add_argument(
        "--residual-scales",
        default="0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3",
    )
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    frame = engine.assemble()
    inventory = predictor_inventory(frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(PREDICTOR_OUTPUT, index=False, encoding="utf-8-sig")
    if args.inventory_only:
        print(
            f"assembled_columns={len(frame.columns)} predictors={len(engine.PREDICTORS)} "
            f"parameters_with_intercept={len(engine.PREDICTORS) + 1}"
        )
        print(f"saved: {PREDICTOR_OUTPUT}")
        return

    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    scales = [float(value) for value in args.residual_scales.split(",") if value.strip()]
    summary, detail = run_experiment(max(args.n_sim, 100), args.seed, alphas, scales)
    summary.to_csv(RESULTS_OUTPUT, index=False, encoding="utf-8-sig")
    detail.to_csv(DETAIL_OUTPUT, index=False, encoding="utf-8-sig")
    inventory.to_csv(PREDICTOR_OUTPUT, index=False, encoding="utf-8-sig")

    print(
        f"assembled_columns={len(frame.columns)} predictors={len(engine.PREDICTORS)} "
        f"parameters_with_intercept={len(engine.PREDICTORS) + 1}"
    )
    print(summary.to_string(index=False))
    print(f"saved: {RESULTS_OUTPUT}")
    print(f"saved: {DETAIL_OUTPUT}")
    print(f"saved: {PREDICTOR_OUTPUT}")


if __name__ == "__main__":
    main()
