"""Evaluate a concrete-floor-separated target in strict rolling folds.

The regression target is the vote share among voters who are not assigned to
the prior-only regional camp core.  Predicted contestable shares are combined
with that core only after prediction.  All core estimates are point-in-time;
no target-election result is used to construct them.
"""

from __future__ import annotations

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
import evaluate_nested_electorate_learning as gain_eval  # noqa: E402
import issue_vote_engine as engine  # noqa: E402
import rederive_layers_through2022 as rederive  # noqa: E402
import robustness_check as robustness  # noqa: E402


ELECTIONS = base_eval.ALLOWED_ELECTIONS
STRENGTH_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_TUNING_ELECTIONS = 2
MAX_PRIOR_WORSENING_PP = 0.05
OUTPUT_DIR = ROOT / "outputs" / "competitive_electorate_experiment"


def _normalize(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= 1e-12:
        return np.repeat(1.0 / max(len(clipped), 1), len(clipped))
    return clipped / total


def _core_for_group(group: pd.DataFrame, strength: float) -> np.ndarray:
    core = pd.to_numeric(
        group["camp_core_voting_mass"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0).to_numpy(float).copy()
    core *= float(np.clip(strength, 0.0, 1.0))
    total = float(core.sum())
    if total > 0.95:
        core *= 0.95 / total
    return core


def contestable_target(
    frame: pd.DataFrame,
    total_share: np.ndarray | pd.Series,
    strength: float,
) -> np.ndarray:
    """Remove prior-only concrete support from an observed total-share target."""

    work = frame[["election_id", "region_id", "slot", "camp_core_voting_mass"]].copy()
    work["_total"] = np.asarray(total_share, dtype=float)
    result = np.zeros(len(work), dtype=float)
    for indices in work.groupby(["election_id", "region_id"], sort=False).indices.values():
        idx = np.asarray(indices, dtype=int)
        group = work.iloc[idx]
        core = _core_for_group(group, strength)
        residual = np.clip(group["_total"].to_numpy(float) - core, 0.0, None)
        result[idx] = _normalize(residual)
    return result


def recompose_total_share(
    frame: pd.DataFrame,
    contestable_share: np.ndarray | pd.Series,
    strength: float,
) -> np.ndarray:
    """Add regional concrete support without changing postprocess ordering.

    Ridge outputs remain on their raw scale here.  Existing candidate and
    residual adjustments perform the same later compositional normalization as
    the active engine, which makes strength zero an exact reproduction.
    """

    work = frame[["election_id", "region_id", "slot", "camp_core_voting_mass"]].copy()
    work["_contestable"] = np.asarray(contestable_share, dtype=float)
    result = np.zeros(len(work), dtype=float)
    for indices in work.groupby(["election_id", "region_id"], sort=False).indices.values():
        idx = np.asarray(indices, dtype=int)
        group = work.iloc[idx]
        core = _core_for_group(group, strength)
        available = max(1.0 - float(core.sum()), 0.0)
        result[idx] = core + available * group["_contestable"].to_numpy(float)
    return result


def _layered_full_frame() -> tuple[pd.DataFrame, dict[str, Path]]:
    overlay_paths = rederive.write_overlay_variants()
    with rederive.configured(rederive.NEUTRAL_CONFIG, overlay_paths):
        all_rows = engine.assemble()
    scored = robustness.competition_frame(all_rows)
    warmup = robustness.rolling_warmup_frame(all_rows)
    full_order = [*rederive.WARMUP_ELECTIONS, *ELECTIONS]
    order_lookup = {election_id: index for index, election_id in enumerate(full_order)}
    full = pd.concat([warmup, scored], ignore_index=True, sort=False)
    for predictor in engine.PREDICTORS:
        full[predictor] = pd.to_numeric(full[predictor], errors="coerce").fillna(0.0)
    full = full.copy()
    full["_order"] = full["election_id"].map(order_lookup)
    keys = full[["election_id", "region_id", "slot", "candidate_name", "bloc"]].copy()
    landscape = engine._candidate_political_landscape_features(keys)
    keys = keys.merge(
        landscape,
        on=["election_id", "region_id", "slot"],
        how="left",
        validate="one_to_one",
    )
    mass = layers.estimate_electorate_layers(
        keys,
        base_eval._read_csv(base_eval.HISTORY_PATH),
        mass_profile="direct_party_layers",
    )
    full = full.drop(columns=["camp_core_voting_mass"], errors="ignore").merge(
        mass[["election_id", "region_id", "slot", "camp_core_voting_mass"]],
        on=["election_id", "region_id", "slot"],
        how="left",
        validate="one_to_one",
    )
    full["camp_core_voting_mass"] = pd.to_numeric(
        full["camp_core_voting_mass"], errors="coerce"
    ).fillna(0.0)
    return full, overlay_paths


def build_base_predictions(
    full: pd.DataFrame,
    overlay_paths: dict[str, Path],
    strength: float,
) -> pd.DataFrame:
    config_rows = base_eval._read_csv(
        ROOT
        / "presidential_issue_engine"
        / "report"
        / "through2022_rederived"
        / "nested_outer_results.csv"
    )
    order_lookup = {
        election_id: index
        for index, election_id in enumerate([*rederive.WARMUP_ELECTIONS, *ELECTIONS])
    }
    warmup_ids = set(rederive.WARMUP_ELECTIONS)
    outputs: list[pd.DataFrame] = []
    for target in ELECTIONS:
        selected_row = config_rows.loc[config_rows["target_election"].eq(target)]
        if len(selected_row) != 1:
            raise RuntimeError(f"expected one nested config for {target}")
        selected = base_eval._layer_config_from_row(selected_row.iloc[0])
        test = engine.scored_contest_rows(full.loc[full["election_id"].eq(target)]).copy()
        train = full.loc[full["_order"] < order_lookup[target]].copy()
        train["_rolling_target"] = engine.normalized_vote_share_target(train)
        train, residual_mask = engine.rolling_training_with_slot_backfill(
            train, test, warmup_ids
        )
        total_target = engine.normalized_vote_share_target(train)
        train["_contestable_target"] = contestable_target(
            train, total_target, strength
        )
        with rederive.configured(selected, overlay_paths):
            x_train = train[engine.PREDICTORS].to_numpy(float)
            x_test = test[engine.PREDICTORS].to_numpy(float)
            beta, _, _, _, means, scales = engine.ridge_fit(
                x_train,
                train["_contestable_target"].to_numpy(float),
                alpha=selected.ridge_alpha,
                sample_weight=engine.election_epoch_sample_weight(train),
            )
            train_contestable = engine.ridge_predict(beta, x_train, means, scales)
            test_contestable = engine.ridge_predict(beta, x_test, means, scales)
            train_pred = recompose_total_share(train, train_contestable, strength)
            pred = recompose_total_share(test, test_contestable, strength)
            train_pred = engine.apply_third_candidate_prediction_adjustment(train, train_pred)
            train_pred = engine.apply_withdrawn_candidate_prediction_adjustment(train, train_pred)
            pred = engine.apply_third_candidate_prediction_adjustment(test, pred)
            pred = engine.apply_withdrawn_candidate_prediction_adjustment(test, pred)
            pred = engine.apply_region_residual_calibration(
                train.loc[residual_mask].copy(),
                test,
                train_pred[residual_mask],
                pred,
            )
            pred = engine.normalize_vote_share_predictions(test, pred)
            pred = engine.apply_prediction_postprocess(
                test,
                pred,
                party_tone=False,
                electorate_layer=False,
            )
        output = test[["election_id", "region_id", "slot"]].copy()
        output["pred"] = pred
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def _apply_preference_schedule(
    frame: pd.DataFrame,
    gains: dict[str, float],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for target in ELECTIONS:
        target_frame = frame.loc[frame["election_id"].eq(target)].copy()
        parts.append(
            base_eval.apply_config(
                target_frame,
                layers.ElectorateLayerConfig(
                    preference_gain=float(gains[target]),
                    mass_profile="direct_party_layers",
                ),
            )
        )
    return pd.concat(parts, ignore_index=True)


def _baseline_preference_schedule(frame: pd.DataFrame) -> dict[str, float]:
    cache: dict[float, pd.Series] = {}
    result: dict[str, float] = {}
    for index, target in enumerate(ELECTIONS):
        gain, _ = gain_eval.select_preference_gain(
            frame,
            ELECTIONS[:index],
            selection_label=f"preference_outer_{target}",
            gain_grid=gain_eval.CAPPED_GAIN_GRID,
            metric_cache=cache,
        )
        result[target] = float(gain)
    return result


def _by_election(frame: pd.DataFrame) -> pd.Series:
    return base_eval.election_weighted_mae(frame, "layer_pred").set_index(
        "election_id"
    )["weighted_row_mae_pp"]


def attach_prior_region_weights(
    frame: pd.DataFrame,
    full_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest prior presidential regional vote volume.

    The value is a point-in-time proxy for regional population/electorate size.
    Target-election turnout and vote totals are deliberately excluded.
    """

    out = frame.copy()
    out["prior_region_weight"] = 1.0
    order = [*rederive.WARMUP_ELECTIONS, *ELECTIONS]
    for index, target in enumerate(order):
        if target not in ELECTIONS or index == 0:
            continue
        prior_id = order[index - 1]
        prior = full_history.loc[full_history["election_id"].eq(prior_id)].copy()
        if prior.empty or "votes" not in prior.columns:
            continue
        weights = pd.to_numeric(prior["votes"], errors="coerce").fillna(0.0).groupby(
            prior["region_id"]
        ).sum()
        positive = weights.loc[weights.gt(0.0)]
        fallback = float(positive.min()) if not positive.empty else 1.0
        target_index = out["election_id"].eq(target)
        mapped = out.loc[target_index, "region_id"].map(weights).fillna(fallback)
        mapped = mapped.clip(lower=max(fallback * 0.10, 1.0))
        out.loc[target_index, "prior_region_weight"] = (
            mapped / max(float(mapped.mean()), 1e-12)
        ).to_numpy(float)
    return out


def preserve_candidate_means(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Rake regions while preserving prior-volume-weighted candidate means.

    This keeps the experiment confined to regional shape.  The target column
    totals use the outcome-blind baseline and latest prior-election regional
    volume, never actual target-election turnout or vote totals.
    """

    keys = ["election_id", "region_id", "slot"]
    weight_column = "prior_region_weight"
    baseline_values = baseline[keys + ["layer_pred"]].rename(
        columns={"layer_pred": "_baseline_pred"}
    )
    out = candidate.merge(
        baseline_values, on=keys, how="left", validate="one_to_one"
    )
    raked = np.zeros(len(out), dtype=float)
    for _, election in out.groupby("election_id", sort=False):
        regions = list(dict.fromkeys(election["region_id"].astype(str)))
        slots = list(dict.fromkeys(election["slot"].astype(str)))
        row_lookup = {region: index for index, region in enumerate(regions)}
        column_lookup = {slot: index for index, slot in enumerate(slots)}
        matrix = np.full((len(regions), len(slots)), 1e-9, dtype=float)
        region_weights = np.ones(len(regions), dtype=float)
        target = np.zeros(len(slots), dtype=float)
        index_matrix = np.full((len(regions), len(slots)), -1, dtype=int)
        for index, row in election.iterrows():
            r = row_lookup[str(row["region_id"])]
            c = column_lookup[str(row["slot"])]
            matrix[r, c] = max(float(row["layer_pred"]), 1e-9)
            weight = max(float(row[weight_column]), 1e-12)
            region_weights[r] = weight
            target[c] += weight * max(float(row["_baseline_pred"]), 0.0)
            index_matrix[r, c] = int(index)
        weighted_matrix = matrix * region_weights[:, None]
        for _ in range(500):
            previous = weighted_matrix.copy()
            weighted_matrix *= target[None, :] / np.maximum(
                weighted_matrix.sum(axis=0), 1e-12
            )
            weighted_matrix *= region_weights[:, None] / np.maximum(
                weighted_matrix.sum(axis=1, keepdims=True), 1e-12
            )
            if float(np.max(np.abs(weighted_matrix - previous))) < 1e-13:
                break
        matrix = weighted_matrix / region_weights[:, None]
        for r in range(len(regions)):
            for c in range(len(slots)):
                if index_matrix[r, c] >= 0:
                    raked[index_matrix[r, c]] = matrix[r, c]
    out["layer_pred"] = raked
    return out.drop(columns=["_baseline_pred"])


def _regional_slope(frame: pd.DataFrame, election_id: str) -> float:
    slopes: list[float] = []
    for _, group in frame.loc[frame["election_id"].eq(election_id)].groupby("slot"):
        actual = group["actual"].to_numpy(float)
        pred = group["layer_pred"].to_numpy(float)
        weights = group["contest_votes"].to_numpy(float)
        actual_mean = float(np.average(actual, weights=weights))
        pred_mean = float(np.average(pred, weights=weights))
        variance = float(np.average((actual - actual_mean) ** 2, weights=weights))
        if variance > 1e-12:
            slopes.append(
                float(
                    np.average(
                        (actual - actual_mean) * (pred - pred_mean), weights=weights
                    )
                    / variance
                )
            )
    return float(np.mean(slopes)) if slopes else float("nan")


def weighted_error_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate vote-volume-weighted national bias from centered residual error."""

    rows: list[dict[str, object]] = []
    for (election_id, slot), group in frame.groupby(["election_id", "slot"], sort=True):
        weights = pd.to_numeric(group["contest_votes"], errors="coerce").fillna(0.0).to_numpy(float)
        error = (
            group["layer_pred"].to_numpy(float) - group["actual"].to_numpy(float)
        ) * 100.0
        bias = float(np.average(error, weights=weights))
        rows.append(
            {
                "election_id": election_id,
                "slot": slot,
                "regional_contest_votes": int(
                    group[["region_id", "contest_votes"]]
                    .drop_duplicates()["contest_votes"]
                    .sum()
                ),
                "weighted_total_regional_mae_pp": float(
                    np.average(np.abs(error), weights=weights)
                ),
                "weighted_national_bias_pp": bias,
                "weighted_centered_residual_mae_pp": float(
                    np.average(np.abs(error - bias), weights=weights)
                ),
                "evaluation_weight": "target_election_actual_contest_votes",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_frame = base_eval.prepare_frame(mass_profile="direct_party_layers")
    full, overlay_paths = _layered_full_frame()
    evaluation_frame = attach_prior_region_weights(evaluation_frame, full)
    preference_gains = _baseline_preference_schedule(evaluation_frame)
    baseline = _apply_preference_schedule(evaluation_frame, preference_gains)
    baseline_by = _by_election(baseline)
    decomposition = weighted_error_decomposition(baseline)
    candidates: dict[float, pd.DataFrame] = {}
    metrics: dict[float, pd.Series] = {}
    fixed_rows: list[dict[str, object]] = []
    for strength in STRENGTH_GRID:
        predictions = build_base_predictions(full, overlay_paths, strength)
        candidate_frame = evaluation_frame.drop(columns=["pred"]).merge(
            predictions,
            on=["election_id", "region_id", "slot"],
            how="left",
            validate="one_to_one",
        )
        evaluated = _apply_preference_schedule(candidate_frame, preference_gains)
        regional_only = preserve_candidate_means(evaluated, baseline)
        candidates[strength] = regional_only
        metrics[strength] = _by_election(regional_only)
        raw_metrics = _by_election(evaluated)
        fixed_rows.append(
            {
                "core_separation_strength": strength,
                "weighted_macro_mae_pp": float(metrics[strength].mean()),
                "unconstrained_weighted_macro_mae_pp": float(raw_metrics.mean()),
                "max_prediction_shift_pp": float(
                    np.max(
                        np.abs(
                            regional_only["layer_pred"].to_numpy(float)
                            - baseline["layer_pred"].to_numpy(float)
                        )
                    )
                    * 100.0
                ),
                **{
                    f"{election_id}_mae_pp": float(metrics[strength].loc[election_id])
                    for election_id in ELECTIONS
                },
                **{
                    f"{election_id}_regional_slope": _regional_slope(
                        regional_only, election_id
                    )
                    for election_id in ELECTIONS
                },
            }
        )

    zero_drift = float(
        np.max(
            np.abs(
                candidates[0.0]["layer_pred"].to_numpy(float)
                - baseline["layer_pred"].to_numpy(float)
            )
        )
    )
    if zero_drift > 1e-10:
        raise RuntimeError(f"zero-strength path does not reproduce baseline: {zero_drift}")

    selected_parts: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    for index, target in enumerate(ELECTIONS):
        prior = ELECTIONS[:index]
        selected = 0.0
        if len(prior) >= MIN_TUNING_ELECTIONS:
            baseline_prior = metrics[0.0].loc[list(prior)]
            required = math.ceil(len(prior) / 2.0)
            options: list[tuple[float, float]] = []
            for strength in STRENGTH_GRID:
                candidate_prior = metrics[strength].loc[list(prior)]
                improvement = baseline_prior - candidate_prior
                eligible = bool(
                    strength == 0.0
                    or (
                        int((improvement > 1e-12).sum()) >= required
                        and float((-improvement).clip(lower=0.0).max())
                        <= MAX_PRIOR_WORSENING_PP
                    )
                )
                if eligible:
                    options.append((float(candidate_prior.mean()), strength))
            _, selected = min(options, key=lambda item: (item[0], item[1]))
        target_part = candidates[selected].loc[
            candidates[selected]["election_id"].eq(target)
        ].copy()
        selected_parts.append(target_part)
        selected_rows.append(
            {
                "target_election": target,
                "tuning_elections": "|".join(prior),
                "selected_core_separation_strength": selected,
                "target_excluded_from_tuning": target not in prior,
            }
        )
    nested = pd.concat(selected_parts, ignore_index=True)
    nested_by = _by_election(nested)
    comparison = pd.DataFrame(
        {
            "election_id": ELECTIONS,
            "baseline_mae_pp": baseline_by.loc[list(ELECTIONS)].to_numpy(),
            "separated_mae_pp": nested_by.loc[list(ELECTIONS)].to_numpy(),
        }
    )
    comparison["improvement_pp"] = comparison["baseline_mae_pp"] - comparison["separated_mae_pp"]
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(OUTPUT_DIR / "fixed_strength_comparison.csv", index=False, encoding="utf-8-sig")
    decomposition.to_csv(
        OUTPUT_DIR / "weighted_error_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(OUTPUT_DIR / "nested_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(
        OUTPUT_DIR / "outer_selected_strengths.csv", index=False, encoding="utf-8-sig"
    )
    nested.to_csv(OUTPUT_DIR / "nested_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        "scope": {
            "scored_elections": list(ELECTIONS),
            "warmup_elections": list(rederive.WARMUP_ELECTIONS),
            "post_2022_presidential_outcomes_loaded": False,
            "target_definition": "(total share - PIT regional camp core) / contestable mass",
            "regional_size_weight": (
                "latest prior presidential contest vote volume for prediction constraints; "
                "target contest vote volume only for post-election MAE"
            ),
        },
        "preference_gain_schedule": preference_gains,
        "zero_strength_reproduction_max_abs": zero_drift,
        "baseline_weighted_macro_mae_pp": float(baseline_by.mean()),
        "strict_nested_weighted_macro_mae_pp": float(nested_by.mean()),
        "strict_nested_improvement_pp": float(baseline_by.mean() - nested_by.mean()),
        "weighted_error_decomposition_artifact": (
            "outputs/competitive_electorate_experiment/weighted_error_decomposition.csv"
        ),
        "outer_configs": selected_rows,
        "mechanism": {
            "fixed_layer": "prior-only regional camp concrete support",
            "competitive_layer": "critical plus middle/swing voters",
            "strength_grid": list(STRENGTH_GRID),
            "maximum_prior_worsening_pp": MAX_PRIOR_WORSENING_PP,
        },
        "caveat": (
            "The hypothesis and grid were defined after inspecting 2002-2022 diagnostics; "
            "fold targets remain excluded, but this is not an untouched holdout."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(fixed.to_string(index=False))
    print("\nStrict nested comparison")
    print(comparison.to_string(index=False))
    print("\n" + json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
