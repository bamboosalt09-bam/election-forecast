"""Strict shadow nested evaluation of outcome-blind preliminary slots.

Old realized-rank slot predictors are forbidden. Forecast-safe replacements
are derived from the separately generated preliminary assignment and are not
enabled until two prior scored elections are available. Training and target
folds use the same scored-candidate denominator.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import electorate_layers as layers  # noqa: E402
import issue_vote_engine as engine  # noqa: E402
from presidential_issue_engine.preliminary_slots import (  # noqa: E402
    apply_hierarchical_third_constraint,
)
import evaluate_electorate_layers as base_eval  # noqa: E402
import evaluate_nested_electorate_learning as gain_eval  # noqa: E402
import rederive_layers_through2022 as rederive  # noqa: E402
import robustness_check as robustness  # noqa: E402


ELECTIONS = base_eval.ALLOWED_ELECTIONS
OUTPUT_DIR = ROOT / "outputs" / "preliminary_slot_shadow_nested"
ASSIGNMENT_PATH = (
    ROOT
    / "outputs"
    / "preliminary_slot_assignment"
    / "candidate_slot_assignments_v2.csv"
)
CONFIG_PATH = (
    ROOT
    / "presidential_issue_engine"
    / "report"
    / "through2022_rederived"
    / "nested_outer_results.csv"
)
ACTIVE_PATH = ROOT / "outputs" / "electorate_nested_learning" / "nested_predictions.csv"
OLD_SLOT_PREDICTORS = frozenset({"slot_A", "slot_B", "slotA_prior", "slotB_prior"})
BASE_PREDICTORS = (
    "issue_advantage",
    "rif",
    "partisan_prior",
    "landscape_bloc_alignment",
    "landscape_centrist",
    "landscape_inferred_prior",
)
DIRECT_PRELIMINARY = ("prelim_slot_A", "prelim_slot_B", *BASE_PREDICTORS)
ALL_PRELIMINARY = (
    "prelim_slot_A",
    "prelim_slot_B",
    "issue_advantage",
    "rif",
    "partisan_prior",
    "prelim_slotA_prior",
    "prelim_slotB_prior",
    "landscape_bloc_alignment",
    "landscape_centrist",
    "landscape_inferred_prior",
)
SHARE_PRELIMINARY = ("preliminary_mean_share", *BASE_PREDICTORS)
WITHDRAWAL_PRELIMINARY = (
    "prelim_withdrawal_share",
    "prelim_withdrawal_event",
    *BASE_PREDICTORS,
)
VARIANTS = {
    "slot_free_roles_only": BASE_PREDICTORS,
    "slot_free_hierarchy_no_neutral": BASE_PREDICTORS,
    "preliminary_direct_min2": DIRECT_PRELIMINARY,
    "preliminary_all_min2": ALL_PRELIMINARY,
    "preliminary_hierarchy_min2": ALL_PRELIMINARY,
    "preliminary_hierarchy_no_neutral_min2": ALL_PRELIMINARY,
    "preliminary_share_hierarchy_no_neutral_min2": SHARE_PRELIMINARY,
    "preliminary_share_hierarchy_no_neutral_no_transfer_min2": SHARE_PRELIMINARY,
    "preliminary_withdrawal_hierarchy_no_neutral_min2": WITHDRAWAL_PRELIMINARY,
    "preliminary_withdrawal_hierarchy_no_neutral_no_transfer_min2": WITHDRAWAL_PRELIMINARY,
    "preliminary_withdrawal_target_gated_no_neutral_min2": WITHDRAWAL_PRELIMINARY,
    "preliminary_withdrawal_target_gated_no_neutral_no_transfer_min2": WITHDRAWAL_PRELIMINARY,
    "preliminary_withdrawal_target_gated_no_neutral_replace_transfer_min2": WITHDRAWAL_PRELIMINARY,
}
MIN_PRIOR_SCORED_FOR_PRELIMINARY_COLUMNS = 2


def _orthogonalize_predictor_pairs(
    x_train: np.ndarray,
    x_test: np.ndarray,
    predictors: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    """Residualize correlated predictors using training X only.

    Each pair is ``(base, residualized)``. The fitted intercept and slope never
    inspect the response or target-election rows, so the transform remains
    valid inside a strict rolling fold.
    """

    train = np.asarray(x_train, dtype=float).copy()
    test = np.asarray(x_test, dtype=float).copy()
    names = tuple(predictors)
    audit: list[dict[str, float | str]] = []
    for base_name, residual_name in pairs:
        if base_name not in names or residual_name not in names:
            raise ValueError(
                f"orthogonalization pair is not in predictor set: "
                f"{base_name}->{residual_name}"
            )
        if base_name == residual_name:
            raise ValueError("a predictor cannot be residualized against itself")
        base_index = names.index(base_name)
        residual_index = names.index(residual_name)
        design = np.column_stack(
            [np.ones(len(train), dtype=float), train[:, base_index]]
        )
        intercept, slope = np.linalg.lstsq(
            design, train[:, residual_index], rcond=None
        )[0]
        train[:, residual_index] -= intercept + slope * train[:, base_index]
        test[:, residual_index] -= intercept + slope * test[:, base_index]
        audit.append(
            {
                "base_predictor": base_name,
                "residualized_predictor": residual_name,
                "training_intercept": float(intercept),
                "training_slope": float(slope),
            }
        )
    return train, test, audit


def _maximum_predictor_vif(x: np.ndarray) -> float:
    """Return the largest finite VIF after ignoring exact duplicate columns.

    Exact duplicates are a separate basis-definition defect and cannot rank
    fold instability because they occur in every strict fold. If every
    non-constant predictor is exactly collinear, infinity is returned.
    """

    matrix = np.asarray(x, dtype=float)
    variable = np.var(matrix, axis=0) > np.finfo(float).tiny
    matrix = matrix[:, variable]
    if matrix.shape[1] < 2:
        return 1.0
    values: list[float] = []
    for index in range(matrix.shape[1]):
        y = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(len(matrix), dtype=float), others])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        total = float(np.square(y - y.mean()).sum())
        residual = float(np.square(y - fitted).sum())
        if total <= np.finfo(float).tiny:
            values.append(float("inf"))
            continue
        r_squared = min(max(1.0 - residual / total, 0.0), 1.0)
        values.append(
            float("inf")
            if r_squared >= 1.0 - 1e-12
            else 1.0 / (1.0 - r_squared)
        )
    finite = [value for value in values if np.isfinite(value)]
    return max(finite) if finite else float("inf")


def _neutral_overlay_paths() -> dict[float, Path | None]:
    return {0.0: None}


def _prepare_rows() -> pd.DataFrame:
    assignments = pd.read_csv(ASSIGNMENT_PATH, encoding="utf-8-sig")
    forbidden = {
        "actual",
        "actual_share",
        "actual_vote_share",
        "candidate_votes",
        "contest_votes",
        "vote_share",
        "votes",
        "realized_rank",
        "winner",
    }
    present = sorted(forbidden.intersection(assignments.columns))
    if present:
        raise RuntimeError(f"preliminary assignments contain outcomes: {present}")
    assignment_columns = [
        "election_id",
        "candidate_name",
        "assigned_slot",
        "preliminary_mean_share",
        "pre_withdrawal_mean_share",
        "post_withdrawal_mean_share",
        "withdrawal_event_applied",
    ]
    assignments = assignments[assignment_columns].copy()

    paths = _neutral_overlay_paths()
    with rederive.configured(rederive.NEUTRAL_CONFIG, paths):
        assembled = engine.assemble()
    scored = engine.scored_contest_rows(robustness.competition_frame(assembled))
    scored = scored.merge(
        assignments,
        on=["election_id", "candidate_name"],
        how="left",
        validate="many_to_one",
    )
    if scored["assigned_slot"].isna().any():
        missing = scored.loc[
            scored["assigned_slot"].isna(), ["election_id", "candidate_name"]
        ].drop_duplicates()
        raise RuntimeError(f"missing preliminary assignments: {missing.to_dict('records')}")
    scored["source_slot"] = scored["slot"]
    scored["slot"] = scored["assigned_slot"]
    scored["prelim_slot_A"] = scored["slot"].eq("A").astype(float)
    scored["prelim_slot_B"] = scored["slot"].eq("B").astype(float)
    scored["prelim_slotA_prior"] = scored["prelim_slot_A"] * scored["partisan_prior"]
    scored["prelim_slotB_prior"] = scored["prelim_slot_B"] * scored["partisan_prior"]
    event = scored["withdrawal_event_applied"].fillna(False).astype(bool)
    scored["prelim_withdrawal_event"] = event.astype(float)
    scored["prelim_withdrawal_share"] = np.where(
        event,
        pd.to_numeric(scored["post_withdrawal_mean_share"], errors="coerce").fillna(0.0),
        0.0,
    )

    warmup = robustness.rolling_warmup_frame(assembled)
    warmup["source_slot"] = warmup["slot"]
    warmup["preliminary_mean_share"] = np.nan
    for column in (
        "prelim_slot_A",
        "prelim_slot_B",
        "prelim_slotA_prior",
        "prelim_slotB_prior",
        "prelim_withdrawal_share",
        "prelim_withdrawal_event",
    ):
        warmup[column] = 0.0
    order = [*rederive.WARMUP_ELECTIONS, *ELECTIONS]
    lookup = {election_id: index for index, election_id in enumerate(order)}
    full = pd.concat([warmup, scored], ignore_index=True, sort=False).copy()
    full["_order"] = full["election_id"].map(lookup)
    return full


def _predictors_for_fold(
    full: pd.DataFrame,
    variant: str,
    target_index: int,
    target: str,
) -> tuple[str, ...]:
    if variant == "slot_free_roles_only":
        predictors = BASE_PREDICTORS
    elif "withdrawal_target_gated" in variant:
        prior_ids = set(ELECTIONS[:target_index])
        prior_event_count = full.loc[
            full["election_id"].isin(prior_ids)
            & pd.to_numeric(full["prelim_withdrawal_event"], errors="coerce").fillna(0.0).gt(0.0),
            "election_id",
        ].nunique()
        target_has_event = full.loc[
            full["election_id"].eq(target), "prelim_withdrawal_event"
        ].fillna(0.0).astype(float).gt(0.0).any()
        predictors = (
            VARIANTS[variant]
            if target_has_event and prior_event_count >= MIN_PRIOR_SCORED_FOR_PRELIMINARY_COLUMNS
            else BASE_PREDICTORS
        )
    elif target_index < MIN_PRIOR_SCORED_FOR_PRELIMINARY_COLUMNS:
        predictors = BASE_PREDICTORS
    else:
        predictors = VARIANTS[variant]
    overlap = OLD_SLOT_PREDICTORS.intersection(predictors)
    if overlap:
        raise RuntimeError(f"old slot predictors leaked into fold: {sorted(overlap)}")
    return tuple(predictors)


def _build_outer_predictions(
    full: pd.DataFrame,
    variant: str,
    *,
    layer_config_overrides: Mapping[str, object] | None = None,
    layer_config_overrides_by_target: Mapping[
        str, Mapping[str, object]
    ] | None = None,
    predictor_orthogonalization_pairs: Sequence[tuple[str, str]] = (),
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    configs = pd.read_csv(CONFIG_PATH, encoding="utf-8-sig")
    paths = _neutral_overlay_paths()
    order = [*rederive.WARMUP_ELECTIONS, *ELECTIONS]
    lookup = {election_id: index for index, election_id in enumerate(order)}
    warmup_ids = set(rederive.WARMUP_ELECTIONS)
    outputs: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for target_index, target in enumerate(ELECTIONS):
        predictors = _predictors_for_fold(full, variant, target_index, target)
        work = full.copy()
        for predictor in predictors:
            work[predictor] = pd.to_numeric(
                work[predictor], errors="coerce"
            ).fillna(0.0)
        selected_row = configs.loc[configs["target_election"].eq(target)]
        if len(selected_row) != 1:
            raise RuntimeError(f"expected one frozen config for {target}")
        selected = base_eval._layer_config_from_row(selected_row.iloc[0])
        if "no_neutral" in variant:
            selected = replace(selected, neutral_context_scale=0.0)
        if layer_config_overrides:
            selected = replace(selected, **dict(layer_config_overrides))
        if layer_config_overrides_by_target and target in layer_config_overrides_by_target:
            selected = replace(
                selected, **dict(layer_config_overrides_by_target[target])
            )
        test = work.loc[work["election_id"].eq(target)].copy().reset_index(drop=True)
        train = work.loc[work["_order"] < lookup[target]].copy().reset_index(drop=True)
        train["_rolling_target"] = engine.normalized_vote_share_target(train)
        train, residual_mask = engine.rolling_training_with_slot_backfill(
            train, test, warmup_ids
        )
        with rederive.configured(selected, paths):
            x_train = train[list(predictors)].to_numpy(float)
            x_test = test[list(predictors)].to_numpy(float)
            raw_max_predictor_vif = _maximum_predictor_vif(x_train)
            x_train, x_test, orthogonalization_audit = (
                _orthogonalize_predictor_pairs(
                    x_train,
                    x_test,
                    predictors,
                    predictor_orthogonalization_pairs,
                )
            )
            beta, _, _, _, means, scales = engine.ridge_fit(
                x_train,
                train["_rolling_target"].to_numpy(float),
                alpha=selected.ridge_alpha,
                sample_weight=engine.election_epoch_sample_weight(train),
            )
            train_pred = engine.ridge_predict(beta, x_train, means, scales)
            pred = engine.ridge_predict(beta, x_test, means, scales)
            train_pred = engine.apply_third_candidate_prediction_adjustment(train, train_pred)
            pred = engine.apply_third_candidate_prediction_adjustment(test, pred)
            withdrawal_model_enabled = set(WITHDRAWAL_PRELIMINARY[:2]).issubset(predictors)
            replace_old_transfer = (
                "replace_transfer" in variant and withdrawal_model_enabled
            )
            apply_old_transfer = "no_transfer" not in variant and not replace_old_transfer
            if apply_old_transfer:
                train_pred = engine.apply_withdrawn_candidate_prediction_adjustment(
                    train, train_pred
                )
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
            pre_hierarchy_pred = pred.copy()
            if variant in {
                "slot_free_hierarchy_no_neutral",
                "preliminary_hierarchy_min2",
                "preliminary_hierarchy_no_neutral_min2",
                "preliminary_share_hierarchy_no_neutral_min2",
                "preliminary_share_hierarchy_no_neutral_no_transfer_min2",
                "preliminary_withdrawal_hierarchy_no_neutral_min2",
                "preliminary_withdrawal_hierarchy_no_neutral_no_transfer_min2",
                "preliminary_withdrawal_target_gated_no_neutral_min2",
                "preliminary_withdrawal_target_gated_no_neutral_no_transfer_min2",
                "preliminary_withdrawal_target_gated_no_neutral_replace_transfer_min2",
            }:
                prior_id = (
                    rederive.WARMUP_ELECTIONS[-1]
                    if target_index == 0
                    else ELECTIONS[target_index - 1]
                )
                prior = work.loc[work["election_id"].eq(prior_id)].copy()
                if "votes" in prior.columns:
                    prior_votes = pd.to_numeric(prior["votes"], errors="coerce").fillna(0.0)
                else:
                    prior_votes = pd.Series(0.0, index=prior.index)
                prior_volume = prior_votes.groupby(prior["region_id"]).sum()
                positive_volume = prior_volume.loc[prior_volume.gt(0.0)]
                if (
                    positive_volume.empty
                    or float(positive_volume.max() / positive_volume.min()) < 1.5
                ):
                    hierarchy_weights = np.ones(len(test), dtype=float)
                else:
                    fallback_volume = float(positive_volume.min())
                    hierarchy_weights = (
                        test["region_id"]
                        .map(prior_volume)
                        .fillna(fallback_volume)
                        .to_numpy(float)
                    )
                pred, _ = apply_hierarchical_third_constraint(
                    test,
                    pred,
                    region_weights=hierarchy_weights,
                )
        output = test[
            [
                "election_id",
                "region_id",
                "source_slot",
                "slot",
                "preliminary_mean_share",
            ]
        ].copy()
        output = output.rename(
            columns={"source_slot": "join_slot", "slot": "preliminary_slot"}
        )
        output["shadow_pred"] = pred
        output["pre_hierarchy_pred"] = pre_hierarchy_pred
        outputs.append(output)
        prior_withdrawal_elections = full.loc[
            full["election_id"].isin(ELECTIONS[:target_index])
            & pd.to_numeric(full["prelim_withdrawal_event"], errors="coerce").fillna(0.0).gt(0.0),
            "election_id",
        ].nunique()
        audit_row = {
                "variant": variant,
                "target_election": target,
                "training_elections": "|".join(
                    [*rederive.WARMUP_ELECTIONS, *ELECTIONS[:target_index]]
                ),
                "predictors": "|".join(predictors),
                "predictor_count": len(predictors),
                "preliminary_columns_enabled": bool(
                    set(predictors).difference(BASE_PREDICTORS)
                ),
                "target_excluded_from_fit": True,
                "old_slot_predictors_used": False,
                "consistent_scored_denominator": True,
                "neutral_context_direct_adjustment": (
                    "no_neutral" not in variant
                ),
                "withdrawn_candidate_direct_adjustment": bool(apply_old_transfer),
                "withdrawal_model_enabled": bool(withdrawal_model_enabled),
                "target_has_withdrawal_event": bool(
                    pd.to_numeric(test["prelim_withdrawal_event"], errors="coerce")
                    .fillna(0.0)
                    .gt(0.0)
                    .any()
                ),
                "prior_withdrawal_elections": int(prior_withdrawal_elections),
                "predictor_orthogonalization": "|".join(
                    f"{row['base_predictor']}->{row['residualized_predictor']}"
                    for row in orthogonalization_audit
                ),
                "raw_max_predictor_vif": raw_max_predictor_vif,
            }
        for name, value in zip(("intercept", *predictors), beta, strict=True):
            audit_row[f"standardized_coef_{name}"] = float(value)
        for name, value in asdict(selected).items():
            audit_row[f"layer_config_{name}"] = value
        audit.append(audit_row)
    return pd.concat(outputs, ignore_index=True), audit


def _base_layer_frame(*, require_frozen_reproduction: bool = True) -> pd.DataFrame:
    original = rederive.write_overlay_variants
    rederive.write_overlay_variants = _neutral_overlay_paths
    try:
        return base_eval.prepare_frame(
            require_frozen_reproduction=require_frozen_reproduction
        )
    finally:
        rederive.write_overlay_variants = original


def _attach_layers(base: pd.DataFrame, outer: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(
        outer,
        left_on=["election_id", "region_id", "slot"],
        right_on=["election_id", "region_id", "join_slot"],
        how="left",
        validate="one_to_one",
    )
    if merged["shadow_pred"].isna().any():
        raise RuntimeError("shadow prediction coverage is incomplete")
    merged["source_slot"] = merged["slot"]
    merged["slot"] = merged["preliminary_slot"]
    merged["pred"] = merged["shadow_pred"]
    return merged


def _apply_nested_preference(
    frame: pd.DataFrame,
    variant: str,
    *,
    preference_gain_floor: float = 0.0,
    terrain_gain_by_target: Mapping[str, float] | None = None,
    regional_accent_gain_by_target: Mapping[str, float] | None = None,
    regional_accent_signal_width: float = 0.10,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    parts: list[pd.DataFrame] = []
    configs: list[dict[str, object]] = []
    cache: dict[float, pd.Series] = {}
    for index, target in enumerate(ELECTIONS):
        prior = ELECTIONS[:index]
        gain, _ = gain_eval.select_preference_gain(
            frame,
            prior,
            selection_label=f"{variant}_{target}",
            gain_grid=gain_eval.CAPPED_GAIN_GRID,
            metric_cache=cache,
        )
        gain = max(float(gain), max(float(preference_gain_floor), 0.0))
        terrain_gain = max(float((terrain_gain_by_target or {}).get(target, 0.0)), 0.0)
        regional_accent_gain = max(
            float((regional_accent_gain_by_target or {}).get(target, 0.0)),
            0.0,
        )
        target_frame = frame.loc[frame["election_id"].eq(target)].copy()
        evaluated = base_eval.apply_config(
            target_frame,
            layers.ElectorateLayerConfig(
                terrain_anchor_gain=terrain_gain,
                regional_accent_gain=regional_accent_gain,
                regional_accent_signal_width=regional_accent_signal_width,
                preference_gain=gain,
                mass_profile="direct_party_layers",
            ),
        )
        parts.append(evaluated)
        configs.append(
            {
                "variant": variant,
                "target_election": target,
                "tuning_elections": "|".join(prior),
                "preference_gain": gain,
                "terrain_anchor_gain": terrain_gain,
                "regional_accent_gain": regional_accent_gain,
                "regional_accent_signal_width": regional_accent_signal_width,
                "target_excluded_from_tuning": target not in prior,
            }
        )
    return pd.concat(parts, ignore_index=True), configs


def _metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    variant: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    regional = base_eval.election_weighted_mae(frame, prediction_column).rename(
        columns={"weighted_row_mae_pp": "regional_weighted_mae_pp"}
    )
    national_rows: list[dict[str, object]] = []
    key = "source_slot" if "source_slot" in frame.columns else "slot"
    name_column = "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"
    for (election_id, candidate_key), group in frame.groupby(
        ["election_id", key], sort=True
    ):
        weights = pd.to_numeric(group["contest_votes"], errors="coerce").fillna(0.0).to_numpy(float)
        pred = float(np.average(group[prediction_column], weights=weights))
        actual = float(np.average(group["actual"], weights=weights))
        national_rows.append(
            {
                "variant": variant,
                "election_id": election_id,
                "candidate_key": candidate_key,
                "candidate_name": group[name_column].iloc[0],
                "pred_pct": pred * 100.0,
                "actual_pct": actual * 100.0,
                "error_pp": (pred - actual) * 100.0,
                "abs_error_pp": abs(pred - actual) * 100.0,
            }
        )
    national = pd.DataFrame(national_rows)
    national_by = (
        national.groupby("election_id", as_index=False)["abs_error_pp"]
        .mean()
        .rename(columns={"abs_error_pp": "national_candidate_mae_pp"})
    )
    regional = regional.merge(national_by, on="election_id", how="left")
    regional.insert(0, "variant", variant)
    winner_correct = []
    for _, group in national.groupby("election_id"):
        winner_correct.append(
            group.loc[group["pred_pct"].idxmax(), "candidate_key"]
            == group.loc[group["actual_pct"].idxmax(), "candidate_key"]
        )
    summary = {
        "variant": variant,
        "regional_equal_election_macro_mae_pp": float(
            regional["regional_weighted_mae_pp"].mean()
        ),
        "national_equal_election_macro_mae_pp": float(
            regional["national_candidate_mae_pp"].mean()
        ),
        "winner_accuracy": float(np.mean(winner_correct)),
        "rows": len(frame),
    }
    return summary, regional, national


def main() -> None:
    if engine is not rederive.engine or engine is not base_eval.engine:
        raise RuntimeError(
            "shadow evaluator and fold configuration must share one engine module"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full = _prepare_rows()
    base = _base_layer_frame()
    summaries: list[dict[str, object]] = []
    by_election: list[pd.DataFrame] = []
    national: list[pd.DataFrame] = []
    fold_audit: list[dict[str, object]] = []
    preference_configs: list[dict[str, object]] = []
    primary: pd.DataFrame | None = None

    active = pd.read_csv(ACTIVE_PATH, encoding="utf-8-sig")
    active_summary, active_by, active_national = _metrics(
        active, "layer_pred", "active_outcome_aligned_slots"
    )
    summaries.append(active_summary)
    by_election.append(active_by)
    national.append(active_national)

    for variant in VARIANTS:
        outer, audit = _build_outer_predictions(full, variant)
        layered = _attach_layers(base, outer)
        nested, configs = _apply_nested_preference(layered, variant)
        summary, election_rows, national_rows = _metrics(
            nested, "layer_pred", variant
        )
        summaries.append(summary)
        by_election.append(election_rows)
        national.append(national_rows)
        fold_audit.extend(audit)
        preference_configs.extend(configs)
        if variant == "preliminary_hierarchy_min2":
            primary = nested

    summary_frame = pd.DataFrame(summaries)
    active_regional = float(
        summary_frame.loc[
            summary_frame["variant"].eq("active_outcome_aligned_slots"),
            "regional_equal_election_macro_mae_pp",
        ].iloc[0]
    )
    summary_frame["regional_change_vs_active_pp"] = (
        summary_frame["regional_equal_election_macro_mae_pp"] - active_regional
    )
    payload = {
        "scope": {
            "scored_elections": list(ELECTIONS),
            "post_2022_outcomes_used": False,
            "target_excluded_from_each_outer_fit": True,
            "candidate_denominator": "same scored rows in train and target",
        },
        "old_slot_predictors_forbidden": sorted(OLD_SLOT_PREDICTORS),
        "minimum_prior_scored_elections_for_preliminary_columns": (
            MIN_PRIOR_SCORED_FOR_PRELIMINARY_COLUMNS
        ),
        "hierarchy_constraint": {
            "regimes": [
                "two_strong_one_medium",
                "one_strong_two_medium",
                "two_strong_one_weak",
            ],
            "third_prior_log_odds_weight": 0.25,
            "absolute_third_cap": 0.30,
            "third_to_second_cap": 0.95,
            "application_level": "national only; regional shape preserved",
            "region_weights": "prior-election regional vote volume when available",
            "target_outcomes_used": False,
        },
        "frozen_components": {
            "ridge_alpha_and_postprocess": "existing target-fold configs held fixed",
            "electorate_preference_gain": "reselected from earlier shadow folds only",
        },
        "results": summary_frame.to_dict("records"),
        "interpretation": (
            "Experimental comparison only. Preliminary assignments were designed "
            "after through-2022 diagnostics, so this is not an untouched holdout."
        ),
    }
    summary_frame.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(by_election, ignore_index=True).to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(national, ignore_index=True).to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_audit).to_csv(
        OUTPUT_DIR / "fold_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(preference_configs).to_csv(
        OUTPUT_DIR / "preference_gain_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    if primary is not None:
        primary.to_csv(
            OUTPUT_DIR / "preliminary_hierarchy_min2_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary_frame.to_string(index=False))
    print("\n" + pd.concat(by_election, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
