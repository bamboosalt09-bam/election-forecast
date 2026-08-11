"""Evaluate latent electorate layers on frozen nested outer predictions.

The script never loads a post-2022 presidential result. It reports both an
adaptive selector that uses only earlier frozen outer predictions and the same
weak structural overlay in every fold. Regional MAE is weighted by observed
contest votes inside each election, then averaged equally across elections.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENGINE_DIR = ROOT / "presidential_issue_engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from presidential_issue_engine import electorate_layers as layers  # noqa: E402
import issue_vote_engine as engine  # noqa: E402
import rederive_layers_through2022 as rederive  # noqa: E402
import robustness_check as robustness  # noqa: E402


ALLOWED_ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
BASELINE_PATH = (
    ROOT
    / "presidential_issue_engine"
    / "report"
    / "tables"
    / "issue_vote_engine_nested_outer_predictions.csv"
)
HISTORY_PATH = ROOT / "presidential_issue_engine" / "fixed_dataset" / "bloc_history_results.csv"
SALIENCE_PATH = ROOT / "data" / "issue_salience_assembly.csv"
LINK_PATH = ROOT / "data" / "candidate_issue_link.csv"
OVERLAY_PATH = ROOT / "data" / "raw" / "assembly_issue_character_overlay.csv"
STANCE_PATH = ROOT / "data" / "raw" / "candidate_party_tone_gap.csv"
SENSITIVITY_PATH = (
    ROOT
    / "presidential_issue_engine"
    / "fixed_dataset"
    / "region_issue_sensitivity_curated.csv"
)
TURNOUT_HISTORY_PATH = ROOT / "data" / "raw" / "regional_turnout_history.csv"
OUTPUT_DIR = ROOT / "outputs" / "electorate_layer_experiment"

ANCHOR_GAINS = (0.0, 0.10, 0.20, 0.30, 0.45)
PREFERENCE_GAINS = (0.0, 0.01, 0.02, 0.03, 0.04)
TURNOUT_GAINS = (0.0, 0.04, 0.08, 0.15)
FIXED_STRUCTURAL_CONFIG = layers.ElectorateLayerConfig(preference_gain=0.04)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _assert_scope(frame: pd.DataFrame, source_name: str) -> None:
    elections = set(frame.get("election_id", pd.Series(dtype=str)).dropna().astype(str))
    unexpected = sorted(elections - set(ALLOWED_ELECTIONS))
    if unexpected:
        raise RuntimeError(f"{source_name} contains out-of-scope scored elections: {unexpected}")


def _attach_nonvoter_reservoir(frame: pd.DataFrame, turnout: pd.DataFrame) -> pd.DataFrame:
    """Attach a prior-only non-voter reservoir when official history exists.

    No target-election turnout is used. The regional reservoir is the weighted
    mean non-turnout rate from earlier elections. Header-only or missing input
    leaves the feature exactly zero.
    """

    out = frame.copy()
    out["nonvoter_reservoir"] = 0.0
    required = {"election_id", "region_id", "turnout_rate", "available_date"}
    if turnout.empty or not required.issubset(turnout.columns):
        return out
    source = turnout.copy()
    source["turnout_rate"] = pd.to_numeric(source["turnout_rate"], errors="coerce")
    source["available_date"] = pd.to_datetime(source["available_date"], errors="coerce")
    for election_id, index in out.groupby("election_id").groups.items():
        target_date = layers.election_date(str(election_id))
        if target_date is None:
            continue
        eligible = source.loc[
            source["available_date"].notna()
            & (source["available_date"] < pd.Timestamp(target_date))
        ].copy()
        if eligible.empty:
            continue
        eligible["turnout_rate"] = eligible["turnout_rate"].where(
            eligible["turnout_rate"].le(1.0), eligible["turnout_rate"] / 100.0
        )
        eligible["nonvoter"] = (1.0 - eligible["turnout_rate"]).clip(0.0, 1.0)
        regional = eligible.groupby("region_id")["nonvoter"].mean()
        out.loc[index, "nonvoter_reservoir"] = (
            out.loc[index, "region_id"].map(regional).fillna(0.0).to_numpy(float)
        )
    return out


def _layer_config_from_row(row: pd.Series) -> rederive.LayerConfig:
    values: dict[str, object] = {}
    for name in rederive.LayerConfig.__dataclass_fields__:
        default = getattr(rederive.NEUTRAL_CONFIG, name)
        value = row[name]
        if isinstance(default, bool):
            values[name] = str(value).strip().lower() in {"1", "true", "yes", "y"}
        elif isinstance(default, float):
            values[name] = float(value)
        else:
            values[name] = value
    return rederive.LayerConfig(**values)


def build_replacement_nested_predictions() -> pd.DataFrame:
    """Reproduce nested folds with legacy party layers disabled.

    Keeping all other selected fold settings fixed isolates the replacement:
    the new electorate layer substitutes for the aggregate party-tone
    adjustment while retaining the proven regional partisan moderation.
    """

    config_rows = _read_csv(
        ROOT
        / "presidential_issue_engine"
        / "report"
        / "through2022_rederived"
        / "nested_outer_results.csv"
    )
    overlay_paths = rederive.write_overlay_variants()
    with rederive.configured(rederive.NEUTRAL_CONFIG, overlay_paths):
        all_rows = engine.assemble()
    scored = robustness.competition_frame(all_rows)
    warmup = robustness.rolling_warmup_frame(all_rows)
    full_order = [*rederive.WARMUP_ELECTIONS, *ALLOWED_ELECTIONS]
    order_lookup = {election_id: index for index, election_id in enumerate(full_order)}
    full = pd.concat([warmup, scored], ignore_index=True, sort=False)
    for predictor in engine.PREDICTORS:
        full[predictor] = pd.to_numeric(full[predictor], errors="coerce").fillna(0.0)
    full = full.copy()
    full["_order"] = full["election_id"].map(order_lookup)
    warmup_ids = set(rederive.WARMUP_ELECTIONS)
    outputs: list[pd.DataFrame] = []
    for target in ALLOWED_ELECTIONS:
        selected_row = config_rows.loc[config_rows["target_election"].eq(target)]
        if len(selected_row) != 1:
            raise RuntimeError(f"expected one nested config for {target}")
        selected = _layer_config_from_row(selected_row.iloc[0])
        test = engine.scored_contest_rows(full.loc[full["election_id"].eq(target)]).copy()
        train = full.loc[full["_order"] < order_lookup[target]].copy()
        train["_rolling_target"] = engine.normalized_vote_share_target(train)
        train, residual_mask = engine.rolling_training_with_slot_backfill(
            train, test, warmup_ids
        )
        with rederive.configured(selected, overlay_paths):
            x_train = train[engine.PREDICTORS].to_numpy(float)
            x_test = test[engine.PREDICTORS].to_numpy(float)
            y_train = train["_rolling_target"].to_numpy(float)
            beta, _, _, _, means, scales = engine.ridge_fit(
                x_train,
                y_train,
                alpha=selected.ridge_alpha,
                sample_weight=engine.election_epoch_sample_weight(train),
            )
            train_pred = engine.ridge_predict(beta, x_train, means, scales)
            pred = engine.ridge_predict(beta, x_test, means, scales)
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
            legacy = engine.apply_prediction_postprocess(
                test, pred, electorate_layer=False
            )
            replacement = engine.apply_prediction_postprocess(
                test,
                pred,
                party_tone=False,
                electorate_layer=False,
            )
        output = test[["election_id", "region_id", "slot"]].copy()
        output["reproduced_legacy_pred"] = legacy
        output["replacement_base_pred"] = replacement
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def prepare_frame(
    mass_profile: str = "direct_party_layers",
    *,
    require_frozen_reproduction: bool = True,
) -> pd.DataFrame:
    baseline = _read_csv(BASELINE_PATH)
    _assert_scope(baseline, "nested baseline")
    if tuple(dict.fromkeys(baseline["election_id"].astype(str))) != ALLOWED_ELECTIONS:
        raise RuntimeError("nested baseline election order is not the frozen 2002-2022 order")
    replacement = build_replacement_nested_predictions()
    baseline = baseline.merge(replacement, on=["election_id", "region_id", "slot"], how="left")
    reproduction_difference = (
        baseline["pred"] - baseline["reproduced_legacy_pred"]
    ).abs().max()
    if not np.isfinite(reproduction_difference):
        raise RuntimeError("nested fold reproduction difference is not finite")
    if require_frozen_reproduction and reproduction_difference > 1e-10:
        raise RuntimeError(
            f"nested fold reproduction drifted from frozen predictions: {reproduction_difference}"
        )
    baseline = baseline.rename(columns={"pred": "official_pred"})
    baseline["pred"] = baseline["replacement_base_pred"]
    baseline["frozen_reproduction_difference"] = float(reproduction_difference)
    baseline["frozen_reproduction_guard_required"] = bool(require_frozen_reproduction)
    history = _read_csv(HISTORY_PATH)
    salience = _read_csv(SALIENCE_PATH)
    link = _read_csv(LINK_PATH)
    overlay = _read_csv(OVERLAY_PATH)
    if os.getenv(engine.STRICT_UNDATED_CURATED_INPUTS_ENV, "0") == "1":
        sensitivity = pd.DataFrame(columns=["issue_name", "region_id", "sensitivity"])
    else:
        sensitivity = _read_csv(SENSITIVITY_PATH)
    # Outcome sources are scope-locked; predictor histories may contain later
    # non-presidential elections, but every estimator filters them before each target.
    _assert_scope(overlay, "speech character overlay")
    candidate_keys = baseline[
        ["election_id", "region_id", "slot", "candidate_name", "bloc"]
    ].copy()
    landscape = engine._candidate_political_landscape_features(candidate_keys)
    candidate_keys = candidate_keys.merge(
        landscape,
        on=["election_id", "region_id", "slot"],
        how="left",
        validate="one_to_one",
    )
    layer_frame = layers.estimate_electorate_layers(
        candidate_keys,
        history,
        mass_profile=mass_profile,
    )
    issue_signals = layers.compile_issue_class_signals(
        candidate_keys,
        salience,
        link,
        overlay,
        regional_sensitivity=sensitivity,
        candidate_stance=_read_csv(STANCE_PATH),
    )
    out = baseline.merge(
        layer_frame.drop(columns=["bloc"], errors="ignore"),
        on=["election_id", "region_id", "slot"],
        how="left",
    ).merge(issue_signals, on=["election_id", "region_id", "slot"], how="left")
    out = _attach_nonvoter_reservoir(out, _read_csv(TURNOUT_HISTORY_PATH))
    for column in out.columns:
        if column.startswith("issue_pref_") or column.startswith("issue_attention_"):
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def election_weighted_mae(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for election_id in ALLOWED_ELECTIONS:
        group = frame.loc[frame["election_id"].eq(election_id)].copy()
        error = (group[prediction_column] - group["actual"]).abs() * 100.0
        weights = pd.to_numeric(group["contest_votes"], errors="coerce").fillna(0.0).clip(lower=0.0)
        mae = float(np.average(error, weights=weights)) if float(weights.sum()) > 0 else float(error.mean())
        rows.append({"election_id": election_id, "weighted_row_mae_pp": mae, "rows": len(group)})
    return pd.DataFrame(rows)


def macro_weighted_mae(frame: pd.DataFrame, prediction_column: str, elections: tuple[str, ...]) -> float:
    by = election_weighted_mae(frame.loc[frame["election_id"].isin(elections)], prediction_column)
    by = by.loc[by["election_id"].isin(elections)]
    return float(by["weighted_row_mae_pp"].mean()) if not by.empty else float("nan")


def apply_config(frame: pd.DataFrame, config: layers.ElectorateLayerConfig) -> pd.DataFrame:
    pred, diagnostics = layers.apply_electorate_layer_response(frame, frame["pred"], config)
    out = frame.copy()
    out["layer_pred"] = pred
    out = out.merge(
        diagnostics,
        on=["election_id", "region_id", "slot"],
        how="left",
        suffixes=("", "_diagnostic"),
    )
    return out


def improvement_threshold(n_elections: int) -> float:
    if n_elections <= 2:
        return 0.10
    if n_elections == 3:
        return 0.075
    return 0.05


def select_config(
    frame: pd.DataFrame,
    tuning_elections: tuple[str, ...],
    label: str,
) -> tuple[layers.ElectorateLayerConfig, list[dict[str, object]]]:
    if len(tuning_elections) < 2:
        return layers.NEUTRAL_LAYER_CONFIG, []
    current = layers.NEUTRAL_LAYER_CONFIG
    trace: list[dict[str, object]] = []
    threshold = improvement_threshold(len(tuning_elections))
    turnout_values = TURNOUT_GAINS
    if "nonvoter_reservoir" not in frame.columns or float(
        pd.to_numeric(frame["nonvoter_reservoir"], errors="coerce").fillna(0.0).max()
    ) <= 0.0:
        # Vote-share outcomes alone cannot identify preference change versus
        # differential turnout. Keep the entire turnout channel off until
        # official prior regional turnout history is supplied.
        turnout_values = (0.0,)
    steps = (
        ("terrain_anchor_gain", ANCHOR_GAINS),
        ("preference_gain", PREFERENCE_GAINS),
        ("turnout_gain", turnout_values),
    )
    for field_name, values in steps:
        current_frame = apply_config(frame, current)
        current_mae = macro_weighted_mae(current_frame, "layer_pred", tuning_elections)
        options: list[tuple[float, tuple[int, float], layers.ElectorateLayerConfig]] = []
        for value in values:
            option = replace(current, **{field_name: value})
            evaluated = apply_config(frame, option)
            mae = macro_weighted_mae(evaluated, "layer_pred", tuning_elections)
            options.append((mae, option.complexity, option))
            trace.append(
                {
                    "selection_label": label,
                    "tuning_elections": "|".join(tuning_elections),
                    "step": field_name,
                    **asdict(option),
                    "weighted_macro_mae_pp": mae,
                    "current_before_step_mae_pp": current_mae,
                    "required_improvement_pp": threshold,
                }
            )
        best_mae, _, best = min(options, key=lambda item: (item[0], item[1]))
        if np.isfinite(best_mae) and current_mae - best_mae >= threshold:
            current = best
    return current, trace


def _third_shape_correlation(frame: pd.DataFrame, prediction_column: str) -> float:
    values: list[float] = []
    for election_id in ("pres_2007", "pres_2017"):
        group = frame.loc[frame["election_id"].eq(election_id) & frame["slot"].eq("C")]
        if len(group) >= 3 and group[prediction_column].std() > 0 and group["actual"].std() > 0:
            values.append(float(group[prediction_column].corr(group["actual"])))
    return float(np.mean(values)) if values else float("nan")


def _region_weighted_mae(frame: pd.DataFrame, prediction_column: str, region_id: str) -> float:
    group = frame.loc[frame["region_id"].eq(region_id)].copy()
    if group.empty:
        return float("nan")
    error = (group[prediction_column] - group["actual"]).abs() * 100.0
    weights = group["contest_votes"].clip(lower=0.0)
    return float(np.average(error, weights=weights))


def history_source_audit() -> pd.DataFrame:
    history = _read_csv(HISTORY_PATH)
    parts: list[pd.DataFrame] = []
    for target in ALLOWED_ELECTIONS:
        target_date = layers.election_date(target)
        eligible = layers._history_before_target(history, target)
        if eligible.empty:
            continue
        grouped = eligible.groupby(
            ["election_id", "election_type", "source_date"], as_index=False
        ).agg(
            rows=("vote_share", "size"),
            weight_sum=("weight", "sum"),
            regions=("region_id", "nunique"),
        )
        grouped.insert(0, "target_election", target)
        grouped["target_date"] = pd.Timestamp(target_date)
        grouped["days_before_target"] = (
            grouped["target_date"] - grouped["source_date"]
        ).dt.days
        grouped["direct_party_ballot"] = grouped["election_type"].isin(
            layers.DIRECT_PARTY_ELECTION_TYPES
        )
        grouped["pit_safe"] = grouped["source_date"] < grouped["target_date"]
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame()
    baseline_by = election_weighted_mae(frame, "official_pred")
    baseline_macro = float(baseline_by["weighted_row_mae_pp"].mean())
    replacement_by = election_weighted_mae(frame, "pred")
    replacement_macro = float(replacement_by["weighted_row_mae_pp"].mean())

    nested_parts: list[pd.DataFrame] = []
    nested_configs: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    for index, target in enumerate(ALLOWED_ELECTIONS):
        prior = ALLOWED_ELECTIONS[:index]
        selected, selected_trace = select_config(frame, prior, f"outer_{target}")
        trace.extend(selected_trace)
        target_frame = frame.loc[frame["election_id"].eq(target)].copy()
        adjusted = apply_config(target_frame, selected)
        nested_parts.append(adjusted)
        target_mae = float(election_weighted_mae(adjusted, "layer_pred").loc[
            lambda value: value["election_id"].eq(target), "weighted_row_mae_pp"
        ].iloc[0])
        nested_configs.append(
            {
                "target_election": target,
                "tuning_elections": "|".join(prior),
                **asdict(selected),
                "outer_weighted_row_mae_pp": target_mae,
            }
        )
    nested = pd.concat(nested_parts, ignore_index=True)
    nested_by = election_weighted_mae(nested, "layer_pred")
    nested_macro = float(nested_by["weighted_row_mae_pp"].mean())

    final_config, final_trace = select_config(frame, ALLOWED_ELECTIONS, "final_deployment")
    trace.extend(final_trace)
    final = apply_config(frame, final_config)
    final_by = election_weighted_mae(final, "layer_pred")
    final_macro = float(final_by["weighted_row_mae_pp"].mean())
    fixed = apply_config(frame, FIXED_STRUCTURAL_CONFIG)
    fixed_by = election_weighted_mae(fixed, "layer_pred")
    fixed_macro = float(fixed_by["weighted_row_mae_pp"].mean())
    source_audit = history_source_audit()
    if not source_audit.empty and not bool(source_audit["pit_safe"].all()):
        raise RuntimeError("electorate layer history contains a non-PIT-safe source")

    ablations = []
    for label, config in (
        ("neutral", layers.NEUTRAL_LAYER_CONFIG),
        ("terrain_only", replace(final_config, preference_gain=0.0, turnout_gain=0.0)),
        ("preference_without_turnout", replace(final_config, turnout_gain=0.0)),
        ("full", final_config),
    ):
        evaluated = apply_config(frame, config)
        ablations.append(
            {
                "ablation": label,
                **asdict(config),
                "selection_sample_weighted_macro_mae_pp": macro_weighted_mae(
                    evaluated, "layer_pred", ALLOWED_ELECTIONS
                ),
            }
        )

    comparison = baseline_by.rename(
        columns={"weighted_row_mae_pp": "baseline_weighted_row_mae_pp"}
    ).merge(
        nested_by[["election_id", "weighted_row_mae_pp"]].rename(
            columns={"weighted_row_mae_pp": "layer_weighted_row_mae_pp"}
        ),
        on="election_id",
    ).merge(
        fixed_by[["election_id", "weighted_row_mae_pp"]].rename(
            columns={"weighted_row_mae_pp": "fixed_layer_weighted_row_mae_pp"}
        ),
        on="election_id",
    )
    comparison["improvement_pp"] = (
        comparison["baseline_weighted_row_mae_pp"] - comparison["layer_weighted_row_mae_pp"]
    )
    comparison["fixed_layer_improvement_pp"] = (
        comparison["baseline_weighted_row_mae_pp"]
        - comparison["fixed_layer_weighted_row_mae_pp"]
    )
    max_worsening = float((-comparison["improvement_pp"]).clip(lower=0.0).max())
    fixed_max_worsening = float(
        (-comparison["fixed_layer_improvement_pp"]).clip(lower=0.0).max()
    )
    baseline_c_corr = _third_shape_correlation(nested, "official_pred")
    layer_c_corr = _third_shape_correlation(nested, "layer_pred")
    fixed_c_corr = _third_shape_correlation(fixed, "layer_pred")
    baseline_gyeongbuk = _region_weighted_mae(nested, "official_pred", "sido_47")
    layer_gyeongbuk = _region_weighted_mae(nested, "layer_pred", "sido_47")
    fixed_gyeongbuk = _region_weighted_mae(fixed, "layer_pred", "sido_47")
    improvement = baseline_macro - nested_macro
    fixed_improvement = baseline_macro - fixed_macro
    election_2022 = comparison.loc[comparison["election_id"].eq("pres_2022")].iloc[0]
    gates = {
        "fixed_structural_macro_improves": fixed_improvement > 0.0,
        "fixed_structural_2022_worsening_at_most_0_05pp": (
            float(-election_2022["fixed_layer_improvement_pp"]) <= 0.05
        ),
        "third_candidate_shape_not_worse_by_0_03": (
            not np.isfinite(baseline_c_corr)
            or not np.isfinite(fixed_c_corr)
            or fixed_c_corr >= baseline_c_corr - 0.03
        ),
        "gyeongbuk_worsening_at_most_0_50pp": (
            not np.isfinite(baseline_gyeongbuk)
            or not np.isfinite(fixed_gyeongbuk)
            or fixed_gyeongbuk <= baseline_gyeongbuk + 0.50
        ),
    }
    turnout_rows = _read_csv(TURNOUT_HISTORY_PATH)
    payload = {
        "scope": {
            "scored_elections": list(ALLOWED_ELECTIONS),
            "post_2022_presidential_outcomes_loaded": False,
            "metric": "contest-vote weighted row MAE within election, equal-election macro",
        },
        "baseline": {
            "nested_weighted_macro_mae_pp": baseline_macro,
            "by_election": baseline_by.set_index("election_id")["weighted_row_mae_pp"].to_dict(),
        },
        "replacement_neutral": {
            "nested_weighted_macro_mae_pp": replacement_macro,
            "by_election": replacement_by.set_index("election_id")["weighted_row_mae_pp"].to_dict(),
            "disabled_legacy_layers": ["party_tone"],
        },
        "strict_nested_layer": {
            "weighted_macro_mae_pp": nested_macro,
            "improvement_pp": improvement,
            "by_election": nested_by.set_index("election_id")["weighted_row_mae_pp"].to_dict(),
            "configs": nested_configs,
        },
        "fixed_structural_layer": {
            "config": asdict(FIXED_STRUCTURAL_CONFIG),
            "weighted_macro_mae_pp": fixed_macro,
            "improvement_pp": fixed_improvement,
            "by_election": fixed_by.set_index("election_id")["weighted_row_mae_pp"].to_dict(),
            "interpretation": (
                "same weak structural response in every fold; not an adaptive per-fold selection"
            ),
        },
        "selection_sample_deployment": {
            "config": asdict(final_config),
            "weighted_macro_mae_pp": final_macro,
            "by_election": final_by.set_index("election_id")["weighted_row_mae_pp"].to_dict(),
        },
        "turnout_layer": {
            "history_rows": int(len(turnout_rows)),
            "active": bool(len(turnout_rows) and final_config.nonvoter_gain > 0.0),
            "note": "inactive when official prior turnout history is absent",
        },
        "history_source_audit": {
            "rows": int(len(source_audit)),
            "all_sources_strictly_before_target": bool(
                source_audit.empty or source_audit["pit_safe"].all()
            ),
            "pres_2002_source_elections": sorted(
                source_audit.loc[
                    source_audit["target_election"].eq("pres_2002"), "election_id"
                ].astype(str).unique().tolist()
            ),
        },
        "diagnostics": {
            "baseline_third_shape_correlation": baseline_c_corr,
            "adaptive_layer_third_shape_correlation": layer_c_corr,
            "fixed_layer_third_shape_correlation": fixed_c_corr,
            "baseline_gyeongbuk_weighted_mae_pp": baseline_gyeongbuk,
            "adaptive_layer_gyeongbuk_weighted_mae_pp": layer_gyeongbuk,
            "fixed_layer_gyeongbuk_weighted_mae_pp": fixed_gyeongbuk,
            "adaptive_maximum_election_worsening_pp": max_worsening,
            "fixed_maximum_election_worsening_pp": fixed_max_worsening,
            "adaptive_selector_retained_layer": any(
                row["preference_gain"] > 0.0 for row in nested_configs
            ),
        },
        "fixed_structural_experiment_safety_gates": gates,
        "fixed_structural_experiment_passes_safety_gates": bool(all(gates.values())),
        "adopt_into_active_engine": bool(
            any(row["preference_gain"] > 0.0 for row in nested_configs)
        ),
    }

    nested.to_csv(OUTPUT_DIR / "nested_shadow_predictions.csv", index=False, encoding="utf-8-sig")
    fixed.to_csv(
        OUTPUT_DIR / "fixed_structural_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fixed_2022 = fixed.loc[fixed["election_id"].eq("pres_2022")].copy()
    fixed_2022["baseline_abs_error_pp"] = (
        fixed_2022["official_pred"] - fixed_2022["actual"]
    ).abs() * 100.0
    fixed_2022["fixed_layer_abs_error_pp"] = (
        fixed_2022["layer_pred"] - fixed_2022["actual"]
    ).abs() * 100.0
    fixed_2022["prediction_shift_pp"] = (
        fixed_2022["layer_pred"] - fixed_2022["official_pred"]
    ) * 100.0
    fixed_2022["error_change_pp"] = (
        fixed_2022["fixed_layer_abs_error_pp"] - fixed_2022["baseline_abs_error_pp"]
    )
    fixed_2022.to_csv(
        OUTPUT_DIR / "pres_2022_region_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    source_audit.to_csv(
        OUTPUT_DIR / "history_source_audit.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(OUTPUT_DIR / "nested_weighted_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(nested_configs).to_csv(
        OUTPUT_DIR / "nested_selected_configs.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(trace).to_csv(OUTPUT_DIR / "selection_trace.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ablations).to_csv(OUTPUT_DIR / "ablations.csv", index=False, encoding="utf-8-sig")
    layer_columns = [
        "election_id",
        "region_id",
        "slot",
        "bloc",
        "durable_core_raw",
        "recent_bloc_base",
        "critical_support_raw",
        "core_voting_mass",
        "critical_voting_mass",
        "swing_voting_mass",
        "bloc_vote_volatility",
        "layer_effective_elections",
        "direct_party_core_raw",
        "candidate_ballot_core_raw",
        "direct_party_recent_base",
        "candidate_ballot_recent_base",
        "direct_party_effective_elections",
        "candidate_ballot_effective_elections",
        "direct_party_reliability",
        "nonvoter_reservoir",
    ]
    frame[[column for column in layer_columns if column in frame.columns]].to_csv(
        OUTPUT_DIR / "layer_mass_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"baseline nested weighted macro MAE: {baseline_macro:.6f}%p")
    print(f"layer strict nested weighted macro MAE: {nested_macro:.6f}%p")
    print(f"strict improvement: {improvement:+.6f}%p")
    print(f"selection-sample config: {asdict(final_config)}")
    print(f"selection-sample weighted macro MAE: {final_macro:.6f}%p")
    print(f"fixed structural weighted macro MAE: {fixed_macro:.6f}%p")
    print(f"fixed structural improvement: {fixed_improvement:+.6f}%p")
    print(f"fixed structural experiment safety gates: {gates}")
    print(f"adopt into active engine: {payload['adopt_into_active_engine']}")


if __name__ == "__main__":
    main()
