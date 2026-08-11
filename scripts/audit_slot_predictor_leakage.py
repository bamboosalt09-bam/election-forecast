"""Audit outcome leakage risk from winner-aligned A/B slot predictors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import electorate_layers as layers  # noqa: E402
import evaluate_electorate_layers as base_eval  # noqa: E402
import issue_vote_engine as engine  # noqa: E402
import rederive_layers_through2022 as rederive  # noqa: E402
import robustness_check as robustness  # noqa: E402


ELECTIONS = base_eval.ALLOWED_ELECTIONS
OUTPUT_DIR = ROOT / "outputs" / "slot_predictor_audit"
ACTIVE_MODEL_DIR = ROOT / "outputs" / "active_presidential_nested_v9"
PREDICTOR_SETS = {
    "active_all_predictors": tuple(engine.PREDICTORS),
    "without_direct_slot_dummies": tuple(
        value for value in engine.PREDICTORS if value not in {"slot_A", "slot_B"}
    ),
    "without_all_slot_derived_predictors": tuple(
        value
        for value in engine.PREDICTORS
        if value not in {"slot_A", "slot_B", "slotA_prior", "slotB_prior"}
    ),
}


def build_outer_predictions(predictors: tuple[str, ...]) -> pd.DataFrame:
    config_rows = base_eval._read_csv(
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
    order = [*rederive.WARMUP_ELECTIONS, *ELECTIONS]
    lookup = {election_id: index for index, election_id in enumerate(order)}
    full = pd.concat([warmup, scored], ignore_index=True, sort=False)
    for predictor in engine.PREDICTORS:
        full[predictor] = pd.to_numeric(full[predictor], errors="coerce").fillna(0.0)
    full = full.copy()
    full["_order"] = full["election_id"].map(lookup)
    warmup_ids = set(rederive.WARMUP_ELECTIONS)
    outputs: list[pd.DataFrame] = []
    for target in ELECTIONS:
        selected_row = config_rows.loc[config_rows["target_election"].eq(target)]
        selected = base_eval._layer_config_from_row(selected_row.iloc[0])
        test = engine.scored_contest_rows(full.loc[full["election_id"].eq(target)]).copy()
        train = full.loc[full["_order"] < lookup[target]].copy()
        train["_rolling_target"] = engine.normalized_vote_share_target(train)
        train, residual_mask = engine.rolling_training_with_slot_backfill(
            train, test, warmup_ids
        )
        with rederive.configured(selected, overlay_paths):
            x_train = train[list(predictors)].to_numpy(float)
            x_test = test[list(predictors)].to_numpy(float)
            beta, _, _, _, means, scales = engine.ridge_fit(
                x_train,
                train["_rolling_target"].to_numpy(float),
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
                train.loc[residual_mask], test, train_pred[residual_mask], pred
            )
            pred = engine.normalize_vote_share_predictions(test, pred)
            pred = engine.apply_prediction_postprocess(
                test, pred, party_tone=False, electorate_layer=False
            )
        output = test[["election_id", "region_id", "slot"]].copy()
        output["pred"] = pred
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def apply_active_preference_schedule(frame: pd.DataFrame) -> pd.DataFrame:
    gains = base_eval._read_csv(
        ROOT / "outputs" / "electorate_nested_learning" / "outer_selected_gains.csv"
    ).set_index("target_election")["preference_gain"]
    parts: list[pd.DataFrame] = []
    for target in ELECTIONS:
        target_frame = frame.loc[frame["election_id"].eq(target)].copy()
        pred, _ = layers.apply_electorate_layer_response(
            target_frame,
            target_frame["pred"],
            layers.ElectorateLayerConfig(
                preference_gain=float(gains.loc[target]),
                mass_profile="direct_party_layers",
            ),
        )
        target_frame["layer_pred"] = pred
        parts.append(target_frame)
    return pd.concat(parts, ignore_index=True)


def metric_rows(label: str, frame: pd.DataFrame) -> tuple[list[dict[str, object]], float, float]:
    regional = base_eval.election_weighted_mae(frame, "layer_pred").set_index(
        "election_id"
    )["weighted_row_mae_pp"]
    national_rows: list[dict[str, object]] = []
    for (election_id, slot), group in frame.groupby(["election_id", "slot"], sort=True):
        weights = group["contest_votes"].to_numpy(float)
        pred = float(np.average(group["layer_pred"], weights=weights))
        actual = float(np.average(group["actual"], weights=weights))
        national_rows.append(
            {
                "predictor_set": label,
                "election_id": election_id,
                "slot": slot,
                "pred_pct": pred * 100.0,
                "actual_pct": actual * 100.0,
                "abs_error_pp": abs(pred - actual) * 100.0,
                "regional_weighted_mae_pp": float(regional.loc[election_id]),
            }
        )
    national = pd.DataFrame(national_rows)
    national_macro = float(
        national.groupby("election_id")["abs_error_pp"].mean().mean()
    )
    return national_rows, float(regional.mean()), national_macro


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluation = base_eval.prepare_frame(
        mass_profile="direct_party_layers",
        require_frozen_reproduction=False,
    )
    all_national: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    active_reference: pd.DataFrame | None = None
    for label, predictors in PREDICTOR_SETS.items():
        outer = build_outer_predictions(predictors)
        candidate = evaluation.drop(columns=["pred"]).merge(
            outer,
            on=["election_id", "region_id", "slot"],
            how="left",
            validate="one_to_one",
        )
        evaluated = apply_active_preference_schedule(candidate)
        evaluated["predictor_set"] = label
        predictions.append(evaluated)
        national_rows, regional_macro, national_macro = metric_rows(label, evaluated)
        all_national.extend(national_rows)
        row_2012 = pd.DataFrame(national_rows)
        row_2012 = row_2012.loc[row_2012["election_id"].eq("pres_2012")]
        summaries.append(
            {
                "predictor_set": label,
                "predictors": "|".join(predictors),
                "predictor_count": len(predictors),
                "nested_regional_weighted_macro_mae_pp": regional_macro,
                "nested_national_equal_election_macro_mae_pp": national_macro,
                "pres_2012_regional_weighted_mae_pp": float(
                    row_2012["regional_weighted_mae_pp"].iloc[0]
                ),
                "pres_2012_national_mae_pp": float(row_2012["abs_error_pp"].mean()),
            }
        )
        if label == "active_all_predictors":
            active_reference = evaluated

    assert active_reference is not None
    frozen = base_eval._read_csv(
        ROOT / "outputs" / "electorate_nested_learning" / "nested_predictions.csv"
    )
    joined = active_reference.merge(
        frozen[["election_id", "region_id", "slot", "layer_pred"]],
        on=["election_id", "region_id", "slot"],
        suffixes=("_audit", "_frozen"),
        validate="one_to_one",
    )
    reproduction = float(
        (joined["layer_pred_audit"] - joined["layer_pred_frozen"]).abs().max()
    )
    active_fold_audit = pd.read_csv(
        ACTIVE_MODEL_DIR / "fold_audit.csv", encoding="utf-8-sig"
    )
    if active_fold_audit["old_slot_predictors_used"].astype(bool).any():
        raise RuntimeError("realized slot predictors are active in v7")
    if not active_fold_audit["target_excluded_from_fit"].astype(bool).all():
        raise RuntimeError("v7 fold audit includes its target election")

    active_actual = pd.DataFrame(all_national)
    active_actual = active_actual.loc[
        active_actual["predictor_set"].eq("active_all_predictors")
    ]
    winners = (
        active_actual.sort_values(
            ["election_id", "actual_pct"], ascending=[True, False]
        )
        .groupby("election_id", as_index=False)
        .first()[["election_id", "slot", "actual_pct"]]
    )
    winner_a_rate = float(winners["slot"].eq("A").mean())
    payload = {
        "scope": {
            "scored_elections": list(ELECTIONS),
            "post_2022_outcomes_used": False,
            "purpose": "post-hoc leakage audit; not an untouched model selection run",
        },
        "slot_a_winner_rate": winner_a_rate,
        "slot_a_winner_count": int(winners["slot"].eq("A").sum()),
        "scored_election_count": len(winners),
        "legacy_frozen_reproduction_max_abs": reproduction,
        "legacy_frozen_reproduction_required": False,
        "active_model_old_slot_predictors_used": False,
        "active_model_target_excluded_from_fit": True,
        "conclusion": (
            "A is the realized winner in every scored election, so direct A/B slot "
            "predictors are outcome-aligned and are not defensible forecast inputs."
        ),
    }
    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_national).to_csv(
        OUTPUT_DIR / "national_by_election.csv", index=False, encoding="utf-8-sig"
    )
    winners.to_csv(OUTPUT_DIR / "winner_slot_audit.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\n" + winners.to_string(index=False))
    print("\n" + json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
