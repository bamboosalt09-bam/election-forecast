"""Evaluate the active fixed neutral-context setting across validation protocols.

The parameters were selected after reviewing historical outcomes, so LOEO and
rolling gains are diagnostic and must not be described as clean out-of-sample
model-selection estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "presidential_issue_engine"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ENGINE_DIR))

import issue_vote_engine as engine  # noqa: E402
import robustness_check as robustness  # noqa: E402
from scripts.evaluate_stance_pilot_3000_sensitivity import CONFIGS, build_features  # noqa: E402


ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
CONFIG_NAME = "person_party_speaker_confirmed_conf3_context050_issueglobal025_gate2"
SAMPLE_SIZE = 5_000
SHADOW_SCALE = 0.60
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "neutral_context_protocols_5000"


def _post_normalization_adjustments(frame: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    pred = engine.apply_partisan_layer_prediction_moderation(frame, pred)
    pred = engine.apply_party_context_prediction_adjustment(frame, pred)
    pred = engine.apply_party_tone_gap_prediction_adjustment(frame, pred)
    pred = engine.apply_same_orientation_external_adjustment(frame, pred)
    pred = engine.apply_public_treatment_prediction_adjustment(frame, pred)
    pred = engine.apply_generation_prediction_adjustment(frame, pred)
    pred = engine.apply_candidate_conversion_context_adjustment(frame, pred)
    return engine.apply_candidate_regionalism_adjustment(frame, pred)


def _final_adjustments(frame: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    pred = engine.apply_third_candidate_prediction_adjustment(frame, pred)
    pred = engine.apply_withdrawn_candidate_prediction_adjustment(frame, pred)
    pred = engine.normalize_vote_share_predictions(frame, pred)
    return _post_normalization_adjustments(frame, pred)


def _shadow_signals() -> pd.DataFrame:
    config = next(config for config in CONFIGS if config["name"] == CONFIG_NAME)
    pieces: list[pd.DataFrame] = []
    for election_id in ELECTIONS:
        pilot = (
            ROOT
            / "outputs"
            / "assembly_stance"
            / f"pilot_{election_id}_{SAMPLE_SIZE}"
            / "review_batch.csv"
        )
        features = build_features(config, pilot_input=pilot)
        pieces.append(
            features.loc[
                features["election_id"].eq(election_id),
                ["election_id", "slot", "stance_shadow_signal"],
            ].copy()
        )
    signals = pd.concat(pieces, ignore_index=True)
    if signals.duplicated(["election_id", "slot"]).any():
        raise RuntimeError("duplicate election-slot shadow signals")
    return signals


def _apply_shadow(frame: pd.DataFrame, pred: np.ndarray, signals: pd.DataFrame) -> np.ndarray:
    keys = frame[["election_id", "slot"]].copy()
    keys["_row_order"] = np.arange(len(keys))
    keys = keys.merge(signals, on=["election_id", "slot"], how="left").sort_values("_row_order")
    signal = pd.to_numeric(keys["stance_shadow_signal"], errors="coerce").fillna(0.0).to_numpy(float)
    return engine.normalize_vote_share_predictions(frame, np.asarray(pred, dtype=float) + SHADOW_SCALE * signal)


def _result_rows(
    frame: pd.DataFrame,
    baseline_pred: np.ndarray,
    shadow_pred: np.ndarray,
    protocol: str,
) -> pd.DataFrame:
    out = frame[["election_id", "region_id", "slot", "candidate_name", "votes"]].copy()
    out["contest_votes"] = out.groupby(["election_id", "region_id"])["votes"].transform("sum")
    out["actual"] = engine.normalized_vote_share_target(frame)
    out["baseline_pred"] = np.asarray(baseline_pred, dtype=float)
    out["shadow_pred"] = np.asarray(shadow_pred, dtype=float)
    out["baseline_abs_err_pp"] = np.abs(out["baseline_pred"] - out["actual"]) * 100.0
    out["shadow_abs_err_pp"] = np.abs(out["shadow_pred"] - out["actual"]) * 100.0
    out.insert(0, "protocol", protocol)
    return out


def full_fit_rows(frame: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    x = frame[engine.PREDICTORS].to_numpy(float)
    y = engine.normalized_vote_share_target(frame)
    beta, _, _, _, means, scales = engine.ridge_fit(
        x,
        y,
        alpha=engine.RIDGE_ALPHA,
        sample_weight=engine.election_epoch_sample_weight(frame),
    )
    baseline = _final_adjustments(frame, engine.ridge_predict(beta, x, means, scales))
    shadow = _apply_shadow(frame, baseline, signals)
    return _result_rows(frame, baseline, shadow, "full_fit_in_sample")


def loeo_rows(frame: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for election_id in ELECTIONS:
        train = frame.loc[~frame["election_id"].eq(election_id)].copy()
        test = frame.loc[frame["election_id"].eq(election_id)].copy()
        x_train = train[engine.PREDICTORS].to_numpy(float)
        y_train = engine.normalized_vote_share_target(train)
        x_test = test[engine.PREDICTORS].to_numpy(float)
        beta, _, _, _, means, scales = engine.ridge_fit(
            x_train,
            y_train,
            alpha=engine.RIDGE_ALPHA,
            sample_weight=engine.election_epoch_sample_weight(train),
        )
        train_pred = engine.ridge_predict(beta, x_train, means, scales)
        pred = engine.ridge_predict(beta, x_test, means, scales)
        train_pred = engine.apply_third_candidate_prediction_adjustment(train, train_pred)
        train_pred = engine.apply_withdrawn_candidate_prediction_adjustment(train, train_pred)
        pred = engine.apply_third_candidate_prediction_adjustment(test, pred)
        pred = engine.apply_withdrawn_candidate_prediction_adjustment(test, pred)
        pred = engine.apply_region_residual_calibration(train, test, train_pred, pred)
        baseline = _post_normalization_adjustments(
            test,
            engine.normalize_vote_share_predictions(test, pred),
        )
        shadow = _apply_shadow(test, baseline, signals)
        pieces.append(_result_rows(test, baseline, shadow, "loeo"))
    return pd.concat(pieces, ignore_index=True)


def rolling_rows(frame: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    old_scale = engine.THROUGH_2022_REDERIVED_LAYER_CONFIG.get(
        "neutral_context_scale", 0.0
    )
    engine.THROUGH_2022_REDERIVED_LAYER_CONFIG["neutral_context_scale"] = 0.0
    try:
        warmup = robustness.rolling_warmup_frame(frame)
        baseline_rows = robustness.rolling_origin_error_frame(
            frame,
            engine.PREDICTORS,
            warmup=warmup,
            warmup_order=robustness.ROLLING_WARMUP_ELECTIONS,
        )
    finally:
        engine.THROUGH_2022_REDERIVED_LAYER_CONFIG["neutral_context_scale"] = old_scale
    if len(baseline_rows) != 215:
        raise RuntimeError(f"expected 215 rolling baseline rows, found {len(baseline_rows)}")
    metadata = frame[
        ["election_id", "region_id", "slot", "candidate_name", "votes"]
    ].copy()
    metadata["contest_votes"] = metadata.groupby(["election_id", "region_id"])["votes"].transform("sum")
    baseline_rows = baseline_rows.merge(
        metadata,
        on=["election_id", "region_id", "slot", "candidate_name"],
        how="left",
    )
    key_frame = baseline_rows[["election_id", "region_id", "slot"]].merge(
        frame.drop_duplicates(["election_id", "region_id", "slot"]),
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    shadow = _apply_shadow(key_frame, baseline_rows["pred"].to_numpy(float), signals)
    out = baseline_rows[
        ["election_id", "region_id", "slot", "candidate_name", "votes", "contest_votes", "actual"]
    ].copy()
    out["baseline_pred"] = baseline_rows["pred"].to_numpy(float)
    out["shadow_pred"] = shadow
    out["baseline_abs_err_pp"] = np.abs(out["baseline_pred"] - out["actual"]) * 100.0
    out["shadow_abs_err_pp"] = np.abs(out["shadow_pred"] - out["actual"]) * 100.0
    out.insert(0, "protocol", "rolling_origin")
    return out


def row_summary(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        rows.groupby(["protocol", "election_id"], as_index=False)
        .agg(
            n_rows=("actual", "size"),
            baseline_row_mae_pp=("baseline_abs_err_pp", "mean"),
            shadow_row_mae_pp=("shadow_abs_err_pp", "mean"),
        )
    )
    overall = (
        rows.groupby("protocol", as_index=False)
        .agg(
            n_rows=("actual", "size"),
            baseline_row_mae_pp=("baseline_abs_err_pp", "mean"),
            shadow_row_mae_pp=("shadow_abs_err_pp", "mean"),
        )
    )
    overall.insert(1, "election_id", "Overall")
    out = pd.concat([grouped, overall], ignore_index=True)
    out["row_mae_change_pp"] = out["shadow_row_mae_pp"] - out["baseline_row_mae_pp"]
    return out.sort_values(["protocol", "election_id"])


def national_points(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(["protocol", "election_id", "slot", "candidate_name"], sort=True):
        weights = pd.to_numeric(group["contest_votes"], errors="coerce").fillna(0.0).to_numpy(float)
        protocol, election_id, slot, candidate_name = keys
        records.append(
            {
                "protocol": protocol,
                "election_id": election_id,
                "slot": slot,
                "candidate_name": candidate_name,
                "baseline_pred_pct": float(np.average(group["baseline_pred"], weights=weights) * 100.0),
                "shadow_pred_pct": float(np.average(group["shadow_pred"], weights=weights) * 100.0),
                "actual_pct": float(np.average(group["actual"], weights=weights) * 100.0),
                "weight_note": "actual_contest_votes_post_election_diagnostic",
            }
        )
    out = pd.DataFrame(records)
    out["baseline_abs_err_pp"] = np.abs(out["baseline_pred_pct"] - out["actual_pct"])
    out["shadow_abs_err_pp"] = np.abs(out["shadow_pred_pct"] - out["actual_pct"])
    return out


def national_summary(points: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        points.groupby(["protocol", "election_id"], as_index=False)
        .agg(
            n_candidates=("actual_pct", "size"),
            baseline_national_point_mae_pp=("baseline_abs_err_pp", "mean"),
            shadow_national_point_mae_pp=("shadow_abs_err_pp", "mean"),
        )
    )
    overall = (
        points.groupby("protocol", as_index=False)
        .agg(
            n_candidates=("actual_pct", "size"),
            baseline_national_point_mae_pp=("baseline_abs_err_pp", "mean"),
            shadow_national_point_mae_pp=("shadow_abs_err_pp", "mean"),
        )
    )
    overall.insert(1, "election_id", "Overall")
    out = pd.concat([grouped, overall], ignore_index=True)
    out["national_point_mae_change_pp"] = (
        out["shadow_national_point_mae_pp"] - out["baseline_national_point_mae_pp"]
    )
    return out.sort_values(["protocol", "election_id"])


def main() -> None:
    assembled = engine.assemble()
    frame = assembled.loc[assembled["election_id"].isin(ELECTIONS)].copy()
    if len(frame) != 215:
        raise RuntimeError(f"expected 215 scored rows, found {len(frame)}")
    signals = _shadow_signals()
    rows = pd.concat(
        [full_fit_rows(frame, signals), loeo_rows(frame, signals), rolling_rows(frame, signals)],
        ignore_index=True,
    )
    row_metrics = row_summary(rows)
    points = national_points(rows)
    national_metrics = national_summary(points)
    comparison = points.loc[
        points["protocol"].eq("rolling_origin"),
        ["election_id", "slot", "candidate_name", "actual_pct"],
    ].copy()
    for protocol, column in (
        ("full_fit_in_sample", "full_fit_pred_pct"),
        ("loeo", "loeo_pred_pct"),
        ("rolling_origin", "rolling_pred_pct"),
    ):
        protocol_points = points.loc[
            points["protocol"].eq(protocol),
            ["election_id", "slot", "shadow_pred_pct"],
        ].rename(columns={"shadow_pred_pct": column})
        comparison = comparison.merge(protocol_points, on=["election_id", "slot"], how="left")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUTPUT_DIR / "protocol_row_predictions.csv", index=False, encoding="utf-8-sig")
    row_metrics.to_csv(OUTPUT_DIR / "protocol_row_summary.csv", index=False, encoding="utf-8-sig")
    points.to_csv(OUTPUT_DIR / "protocol_national_points.csv", index=False, encoding="utf-8-sig")
    national_metrics.to_csv(OUTPUT_DIR / "protocol_national_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUTPUT_DIR / "protocol_national_comparison.csv", index=False, encoding="utf-8-sig")
    report = [
        "# Neutral Context Protocol Performance",
        "",
        f"Configuration: `{CONFIG_NAME}`",
        f"Sample size: {SAMPLE_SIZE:,} sentences per election; shadow scale: {SHADOW_SCALE:.2f}.",
        "",
        "This fixed-scale layer is active in the engine. Parameters were selected after reviewing historical outcomes, so LOEO and rolling gains are post-hoc diagnostics rather than clean model-selection estimates.",
        "National points use actual target-election contest-vote weights and are post-election aggregation diagnostics. All actual and predicted shares are normalized among modeled active slots.",
        "",
        "## Row MAE",
        "",
        row_metrics.to_csv(index=False),
        "",
        "## National Candidate-Point MAE",
        "",
        national_metrics.to_csv(index=False),
        "",
        "## Predicted Versus Actual National Shares",
        "",
        comparison.to_csv(index=False),
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(row_metrics.to_string(index=False))
    print()
    print(national_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
