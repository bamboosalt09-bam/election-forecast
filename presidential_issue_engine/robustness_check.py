"""Robustness checks for issue vote features and regional bloc priors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from issue_vote_engine import (
    ORDER,
    PREDICTORS,
    RIDGE_ALPHA,
    _macro_issue_reinforcement_table,
    apply_prediction_postprocess,
    apply_region_residual_calibration,
    apply_third_candidate_prediction_adjustment,
    apply_withdrawn_candidate_prediction_adjustment,
    assemble,
    election_epoch_sample_weight,
    historical_presidential_warmup_frame,
    loeo_cv,
    normalize_vote_share_predictions,
    neutral_issue_context_scale,
    normalized_vote_share_target,
    ols,
    ridge_fit,
    ridge_predict,
    scored_contest_rows,
    rolling_training_with_slot_backfill,
    rolling_origin_cv,
)

COMPETITION_ELECTIONS = list(ORDER)
ROLLING_WARMUP_ELECTIONS = ["pres_1997"]


def competition_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict evaluation to the through-2022 weight-selection panel."""

    return df[df["election_id"].isin(COMPETITION_ELECTIONS)].copy()


def rolling_warmup_frame(_: pd.DataFrame) -> pd.DataFrame:
    """Return pre-evaluation presidential rows used only as chronological warmup."""

    warmup = historical_presidential_warmup_frame()
    return warmup[warmup["election_id"].isin(ROLLING_WARMUP_ELECTIONS)].copy()


def slot_mean_cv(df: pd.DataFrame) -> float:
    """Leave-one-election-out MAE for a slot-mean baseline."""

    errs: list[float] = []
    for election_id in df["election_id"].unique():
        train = df[df.election_id != election_id].copy()
        test = scored_contest_rows(df[df.election_id == election_id])
        train["_target"] = normalized_vote_share_target(train)
        means = train.groupby("slot")["_target"].mean()
        pred = test["slot"].map(means).fillna(train["vote_share"].mean())
        errs.extend(np.abs(pred.to_numpy() - normalized_vote_share_target(test)) * 100)
    return float(np.mean(errs))


def global_mean_cv(df: pd.DataFrame) -> float:
    """Leave-one-election-out MAE for a global mean baseline."""

    errs: list[float] = []
    for election_id in df["election_id"].unique():
        train = df[df.election_id != election_id]
        test = scored_contest_rows(df[df.election_id == election_id])
        pred = normalized_vote_share_target(train).mean()
        errs.extend(np.abs(pred - normalized_vote_share_target(test)) * 100)
    return float(np.mean(errs))


def r2(df: pd.DataFrame, cols: list[str]) -> float:
    """Return in-sample R-squared for a predictor subset."""

    _, value, _, _ = ols(df[cols].to_numpy(float), normalized_vote_share_target(df))
    return value


def group_cv(df: pd.DataFrame, predictors: list[str], alpha: float = RIDGE_ALPHA) -> dict[str, float]:
    """Return LOEO MAE by selected region groups."""

    groups = {
        "All": None,
        "Honam": ["sido_45", "sido_46", "sido_29"],
        "TK": ["sido_27", "sido_47"],
        "PK": ["sido_26", "sido_48", "sido_31"],
        "Gangwon": ["sido_42"],
    }
    errs: dict[str, list[float]] = {name: [] for name in groups}
    for election_id in df["election_id"].unique():
        train = df[df.election_id != election_id]
        test = scored_contest_rows(df[df.election_id == election_id])
        beta, _, _, _, means, scales = ridge_fit(
            train[predictors].to_numpy(float),
            normalized_vote_share_target(train),
            alpha=alpha,
            sample_weight=election_epoch_sample_weight(train),
        )
        pred = ridge_predict(beta, test[predictors].to_numpy(float), means, scales)
        test["err"] = np.abs(pred - normalized_vote_share_target(test)) * 100
        for row in test.itertuples():
            errs["All"].append(float(row.err))
            for group_name, region_ids in groups.items():
                if region_ids is not None and row.region_id in region_ids:
                    errs[group_name].append(float(row.err))
    return {name: float(np.mean(values)) for name, values in errs.items() if values}


def rolling_group_cv(
    df: pd.DataFrame,
    predictors: list[str],
    alpha: float = RIDGE_ALPHA,
    warmup: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Return final-engine chronological CV MAE by selected region groups."""

    rows = rolling_origin_error_frame(
        df,
        predictors,
        alpha=alpha,
        warmup=warmup,
        warmup_order=ROLLING_WARMUP_ELECTIONS,
    )
    if rows.empty:
        return {}

    groups = {
        "All": None,
        "Honam": ["sido_45", "sido_46", "sido_29"],
        "TK": ["sido_27", "sido_47"],
        "PK": ["sido_26", "sido_48", "sido_31"],
        "Gangwon": ["sido_42"],
    }
    out: dict[str, float] = {}
    for group_name, region_ids in groups.items():
        if region_ids is None:
            values = rows["abs_err_pp"]
        else:
            values = rows.loc[rows["region_id"].isin(region_ids), "abs_err_pp"]
        if not values.empty:
            out[group_name] = float(values.mean())
    return out


def rolling_origin_error_frame(
    df: pd.DataFrame,
    predictors: list[str],
    alpha: float = RIDGE_ALPHA,
    warmup: pd.DataFrame | None = None,
    warmup_order: list[str] = ROLLING_WARMUP_ELECTIONS,
) -> pd.DataFrame:
    """Return row-level rolling-origin errors using final engine adjustments."""

    full_order = [*warmup_order, *COMPETITION_ELECTIONS] if warmup is not None and not warmup.empty else COMPETITION_ELECTIONS
    order_lookup = {election_id: index for index, election_id in enumerate(full_order)}
    frame = df.copy()
    warmup_ids: set[str] = set()
    if warmup is not None and not warmup.empty:
        warmup_ids = set(warmup["election_id"].astype(str))
        frame = pd.concat([warmup.copy(), frame], ignore_index=True, sort=False)
        for predictor in predictors:
            frame[predictor] = pd.to_numeric(frame[predictor], errors="coerce").fillna(0.0)
        frame = frame.copy()
    frame["_order"] = frame["election_id"].map(order_lookup)
    rows: list[pd.DataFrame] = []

    for election_id in COMPETITION_ELECTIONS:
        target_order = order_lookup[election_id]
        train = frame[frame["_order"] < target_order].copy()
        train["_rolling_target"] = normalized_vote_share_target(train)
        test = scored_contest_rows(frame[frame["election_id"] == election_id])
        train, residual_mask = rolling_training_with_slot_backfill(
            train,
            test,
            warmup_ids,
        )
        if train.empty or test.empty:
            continue

        X_train = train[predictors].to_numpy(float)
        y_train = train["_rolling_target"].to_numpy(float)
        X_test = test[predictors].to_numpy(float)
        beta, _, _, _, means, scales = ridge_fit(
            X_train,
            y_train,
            alpha=alpha,
            sample_weight=election_epoch_sample_weight(train),
        )
        train_pred = ridge_predict(beta, X_train, means, scales)
        pred = ridge_predict(beta, X_test, means, scales)
        train_pred = apply_third_candidate_prediction_adjustment(train, train_pred)
        train_pred = apply_withdrawn_candidate_prediction_adjustment(train, train_pred)
        pred = apply_third_candidate_prediction_adjustment(test, pred)
        pred = apply_withdrawn_candidate_prediction_adjustment(test, pred)
        residual_train = train.loc[residual_mask].copy()
        residual_train_pred = train_pred[residual_mask]
        pred = apply_region_residual_calibration(
            residual_train,
            test,
            residual_train_pred,
            pred,
        )
        pred = normalize_vote_share_predictions(test, pred)
        pred = apply_prediction_postprocess(test, pred)

        actual = normalized_vote_share_target(test)
        fold = test[["election_id", "region_id", "slot", "candidate_name"]].copy()
        fold["pred"] = pred
        fold["actual"] = actual
        fold["err_pp"] = (pred - actual) * 100
        fold["abs_err_pp"] = np.abs(pred - actual) * 100
        fold["neutral_issue_context_scale"] = neutral_issue_context_scale()
        rows.append(fold)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    """Print baseline and full-model robustness checks."""

    all_rows = assemble()
    df = competition_frame(all_rows)
    warmup = rolling_warmup_frame(all_rows)
    scored_df = scored_contest_rows(df)
    print(f"[data] training-context rows={len(df)} | scored rows={len(scored_df)}")
    print(f"competition elections: {COMPETITION_ELECTIONS}")
    print(f"rolling warmup only: {ROLLING_WARMUP_ELECTIONS}")
    print(f"predictors: {PREDICTORS}\n")
    macro_diagnostics = _macro_issue_reinforcement_table()
    macro_out = "presidential_issue_engine/report/tables/macro_issue_reinforcement_diagnostics.csv"
    macro_diagnostics.to_csv(macro_out, index=False, encoding="utf-8-sig")
    print(f"macro issue diagnostics: {macro_out}\n")

    print("LOEO CV MAE (%p, lower is better)")
    print(f"  global mean                  : {global_mean_cv(df):.2f}")
    print(f"  slot mean                    : {slot_mean_cv(df):.2f}")
    print(
        "  slot + regional_base         : "
        f"{loeo_cv(df, ['slot_A', 'slot_B', 'regional_base'], alpha=0.0):.2f}"
    )
    print(
        "  slot + regional + issue      : "
        f"{loeo_cv(df, ['slot_A', 'slot_B', 'regional_base', 'issue_advantage', 'rif'], alpha=0.0):.2f}"
    )
    print(
        "  NEC prior + issue (ridge)    : "
        f"{loeo_cv(df, PREDICTORS):.2f}"
    )
    print(f"  ridge alpha                  : {RIDGE_ALPHA:.2f}\n")

    rolling_mae, rolling_by_election = rolling_origin_cv(
        df,
        PREDICTORS,
        election_order=COMPETITION_ELECTIONS,
        warmup=warmup,
        warmup_order=ROLLING_WARMUP_ELECTIONS,
    )
    print("Rolling-origin CV MAE (%p, leakage-safe)")
    print(f"  NEC prior + issue (ridge)    : {rolling_mae:.2f}")
    for election_id, value in rolling_by_election.items():
        print(f"  {election_id:27s}: {value:.2f}")
    print()

    print("R2 ladder")
    print(f"  slot                         : {r2(scored_df, ['slot_A', 'slot_B']):.3f}")
    print(f"  slot + regional_base         : {r2(scored_df, ['slot_A', 'slot_B', 'regional_base']):.3f}")
    print(
        "  slot + regional + issue      : "
        f"{r2(scored_df, ['slot_A', 'slot_B', 'regional_base', 'issue_advantage', 'rif']):.3f}"
    )
    _, ridge_r2, _, _, _, _ = ridge_fit(
        scored_df[PREDICTORS].to_numpy(float),
        normalized_vote_share_target(scored_df),
        alpha=RIDGE_ALPHA,
        sample_weight=election_epoch_sample_weight(scored_df),
    )
    print(f"  NEC prior + issue (ridge)    : {ridge_r2:.3f}\n")

    print("Full-model rolling-origin CV by region group")
    region_errors = rolling_group_cv(df, PREDICTORS, warmup=warmup)
    for name, value in region_errors.items():
        print(f"  {name:7s}: {value:.2f}")

    rows = rolling_origin_error_frame(
        df,
        PREDICTORS,
        warmup=warmup,
        warmup_order=ROLLING_WARMUP_ELECTIONS,
    )
    if not rows.empty:
        out = "presidential_issue_engine/report/tables/competition_rolling_region_errors.csv"
        rows.to_csv(out, index=False, encoding="utf-8-sig")
        meta = scored_contest_rows(all_rows)[
            [
                "election_id",
                "region_id",
                "slot",
                "candidate_name",
                "bloc",
                "votes",
                "vote_share",
            ]
        ].copy()
        meta["contest_votes"] = meta.groupby(["election_id", "region_id"])["votes"].transform("sum")
        canonical_rows = rows.merge(
            meta,
            on=["election_id", "region_id", "slot", "candidate_name"],
            how="left",
        )
        canonical_rows["eval_type"] = "current_final_adjusted_rolling_origin"
        canonical_rows["metric_scope"] = "scored_elections_2002_2022_pres1997_warmup"
        canonical_rows = canonical_rows[
            [
                "election_id",
                "region_id",
                "slot",
                "candidate_name",
                "bloc",
                "votes",
                "vote_share",
                "pred",
                "actual",
                "err_pp",
                "abs_err_pp",
                "neutral_issue_context_scale",
                "eval_type",
                "metric_scope",
                "contest_votes",
            ]
        ]
        canonical_out = "presidential_issue_engine/report/tables/issue_vote_engine_rolling_predictions.csv"
        canonical_rows.to_csv(canonical_out, index=False, encoding="utf-8-sig")
        row_summary = (
            canonical_rows.groupby("election_id", as_index=False)
            .agg(row_mae_pp=("abs_err_pp", "mean"), n_rows=("abs_err_pp", "size"))
        )
        row_summary = pd.concat(
            [
                row_summary,
                pd.DataFrame(
                    [
                        {
                            "election_id": "Overall",
                            "row_mae_pp": canonical_rows["abs_err_pp"].mean(),
                            "n_rows": len(canonical_rows),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        row_summary_out = "presidential_issue_engine/report/tables/issue_vote_engine_rolling_row_summary.csv"
        row_summary.to_csv(row_summary_out, index=False, encoding="utf-8-sig")
        national_rows: list[dict[str, object]] = []
        for (election_id, slot, candidate_name, bloc), group in canonical_rows.groupby(
            ["election_id", "slot", "candidate_name", "bloc"],
            sort=True,
        ):
            weights = pd.to_numeric(group["contest_votes"], errors="coerce").fillna(0.0)
            pred = float(np.average(group["pred"], weights=weights))
            actual = float(np.average(group["actual"], weights=weights))
            national_rows.append(
                {
                    "election_id": election_id,
                    "slot": slot,
                    "candidate_name": candidate_name,
                    "bloc": bloc,
                    "pred_pct": pred * 100.0,
                    "actual_pct": actual * 100.0,
                    "err_pp": (pred - actual) * 100.0,
                    "abs_err_pp": abs(pred - actual) * 100.0,
                    "weight_note": "actual_contest_votes_observed_after_election",
                    "eval_type": "post_election_actual_turnout_weighted_rolling_diagnostic",
                    "metric_scope": "scored_elections_2002_2022_pres1997_warmup",
                }
            )
        national_out = "presidential_issue_engine/report/tables/issue_vote_engine_rolling_national_summary.csv"
        pd.DataFrame(national_rows).sort_values(["election_id", "slot"]).to_csv(
            national_out,
            index=False,
            encoding="utf-8-sig",
        )
        region_names = {
            "sido_11": "서울",
            "sido_26": "부산",
            "sido_27": "대구",
            "sido_28": "인천",
            "sido_29": "광주",
            "sido_30": "대전",
            "sido_31": "울산",
            "sido_36": "세종",
            "sido_41": "경기",
            "sido_42": "강원",
            "sido_43": "충북",
            "sido_44": "충남",
            "sido_45": "전북",
            "sido_46": "전남",
            "sido_47": "경북",
            "sido_48": "경남",
            "sido_50": "제주",
        }
        region_out = (
            rows.groupby("region_id", as_index=False)["abs_err_pp"]
            .mean()
            .sort_values("abs_err_pp", ascending=False)
        )
        region_out.insert(1, "region_name", region_out["region_id"].map(region_names).fillna(""))
        region_summary_out = "presidential_issue_engine/report/tables/competition_rolling_region_summary.csv"
        region_out.to_csv(region_summary_out, index=False, encoding="utf-8-sig")
        print(f"\nsaved row errors: {out}")
        print(f"saved canonical rolling rows: {canonical_out}")
        print(f"saved rolling row summary: {row_summary_out}")
        print(f"saved rolling national summary: {national_out}")
        print(f"saved region summary: {region_summary_out}")


if __name__ == "__main__":
    main()
