"""Reproduce the active Chungcheong error diagnosis.

This script is diagnostic only. Actual outcomes are used to score and explain
the frozen predictions; they are never fed back into forecast construction.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import evaluate_preliminary_slot_shadow_nested as nested  # noqa: E402
from scripts import rederive_layers_through2022 as rederive  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


ACTIVE_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v15"
INPUT = ACTIVE_OUTPUT / "nested_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "chungcheong_error_audit_v15"
CHUNGCHEONG = ("sido_30", "sido_36", "sido_43", "sido_44")
REGION_NAMES = {
    "sido_11": "Seoul",
    "sido_26": "Busan",
    "sido_27": "Daegu",
    "sido_28": "Incheon",
    "sido_29": "Gwangju",
    "sido_30": "Daejeon",
    "sido_31": "Ulsan",
    "sido_36": "Sejong",
    "sido_41": "Gyeonggi",
    "sido_42": "Gangwon",
    "sido_43": "Chungbuk",
    "sido_44": "Chungnam",
    "sido_45": "Jeonbuk",
    "sido_46": "Jeonnam",
    "sido_47": "Gyeongbuk",
    "sido_48": "Gyeongnam",
    "sido_50": "Jeju",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _weighted_metric(group: pd.DataFrame) -> pd.Series:
    weights = pd.to_numeric(group["contest_votes"], errors="coerce").fillna(0.0)
    error = (group["layer_pred"] - group["actual"]) * 100.0
    return pd.Series(
        {
            "row_count": len(group),
            "weighted_mae_pp": float(np.average(np.abs(error), weights=weights)),
            "weighted_rmse_pp": float(
                np.sqrt(np.average(np.square(error), weights=weights))
            ),
            "weighted_bias_pp": float(np.average(error, weights=weights)),
        }
    )


def _vif(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    x = frame[list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rows: list[dict[str, object]] = []
    for column in columns:
        y = x[column].to_numpy(float)
        others = x[[name for name in columns if name != column]].to_numpy(float)
        design = np.column_stack([np.ones(len(x), dtype=float), others])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        total = float(np.square(y - y.mean()).sum())
        residual = float(np.square(y - fitted).sum())
        if total <= 1e-15 or residual <= 1e-15:
            vif = float("inf")
        else:
            r_squared = min(max(1.0 - residual / total, 0.0), 1.0)
            vif = float("inf") if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)
        rows.append({"predictor": column, "vif": vif, "rows": len(x)})
    return pd.DataFrame(rows)


def _fold_vif(full: pd.DataFrame) -> pd.DataFrame:
    order = [*rederive.WARMUP_ELECTIONS, *nested.ELECTIONS]
    lookup = {election_id: index for index, election_id in enumerate(order)}
    warmup_ids = set(rederive.WARMUP_ELECTIONS)
    rows: list[pd.DataFrame] = []
    for target in nested.ELECTIONS:
        train = full.loc[full["_order"] < lookup[target]].copy().reset_index(drop=True)
        target_rows = full.loc[full["election_id"].eq(target)].copy().reset_index(drop=True)
        train, _ = nested.engine.rolling_training_with_slot_backfill(
            train, target_rows, warmup_ids
        )
        vif = _vif(train, nested.BASE_PREDICTORS)
        vif.insert(0, "target_election", target)
        rows.append(vif)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT, encoding="utf-8-sig").copy()
    frame["error_pp"] = (frame["layer_pred"] - frame["actual"]) * 100.0

    regional = (
        frame.groupby("region_id", sort=True)
        .apply(_weighted_metric, include_groups=False)
        .reset_index()
    )
    regional.insert(1, "region_name", regional["region_id"].map(REGION_NAMES))
    regional = regional.sort_values("weighted_mae_pp", ascending=False)

    national_actual = (
        frame.groupby(["election_id", "candidate_name_x"], sort=True)
        .apply(
            lambda group: float(
                np.average(group["actual"], weights=group["contest_votes"])
            ),
            include_groups=False,
        )
        .rename("national_actual")
        .reset_index()
    )
    winner_names = national_actual.loc[
        national_actual.groupby("election_id")["national_actual"].transform("max").eq(
            national_actual["national_actual"]
        ),
        ["election_id", "candidate_name_x"],
    ]
    winners = (
        frame.merge(
            winner_names,
            on=["election_id", "candidate_name_x"],
            how="inner",
            validate="many_to_one",
        )
        .loc[lambda data: data["region_id"].isin(CHUNGCHEONG)]
        [
            [
                "election_id",
                "region_id",
                "candidate_name_x",
                "layer_pred",
                "actual",
                "error_pp",
            ]
        ]
        .copy()
    )
    winners.insert(2, "region_name", winners["region_id"].map(REGION_NAMES))
    winners = winners.rename(columns={"candidate_name_x": "national_winner"})
    winners["pred_pct"] = winners.pop("layer_pred") * 100.0
    winners["actual_pct"] = winners.pop("actual") * 100.0

    election_2022 = frame.loc[frame["election_id"].eq("pres_2022")].copy()
    region_volume = election_2022.drop_duplicates("region_id").set_index("region_id")[
        "contest_votes"
    ]
    region_weight = region_volume / region_volume.sum()
    cancellation = election_2022[
        ["region_id", "candidate_name_x", "layer_pred", "actual", "error_pp"]
    ].copy()
    cancellation.insert(1, "region_name", cancellation["region_id"].map(REGION_NAMES))
    cancellation["national_weight"] = cancellation["region_id"].map(region_weight)
    cancellation["national_error_contribution_pp"] = (
        cancellation["error_pp"] * cancellation["national_weight"]
    )

    with active.strict_input_policy():
        full = nested._prepare_rows()
    pooled = full.loc[full["election_id"].isin(nested.ELECTIONS)].copy()
    pooled_vif = _vif(pooled, nested.BASE_PREDICTORS)
    fold_vif = _fold_vif(full)

    stage_columns = [
        "election_id",
        "region_id",
        "candidate_name_x",
        "direct_party_recent_base",
        "candidate_ballot_recent_base",
        "durable_core_raw",
        "critical_support_raw",
        "core_voting_mass_effective",
        "critical_voting_mass_effective",
        "swing_voting_mass_effective",
        "camp_regional_anchored_pred",
        "regional_accent_signal_diagnostic",
        "regional_accent_log_shift",
        "layer_pred",
        "actual",
        "error_pp",
    ]
    stage_2012 = frame.loc[
        frame["election_id"].eq("pres_2012")
        & frame["region_id"].isin(CHUNGCHEONG),
        stage_columns,
    ].copy()
    stage_2012["direct_party_two_camp_share"] = stage_2012.groupby("region_id")[
        "direct_party_recent_base"
    ].transform(lambda values: values / values.sum())

    active_summary = json.loads((ACTIVE_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "model": active_summary["policy_version"],
        "diagnostic_only": True,
        "actual_outcomes_used_for_forecast": False,
        "regional_macro_mae_pp": float(
            active_summary["metrics"]["regional_equal_election_macro_mae_pp"]
        ),
        "national_macro_mae_pp": float(
            active_summary["metrics"]["national_equal_election_macro_mae_pp"]
        ),
        "winner_accuracy": float(active_summary["metrics"]["winner_accuracy"]),
        "vif_input_policy": "strict_undated_curated_inputs_disabled",
        "chungcheong_regions": list(CHUNGCHEONG),
        "chungcheong_weighted_mae_pp": {
            row.region_name: float(row.weighted_mae_pp)
            for row in regional.loc[regional["region_id"].isin(CHUNGCHEONG)].itertuples()
        },
        "national_winner_chungcheong_mean_bias_pp": float(winners["error_pp"].mean()),
        "national_winner_chungcheong_underprediction_rows": int(
            winners["error_pp"].lt(0.0).sum()
        ),
        "national_winner_chungcheong_rows": int(len(winners)),
        "max_fold_vif": float(fold_vif["vif"].replace(np.inf, np.nan).max()),
        "interpretation": (
            "V15 materially reduces the missing Chungcheong regional-identity error, "
            "especially in 2002, 2007, and 2012. The eventual national winner remains "
            "underpredicted in most Chungcheong rows, so residual central regression "
            "and sparse recipient evidence remain; a fixed party bonus is still not "
            "supported."
        ),
    }

    _atomic_csv(regional, OUTPUT_DIR / "regional_performance.csv")
    _atomic_csv(winners, OUTPUT_DIR / "chungcheong_national_winner_errors.csv")
    _atomic_csv(cancellation, OUTPUT_DIR / "pres_2022_cancellation.csv")
    _atomic_csv(pooled_vif, OUTPUT_DIR / "vif_pooled.csv")
    _atomic_csv(fold_vif, OUTPUT_DIR / "vif_by_fold.csv")
    _atomic_csv(stage_2012, OUTPUT_DIR / "pres_2012_chungcheong_stage.csv")
    _atomic_json(summary, OUTPUT_DIR / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
