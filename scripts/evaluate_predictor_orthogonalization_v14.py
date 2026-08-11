"""Strict nested ablation for outcome-blind predictor orthogonalization."""

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

from scripts import run_active_presidential_model as active  # noqa: E402


BASELINE_DIR = ROOT / "outputs" / "active_presidential_nested_v13"
OUTPUT_DIR = ROOT / "outputs" / "predictor_orthogonalization_v14_experiment"
CHUNGCHEONG = {"sido_30", "sido_36", "sido_43", "sido_44"}
PAIRS = (
    ("issue_advantage", "rif"),
    ("landscape_bloc_alignment", "landscape_centrist"),
)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _regional_metric(frame: pd.DataFrame, regions: set[str] | None = None) -> float:
    work = frame if regions is None else frame.loc[frame["region_id"].isin(regions)]
    election_scores: list[float] = []
    for _, group in work.groupby("election_id"):
        error = np.abs(group["layer_pred"] - group["actual"]) * 100.0
        election_scores.append(
            float(np.average(error, weights=group["contest_votes"]))
        )
    return float(np.mean(election_scores))


def _by_election(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for election_id, group in frame.groupby("election_id", sort=True):
        all_error = np.abs(group["layer_pred"] - group["actual"]) * 100.0
        chung = group.loc[group["region_id"].isin(CHUNGCHEONG)]
        chung_error = np.abs(chung["layer_pred"] - chung["actual"]) * 100.0
        rows.append(
            {
                "variant": label,
                "election_id": election_id,
                "regional_weighted_mae_pp": float(
                    np.average(all_error, weights=group["contest_votes"])
                ),
                "chungcheong_weighted_mae_pp": float(
                    np.average(chung_error, weights=chung["contest_votes"])
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    active.run(
        output_dir=OUTPUT_DIR,
        predictor_orthogonalization_pairs=PAIRS,
        regional_offset_base_gain=0.0,
    )
    experiment = pd.read_csv(OUTPUT_DIR / "nested_predictions.csv", encoding="utf-8-sig")

    baseline_by = _by_election(baseline, "active_v13")
    experiment_by = _by_election(experiment, "orthogonalized_v14")
    comparison = pd.concat([baseline_by, experiment_by], ignore_index=True)
    comparison.to_csv(
        OUTPUT_DIR / "comparison_by_election.csv", index=False, encoding="utf-8-sig"
    )

    baseline_summary = json.loads((BASELINE_DIR / "summary.json").read_text(encoding="utf-8"))
    experiment_summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    baseline_regional = float(
        baseline_summary["metrics"]["regional_equal_election_macro_mae_pp"]
    )
    experiment_regional = float(
        experiment_summary["metrics"]["regional_equal_election_macro_mae_pp"]
    )
    baseline_national = float(
        baseline_summary["metrics"]["national_equal_election_macro_mae_pp"]
    )
    experiment_national = float(
        experiment_summary["metrics"]["national_equal_election_macro_mae_pp"]
    )
    baseline_chung = _regional_metric(baseline, CHUNGCHEONG)
    experiment_chung = _regional_metric(experiment, CHUNGCHEONG)

    payload = {
        "experiment": "fold_local_predictor_orthogonalization_v14",
        "pairs": [list(pair) for pair in PAIRS],
        "target_outcomes_used_by_transform": False,
        "baseline": {
            "regional_macro_mae_pp": baseline_regional,
            "national_macro_mae_pp": baseline_national,
            "chungcheong_macro_mae_pp": baseline_chung,
        },
        "experiment_result": {
            "regional_macro_mae_pp": experiment_regional,
            "national_macro_mae_pp": experiment_national,
            "chungcheong_macro_mae_pp": experiment_chung,
        },
        "change": {
            "regional_macro_mae_pp": experiment_regional - baseline_regional,
            "national_macro_mae_pp": experiment_national - baseline_national,
            "chungcheong_macro_mae_pp": experiment_chung - baseline_chung,
        },
        "promotion": False,
        "decision": (
            "Diagnostic ablation only. Promotion requires lower aggregate error, "
            "lower Chungcheong error, and no material election-level regression."
        ),
    }
    _atomic_json(payload, OUTPUT_DIR / "decision.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
