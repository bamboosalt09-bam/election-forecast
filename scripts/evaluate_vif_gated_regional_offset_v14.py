"""Strict nested presidential ablation of the VIF-gated regional offset."""

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
OUTPUT_DIR = ROOT / "outputs" / "vif_gated_regional_offset_v14_experiment"
CHUNGCHEONG = {"sido_30", "sido_36", "sido_43", "sido_44"}
BASE_GAIN = 0.25
VIF_THRESHOLD = 20.0


def _metric_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for election_id, group in frame.groupby("election_id", sort=True):
        chung = group.loc[group["region_id"].isin(CHUNGCHEONG)]
        rows.append(
            {
                "variant": label,
                "election_id": election_id,
                "regional_weighted_mae_pp": float(
                    np.average(
                        np.abs(group["layer_pred"] - group["actual"]) * 100.0,
                        weights=group["contest_votes"],
                    )
                ),
                "chungcheong_weighted_mae_pp": float(
                    np.average(
                        np.abs(chung["layer_pred"] - chung["actual"]) * 100.0,
                        weights=chung["contest_votes"],
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    active.run(
        output_dir=OUTPUT_DIR,
        regional_offset_base_gain=BASE_GAIN,
        regional_offset_vif_threshold=VIF_THRESHOLD,
    )
    experiment = pd.read_csv(OUTPUT_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    comparison = pd.concat(
        [_metric_rows(baseline, "active_v13"), _metric_rows(experiment, "vif_gated_offset_v14")],
        ignore_index=True,
    )
    comparison.to_csv(
        OUTPUT_DIR / "comparison_by_election.csv", index=False, encoding="utf-8-sig"
    )
    baseline_summary = json.loads((BASELINE_DIR / "summary.json").read_text(encoding="utf-8"))
    result_summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    result = {
        "regional_macro_mae_pp": float(
            result_summary["metrics"]["regional_equal_election_macro_mae_pp"]
        ),
        "national_macro_mae_pp": float(
            result_summary["metrics"]["national_equal_election_macro_mae_pp"]
        ),
        "chungcheong_macro_mae_pp": float(
            comparison.loc[
                comparison["variant"].eq("vif_gated_offset_v14"),
                "chungcheong_weighted_mae_pp",
            ].mean()
        ),
    }
    baseline_values = {
        "regional_macro_mae_pp": float(
            baseline_summary["metrics"]["regional_equal_election_macro_mae_pp"]
        ),
        "national_macro_mae_pp": float(
            baseline_summary["metrics"]["national_equal_election_macro_mae_pp"]
        ),
        "chungcheong_macro_mae_pp": float(
            comparison.loc[
                comparison["variant"].eq("active_v13"),
                "chungcheong_weighted_mae_pp",
            ].mean()
        ),
    }
    changes = {key: result[key] - baseline_values[key] for key in result}
    pivot = comparison.pivot(
        index="election_id", columns="variant", values="regional_weighted_mae_pp"
    )
    worst_regression = float(
        (pivot["vif_gated_offset_v14"] - pivot["active_v13"]).max()
    )
    promoted = bool(
        changes["regional_macro_mae_pp"] < 0.0
        and changes["national_macro_mae_pp"] <= 0.0
        and changes["chungcheong_macro_mae_pp"] < 0.0
        and worst_regression <= 0.25
    )
    payload = {
        "experiment": "vif_gated_nonpresidential_regional_offset_v14",
        "base_gain": BASE_GAIN,
        "vif_threshold": VIF_THRESHOLD,
        "presidential_outcomes_used_by_gate_or_offset": False,
        "nonpresidential_offset_prior_strength": 2.0,
        "baseline": baseline_values,
        "experiment_result": result,
        "change": {**changes, "worst_election_regression_pp": worst_regression},
        "promotion": promoted,
        "decision": (
            "Promotion requires aggregate regional improvement, no national "
            "regression, Chungcheong improvement, and <=0.25%p worst election spillover."
        ),
    }
    _atomic_json(payload, OUTPUT_DIR / "decision.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
