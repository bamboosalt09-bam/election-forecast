"""Apply the non-presidential regional offset to every available scored fold."""

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


BASELINE_DIR = ROOT / "outputs" / "active_presidential_nested_v14"
OUTPUT_DIR = ROOT / "outputs" / "all_fold_regional_offset_v14_experiment"
CHUNGCHEONG = {"sido_30", "sido_36", "sido_43", "sido_44"}


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _by_election(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for election_id, group in frame.groupby("election_id", sort=True):
        chung = group.loc[group["region_id"].isin(CHUNGCHEONG)]
        rows.append(
            {
                "variant": variant,
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    active.run(
        output_dir=OUTPUT_DIR,
        regional_offset_base_gain=0.25,
        regional_offset_gate_mode="all_available",
    )
    experiment = pd.read_csv(OUTPUT_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    comparison = pd.concat(
        [_by_election(baseline, "active_v14_vif_gate"), _by_election(experiment, "all_available")],
        ignore_index=True,
    )
    comparison.to_csv(
        OUTPUT_DIR / "comparison_by_election.csv", index=False, encoding="utf-8-sig"
    )
    baseline_summary = json.loads((BASELINE_DIR / "summary.json").read_text(encoding="utf-8"))
    result_summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    baseline_values = {
        "regional_macro_mae_pp": float(
            baseline_summary["metrics"]["regional_equal_election_macro_mae_pp"]
        ),
        "national_macro_mae_pp": float(
            baseline_summary["metrics"]["national_equal_election_macro_mae_pp"]
        ),
        "chungcheong_macro_mae_pp": float(
            comparison.loc[
                comparison["variant"].eq("active_v14_vif_gate"),
                "chungcheong_weighted_mae_pp",
            ].mean()
        ),
    }
    result_values = {
        "regional_macro_mae_pp": float(
            result_summary["metrics"]["regional_equal_election_macro_mae_pp"]
        ),
        "national_macro_mae_pp": float(
            result_summary["metrics"]["national_equal_election_macro_mae_pp"]
        ),
        "chungcheong_macro_mae_pp": float(
            comparison.loc[
                comparison["variant"].eq("all_available"),
                "chungcheong_weighted_mae_pp",
            ].mean()
        ),
    }
    changes = {key: result_values[key] - baseline_values[key] for key in result_values}
    pivot = comparison.pivot(
        index="election_id", columns="variant", values="regional_weighted_mae_pp"
    )
    election_changes = (
        pivot["all_available"] - pivot["active_v14_vif_gate"]
    ).to_dict()
    promoted = bool(
        changes["regional_macro_mae_pp"] < 0.0
        and changes["national_macro_mae_pp"] <= 0.0
        and max(election_changes.values()) <= 0.25
    )
    payload = {
        "experiment": "all_available_regional_offset_v14",
        "base_gain": 0.25,
        "gate_mode": "all_available",
        "presidential_outcomes_used_by_offset": False,
        "baseline": baseline_values,
        "experiment_result": result_values,
        "change": changes,
        "regional_mae_change_by_election_pp": {
            str(key): float(value) for key, value in election_changes.items()
        },
        "effective_gain_by_election": result_summary[
            "regional_offset_gain_by_election"
        ],
        "promotion": promoted,
        "decision": (
            "This is the required all-fold stress test. The VIF gate remains active "
            "unless broad application improves aggregate metrics without material spillover."
        ),
    }
    _atomic_json(payload, OUTPUT_DIR / "decision.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
