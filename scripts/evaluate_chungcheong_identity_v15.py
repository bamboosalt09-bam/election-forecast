"""Evaluate the PIT-safe Chungcheong regional-identity reservoir on all folds."""

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
OUTPUT_DIR = ROOT / "outputs" / "chungcheong_identity_v15_experiment"
CHUNGCHEONG = {"sido_30", "sido_36", "sido_43", "sido_44"}
GAIN = 0.50
SHIFT_CAP = 0.08


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    active.run(
        output_dir=OUTPUT_DIR,
        chungcheong_identity_gain=GAIN,
        chungcheong_identity_shift_cap=SHIFT_CAP,
    )
    experiment = pd.read_csv(OUTPUT_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    comparison = pd.concat(
        [_metrics(baseline, "active_v14"), _metrics(experiment, "identity_v15")],
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
        "winner_accuracy": float(baseline_summary["metrics"]["winner_accuracy"]),
        "chungcheong_macro_mae_pp": float(
            comparison.loc[
                comparison["variant"].eq("active_v14"), "chungcheong_weighted_mae_pp"
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
        "winner_accuracy": float(result_summary["metrics"]["winner_accuracy"]),
        "chungcheong_macro_mae_pp": float(
            comparison.loc[
                comparison["variant"].eq("identity_v15"), "chungcheong_weighted_mae_pp"
            ].mean()
        ),
    }
    changes = {key: result_values[key] - baseline_values[key] for key in result_values}
    pivot = comparison.pivot(
        index="election_id", columns="variant", values="regional_weighted_mae_pp"
    )
    election_changes = (pivot["identity_v15"] - pivot["active_v14"]).to_dict()
    promoted = bool(
        changes["regional_macro_mae_pp"] < 0.0
        and changes["national_macro_mae_pp"] <= 0.0
        and changes["chungcheong_macro_mae_pp"] < 0.0
        and changes["winner_accuracy"] >= 0.0
        and max(election_changes.values()) <= 0.25
    )
    payload = {
        "experiment": "chungcheong_identity_v15",
        "gain": GAIN,
        "shift_cap": SHIFT_CAP,
        "strict_point_in_time": True,
        "target_outcomes_used_by_layer": False,
        "alignment_evidence_is_dated_pre_election": True,
        "baseline": baseline_values,
        "experiment_result": result_values,
        "change": changes,
        "regional_mae_change_by_election_pp": {
            str(key): float(value) for key, value in election_changes.items()
        },
        "promotion": promoted,
    }
    _atomic_json(payload, OUTPUT_DIR / "decision.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
