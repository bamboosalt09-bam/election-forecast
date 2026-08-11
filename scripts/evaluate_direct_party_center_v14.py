"""Strict nested ablation for bounded direct-party terrain preservation."""

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
OUTPUT_DIR = ROOT / "outputs" / "direct_party_center_v14_experiment"
CHUNGCHEONG = {"sido_30", "sido_36", "sido_43", "sido_44"}
BASE_GAIN = 0.25


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for election_id, group in frame.groupby("election_id", sort=True):
        chung = group.loc[group["region_id"].isin(CHUNGCHEONG)]
        error = np.abs(group["layer_pred"] - group["actual"]) * 100.0
        chung_error = np.abs(chung["layer_pred"] - chung["actual"]) * 100.0
        rows.append(
            {
                "variant": label,
                "election_id": election_id,
                "regional_weighted_mae_pp": float(
                    np.average(error, weights=group["contest_votes"])
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
        direct_party_center_base_gain=BASE_GAIN,
        regional_offset_base_gain=0.0,
    )
    experiment = pd.read_csv(OUTPUT_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    by_election = pd.concat(
        [_metrics(baseline, "active_v13"), _metrics(experiment, "direct_party_center_v14")],
        ignore_index=True,
    )
    by_election.to_csv(
        OUTPUT_DIR / "comparison_by_election.csv", index=False, encoding="utf-8-sig"
    )

    baseline_summary = json.loads((BASELINE_DIR / "summary.json").read_text(encoding="utf-8"))
    result_summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    baseline_regional = float(
        baseline_summary["metrics"]["regional_equal_election_macro_mae_pp"]
    )
    result_regional = float(
        result_summary["metrics"]["regional_equal_election_macro_mae_pp"]
    )
    baseline_national = float(
        baseline_summary["metrics"]["national_equal_election_macro_mae_pp"]
    )
    result_national = float(
        result_summary["metrics"]["national_equal_election_macro_mae_pp"]
    )
    baseline_chung = float(
        by_election.loc[by_election["variant"].eq("active_v13"), "chungcheong_weighted_mae_pp"].mean()
    )
    result_chung = float(
        by_election.loc[
            by_election["variant"].eq("direct_party_center_v14"),
            "chungcheong_weighted_mae_pp",
        ].mean()
    )
    change_by = by_election.pivot(
        index="election_id", columns="variant", values="regional_weighted_mae_pp"
    )
    worst_election_regression = float(
        (
            change_by["direct_party_center_v14"]
            - change_by["active_v13"]
        ).max()
    )
    aggregate_improved = result_regional < baseline_regional and result_national < baseline_national
    chung_improved = result_chung < baseline_chung
    no_material_spillover = worst_election_regression <= 0.25
    promoted = bool(aggregate_improved and chung_improved and no_material_spillover)
    payload = {
        "experiment": "bounded_direct_party_center_v14",
        "base_gain": BASE_GAIN,
        "target_outcomes_used_by_transform": False,
        "minor_and_independent_share_preserved": True,
        "shock_attenuation": True,
        "baseline": {
            "regional_macro_mae_pp": baseline_regional,
            "national_macro_mae_pp": baseline_national,
            "chungcheong_macro_mae_pp": baseline_chung,
        },
        "experiment_result": {
            "regional_macro_mae_pp": result_regional,
            "national_macro_mae_pp": result_national,
            "chungcheong_macro_mae_pp": result_chung,
        },
        "change": {
            "regional_macro_mae_pp": result_regional - baseline_regional,
            "national_macro_mae_pp": result_national - baseline_national,
            "chungcheong_macro_mae_pp": result_chung - baseline_chung,
            "worst_election_regression_pp": worst_election_regression,
        },
        "promotion": promoted,
        "decision": (
            "Promote only when aggregate regional and national errors improve, "
            "Chungcheong improves, and no election worsens by more than 0.25%p."
        ),
    }
    _atomic_json(payload, OUTPUT_DIR / "decision.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
