"""Evaluate lane-preserving tactical transfer without overwriting active v11."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "strategic_lane_transfer_v12_experiment"
BASELINE_SUMMARY = ROOT / "outputs" / "active_presidential_nested_v11" / "summary.json"


def run() -> dict[str, object]:
    baseline = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    policy = copy.deepcopy(active.load_policy())
    policy["structural_layers"]["strategic_lane_transfer"]["enabled"] = True
    policy["policy_version"] = "experimental_v12_strategic_lane_transfer"

    original_output = active.OUTPUT_DIR
    original_loader = active.load_policy
    active.OUTPUT_DIR = OUTPUT_DIR
    active.load_policy = lambda path=active.CONFIG_PATH: active.validate_policy(policy)
    try:
        candidate = active.run()
    finally:
        active.OUTPUT_DIR = original_output
        active.load_policy = original_loader

    predictions = pd.read_csv(OUTPUT_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    transfer = pd.to_numeric(
        predictions.get("strategic_lane_transfer_out", 0.0), errors="coerce"
    ).fillna(0.0)
    by_election = (
        predictions.assign(_transfer_out=transfer)
        .groupby("election_id", as_index=False)
        .agg(
            transfer_out_mean=("_transfer_out", "mean"),
            transfer_out_max=("_transfer_out", "max"),
        )
    )
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    decision = {
        "experiment": "strategic_lane_transfer",
        "scope": "strict nested through-2022 development folds",
        "post_2022_outcomes_used": False,
        "baseline_policy": baseline["policy_version"],
        "candidate_policy": candidate["policy_version"],
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "regional_mae_delta_pp": float(
            candidate_metrics["regional_equal_election_macro_mae_pp"]
            - baseline_metrics["regional_equal_election_macro_mae_pp"]
        ),
        "national_mae_delta_pp": float(
            candidate_metrics["national_equal_election_macro_mae_pp"]
            - baseline_metrics["national_equal_election_macro_mae_pp"]
        ),
        "transfer_by_election": by_election.to_dict("records"),
        "rule": (
            "nonmajor effective critical support remains a lane reservoir and "
            "moves only to aligned major-party candidates under PIT-safe "
            "wasted-vote pressure"
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
