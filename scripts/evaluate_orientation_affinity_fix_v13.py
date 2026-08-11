"""Evaluate the same-lane affinity condition fix without overwriting v12."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "orientation_affinity_fix_v13_experiment"
BASELINE_SUMMARY = ROOT / "outputs" / "active_presidential_nested_v12" / "summary.json"


def corrected_orientation_affinity(left, right) -> float:
    left_label = active.nested.engine._orientation_label(left)
    right_label = active.nested.engine._orientation_label(right)
    if left_label == right_label:
        return 1.0
    pair = {left_label, right_label}
    if pair == {"conservative", "conservative_centrist"}:
        return 0.85
    if pair == {"liberal_centrist", "centrist"}:
        return 0.65
    if pair == {"liberal", "liberal_centrist"}:
        return 0.70
    if pair == {"conservative", "centrist"}:
        return 0.35
    if pair == {"liberal", "centrist"}:
        return 0.45
    if pair == {"conservative_centrist", "centrist"}:
        return 0.50
    return 0.0


def run() -> dict[str, object]:
    baseline = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    policy = copy.deepcopy(active.load_policy())
    policy["policy_version"] = "experimental_v13_orientation_affinity_fix"
    original_output = active.OUTPUT_DIR
    original_loader = active.load_policy
    original_affinity = active.nested.engine._orientation_affinity
    active.OUTPUT_DIR = OUTPUT_DIR
    active.load_policy = lambda path=active.CONFIG_PATH: active.validate_policy(policy)
    active.nested.engine._orientation_affinity = corrected_orientation_affinity
    try:
        candidate = active.run()
    finally:
        active.OUTPUT_DIR = original_output
        active.load_policy = original_loader
        active.nested.engine._orientation_affinity = original_affinity

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    decision = {
        "experiment": "orientation_affinity_condition_fix",
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
        "bug": (
            "the previous set-intersection condition assigned 0.65 affinity "
            "to conservative versus liberal-centrist candidates"
        ),
        "fix": (
            "liberal-centrist receives 0.65 only against centrist and 0.70 "
            "against liberal; cross-camp affinity is zero"
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
