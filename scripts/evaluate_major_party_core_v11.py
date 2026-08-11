"""Evaluate the two-major-party-only concrete-support rule without overwriting v10."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "major_party_core_v11_experiment"
BASELINE_SUMMARY = ROOT / "outputs" / "active_presidential_nested_v10" / "summary.json"


def run() -> dict[str, object]:
    baseline = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    baseline_metrics = baseline["metrics"]
    original_output = active.OUTPUT_DIR
    active.OUTPUT_DIR = OUTPUT_DIR
    try:
        candidate = active.run()
    finally:
        active.OUTPUT_DIR = original_output

    candidate_metrics = candidate["metrics"]
    regional_delta = float(
        candidate_metrics["regional_equal_election_macro_mae_pp"]
        - baseline_metrics["regional_equal_election_macro_mae_pp"]
    )
    national_delta = float(
        candidate_metrics["national_equal_election_macro_mae_pp"]
        - baseline_metrics["national_equal_election_macro_mae_pp"]
    )
    decision = {
        "experiment": "major_party_core_only",
        "scope": "strict nested through-2022 development folds",
        "post_2022_outcomes_used": False,
        "baseline_policy": baseline["policy_version"],
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "regional_mae_delta_pp": regional_delta,
        "national_mae_delta_pp": national_delta,
        "aggregate_nonworsening": regional_delta <= 1e-12 and national_delta <= 1e-12,
        "rule": (
            "concrete support is estimated and assigned only to exact "
            "People Power and Democratic Party lineages; all other stable "
            "support remains critical or swing mass"
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
