"""Strict nested ablation for district-first candidate regional bases."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import build_speech_derived_candidate_context_v4 as context_builder  # noqa: E402
from scripts import evaluate_speech_derived_candidate_context_v2 as evaluator  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "district_reconstructed_candidate_base_v6_ablation"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v16"
DISTRICT_BASE = (
    ROOT / "outputs" / "district_reconstructed_candidate_base_v6" / "candidate_regional_base.csv"
)


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = context_builder.build_context(
        ROOT / "outputs" / "speech_derived_candidate_context_v4"
    )
    payload = evaluator._run(
        context,
        output_dir=OUTPUT_DIR / "district_first",
        role_aware=True,
        rejection_routing=False,
        candidate_regional_base_path=DISTRICT_BASE,
    )
    summary = pd.DataFrame(
        [
            {**_metrics(ACTIVE_DIR), "variant": "active_manual_v16"},
            {**payload["metrics"], "variant": "district_first_v6"},
        ]
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    frames = []
    for variant, path in [
        ("active_manual_v16", ACTIVE_DIR / "by_election.csv"),
        (
            "district_first_v6",
            OUTPUT_DIR / "district_first" / "active_run" / "by_election.csv",
        ),
    ]:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["variant"] = variant
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    decision = {
        "experiment": "district_reconstructed_candidate_base_v6",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "active_model_changed": False,
        "promotion_decision": "superseded_by_clean_v8",
        "isolated_candidate_base_ablation": False,
        "confounds": [
            "role_aware_slot_assignment",
            "speech_derived_candidate_context_v4",
        ],
        "promotion_reason": (
            "The full variant result remains reproducible, but it is not a clean "
            "candidate-regional-base-only comparison. Use "
            "district_candidate_base_clean_v8_ablation instead."
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
