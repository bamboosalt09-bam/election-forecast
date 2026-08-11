"""Evaluate the dated event-class gate on speech-derived shock intensity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import build_speech_derived_candidate_context_v2 as context_builder  # noqa: E402
from scripts import build_speech_derived_mega_intensity_v5 as intensity_builder  # noqa: E402
from scripts import evaluate_speech_derived_candidate_context_v2 as evaluator  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_mega_issue_intensity_v5b_event_gate"
V2_BASELINE = ROOT / "outputs" / "speech_derived_candidate_context_v2" / "active_run"
PURE_RUN = (
    ROOT
    / "outputs"
    / "automatic_mega_issue_intensity_v5_ablation"
    / "automatic_speech_intensity"
    / "active_run"
)


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _by_election(path: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
    frame.insert(0, "intensity_variant", variant)
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = context_builder.build_context(
        ROOT / "outputs" / "speech_derived_candidate_context_v2"
    )
    manifest = intensity_builder.build(
        ROOT / "outputs" / "speech_derived_mega_intensity_v5"
    )
    run_dir = OUTPUT_DIR / "event_class_gate"
    payload = evaluator._run(
        context,
        output_dir=run_dir,
        role_aware=True,
        rejection_routing=True,
        mega_issue_intensity_path=Path(str(manifest["event_class_intensity_path"])),
    )
    rows = pd.DataFrame(
        [
            {"intensity_variant": "manual_intensity_v2_baseline", **_metrics(V2_BASELINE)},
            {"intensity_variant": "automatic_speech_only", **_metrics(PURE_RUN)},
            {"intensity_variant": "automatic_event_class_gate", **payload["metrics"]},
        ]
    )
    elections = pd.concat(
        [
            _by_election(V2_BASELINE, "manual_intensity_v2_baseline"),
            _by_election(PURE_RUN, "automatic_speech_only"),
            _by_election(run_dir / "active_run", "automatic_event_class_gate"),
        ],
        ignore_index=True,
    )
    rows.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    elections.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    decision = {
        "experiment": "automatic_mega_issue_intensity_v5b_event_gate",
        "scope": "strict nested pres_2002 through pres_2022",
        "active_model_changed": False,
        "post_2022_outcomes_used": False,
        "manual_intensity_read_by_automatic_variant": False,
        "taxonomy_numeric_fields_read": [],
        "promotion_status": "not_promoted_remaining_2017_understrength",
        "component_status": "validated_as_experimental_event_class_gate",
        "diagnosis": (
            "Using dated shock_type only, with all taxonomy numeric fields ignored, "
            "substantially improves the speech-only compiler but remains worse than "
            "the manual intensity baseline. Further adjustment on the same five "
            "outcomes would be outcome-aware tuning."
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
