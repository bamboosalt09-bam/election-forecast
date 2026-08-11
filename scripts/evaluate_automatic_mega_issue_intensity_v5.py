"""Compare manual, neutral, and speech-derived mega-issue intensity."""

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


OUTPUT_DIR = ROOT / "outputs" / "automatic_mega_issue_intensity_v5_ablation"
V2_BASELINE = ROOT / "outputs" / "speech_derived_candidate_context_v2" / "active_run"


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _by_election(run_dir: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "by_election.csv", encoding="utf-8-sig")
    frame.insert(0, "intensity_variant", variant)
    return frame


def _neutral_intensity(path: Path, automatic_path: Path) -> None:
    frame = pd.read_csv(automatic_path, encoding="utf-8-sig")
    frame["mega_issue_intensity"] = 1.0
    frame["notes"] = "neutral intensity ablation"
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = context_builder.build_context(
        ROOT / "outputs" / "speech_derived_candidate_context_v2"
    )
    intensity_manifest = intensity_builder.build(
        ROOT / "outputs" / "speech_derived_mega_intensity_v5"
    )
    automatic_path = Path(str(intensity_manifest["intensity_path"]))
    neutral_path = OUTPUT_DIR / "neutral_mega_issue_intensity.csv"
    _neutral_intensity(neutral_path, automatic_path)

    rows = [
        {
            "intensity_variant": "manual_intensity_v2_baseline",
            **_metrics(V2_BASELINE),
        }
    ]
    election_frames = [_by_election(V2_BASELINE, "manual_intensity_v2_baseline")]
    for variant, path in [
        ("neutral_intensity", neutral_path),
        ("automatic_speech_intensity", automatic_path),
    ]:
        variant_dir = OUTPUT_DIR / variant
        payload = evaluator._run(
            context,
            output_dir=variant_dir,
            role_aware=True,
            rejection_routing=True,
            mega_issue_intensity_path=path,
        )
        rows.append({"intensity_variant": variant, **payload["metrics"]})
        election_frames.append(_by_election(variant_dir / "active_run", variant))

    summary = pd.DataFrame(rows)
    by_election = pd.concat(election_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    comparison = by_election.pivot(
        index="election_id",
        columns="intensity_variant",
        values=["regional_weighted_mae_pp", "national_candidate_mae_pp"],
    )
    comparison.columns = [f"{variant}_{metric}" for metric, variant in comparison.columns]
    comparison = comparison.reset_index()
    for metric in ["regional_weighted_mae_pp", "national_candidate_mae_pp"]:
        comparison[f"automatic_minus_manual_{metric}"] = (
            comparison[f"automatic_speech_intensity_{metric}"]
            - comparison[f"manual_intensity_v2_baseline_{metric}"]
        )
    comparison.to_csv(
        OUTPUT_DIR / "comparison_by_election.csv", index=False, encoding="utf-8-sig"
    )
    decision = {
        "experiment": "automatic_mega_issue_intensity_v5",
        "scope": "strict nested pres_2002 through pres_2022",
        "active_model_changed": False,
        "post_2022_outcomes_used": False,
        "manual_intensity_read_by_automatic_variants": False,
        "new_layer_outcome_fields_used": [],
        "promotion_status": "not_promoted_speech_intensity_underfits_2017",
        "smooth_activation_status": "accepted_backward_compatible_consumer_fix",
        "diagnosis": (
            "The original hard gate made intensities just above 1.0 activate a "
            "full direct shift. Continuous excess activation removes that cliff, "
            "but speech-only intensity still underestimates the 2017 institutional "
            "crisis and overactivates 2007 and 2022 relative to the manual baseline."
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
