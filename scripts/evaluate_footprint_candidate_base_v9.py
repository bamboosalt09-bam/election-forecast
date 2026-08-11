"""Clean strict-nested ablation for the footprint-controlled candidate base."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "footprint_candidate_base_v9_ablation"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v16"
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)
VARIANTS = {
    "footprint_exact_0.40": {
        "response": None,
        "rejection_routing": False,
    },
    "footprint_balanced_0.60_routed": {
        "response": {
            "contest_regime_expansion_gain": 0.60,
            "contest_regime_log_shift_cap": 0.60,
            "contest_regime_swing_log_shift_cap": 0.75,
        },
        "rejection_routing": True,
    },
}


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {
        label: clean._run_variant(
            label,
            config["response"],
            rejection_routing=bool(config["rejection_routing"]),
            candidate_base_path=FOOTPRINT_BASE,
            output_root=OUTPUT_DIR,
        )
        for label, config in VARIANTS.items()
    }
    summary_rows = [{"variant_label": "active_v16", **_metrics(ACTIVE_DIR)}]
    election_frames: list[pd.DataFrame] = []
    national_frames: list[pd.DataFrame] = []
    for label, path in [("active_v16", ACTIVE_DIR), *runs.items()]:
        if label != "active_v16":
            summary_rows.append({"variant_label": label, **_metrics(path)})
        by_election = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
        by_election["variant_label"] = label
        election_frames.append(by_election)
        national = pd.read_csv(path / "national_predictions.csv", encoding="utf-8-sig")
        national["variant_label"] = label
        national_frames.append(national)

    summary = pd.DataFrame(summary_rows)
    by_election = pd.concat(election_frames, ignore_index=True)
    national = pd.concat(national_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    national.to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    decision = {
        "experiment": "footprint_candidate_base_v9",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "active_model_changed": False,
        "candidate_base_is_fully_automatic": True,
        "balanced_0_60_parameter_selection_is_outcome_aware": True,
        "promotion_decision": "diagnostic_only",
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print()
    print(
        by_election.sort_values(["election_id", "variant_label"])[
            [
                "variant_label",
                "election_id",
                "regional_weighted_mae_pp",
                "national_candidate_mae_pp",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
