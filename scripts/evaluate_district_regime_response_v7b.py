"""Sensitivity check for stronger evidence-gated contest-regime response."""

from __future__ import annotations

from copy import deepcopy
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


OUTPUT_DIR = ROOT / "outputs" / "district_regime_response_v7b_ablation"
DISTRICT_BASE = (
    ROOT
    / "outputs"
    / "district_reconstructed_candidate_base_v6"
    / "candidate_regional_base.csv"
)
EXISTING_RUNS = {
    "baseline_0.40": (
        ROOT
        / "outputs"
        / "district_reconstructed_candidate_base_v6_ablation"
        / "district_first"
        / "active_run"
    ),
    "cap_only_0.50": (
        ROOT
        / "outputs"
        / "district_2017_regime_cap_v7_ablation"
        / "cap_0.50"
        / "active_run"
    ),
}
RESPONSE_CONFIGS = {
    "balanced_0.60": {
        "contest_regime_expansion_gain": 0.60,
        "contest_regime_log_shift_cap": 0.60,
        "contest_regime_swing_log_shift_cap": 0.75,
    },
    "balanced_0.70": {
        "contest_regime_expansion_gain": 0.70,
        "contest_regime_log_shift_cap": 0.70,
        "contest_regime_swing_log_shift_cap": 0.875,
    },
}


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = context_builder.build_context(
        ROOT / "outputs" / "speech_derived_candidate_context_v4"
    )
    original_load_policy = evaluator.active.load_policy
    runs = dict(EXISTING_RUNS)

    for label, config in RESPONSE_CONFIGS.items():
        run_root = OUTPUT_DIR / label

        def load_policy_with_response(
            path=evaluator.active.CONFIG_PATH,
            *,
            _config=config,
        ):
            policy = deepcopy(original_load_policy(path))
            policy["postprocess"].update(_config)
            return policy

        with evaluator.v1_evaluator.patched(
            [(evaluator.active, "load_policy", load_policy_with_response)]
        ):
            evaluator._run(
                context,
                output_dir=run_root,
                role_aware=True,
                rejection_routing=False,
                candidate_regional_base_path=DISTRICT_BASE,
            )
        runs[label] = run_root / "active_run"

    summary_rows: list[dict[str, object]] = []
    election_frames: list[pd.DataFrame] = []
    national_frames: list[pd.DataFrame] = []
    for label, path in runs.items():
        summary_rows.append({"response_variant": label, **_metrics(path)})
        by_election = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
        by_election["response_variant"] = label
        election_frames.append(by_election)
        national = pd.read_csv(
            path / "national_predictions.csv", encoding="utf-8-sig"
        )
        national["response_variant"] = label
        national_frames.append(national)

    summary = pd.DataFrame(summary_rows)
    by_election = pd.concat(election_frames, ignore_index=True)
    national = pd.concat(national_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    national.to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    decision = {
        "experiment": "district_regime_response_v7b",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "active_model_changed": False,
        "diagnostic_only": True,
        "parameter_selection_is_outcome_aware": True,
        "promotion_decision": "pending_structural_review",
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print()
    print(
        by_election.loc[
            by_election["election_id"].isin(["pres_2007", "pres_2017"]),
            [
                "response_variant",
                "election_id",
                "regional_weighted_mae_pp",
                "national_candidate_mae_pp",
            ],
        ].sort_values(["election_id", "response_variant"]).to_string(index=False)
    )


if __name__ == "__main__":
    main()
