"""Diagnostic cap ablation for the district-first 2017 rupture residual."""

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


OUTPUT_DIR = ROOT / "outputs" / "district_2017_regime_cap_v7_ablation"
DISTRICT_BASE = (
    ROOT
    / "outputs"
    / "district_reconstructed_candidate_base_v6"
    / "candidate_regional_base.csv"
)
BASELINE_DIR = (
    ROOT
    / "outputs"
    / "district_reconstructed_candidate_base_v6_ablation"
    / "district_first"
    / "active_run"
)
CAPS = (0.40, 0.45, 0.50)


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context = context_builder.build_context(
        ROOT / "outputs" / "speech_derived_candidate_context_v4"
    )
    original_load_policy = evaluator.active.load_policy
    runs: dict[float, Path] = {0.40: BASELINE_DIR}

    for cap in CAPS[1:]:
        run_root = OUTPUT_DIR / f"cap_{cap:.2f}"

        def load_policy_with_cap(path=evaluator.active.CONFIG_PATH, *, _cap=cap):
            policy = deepcopy(original_load_policy(path))
            policy["postprocess"]["contest_regime_log_shift_cap"] = float(_cap)
            return policy

        with evaluator.v1_evaluator.patched(
            [(evaluator.active, "load_policy", load_policy_with_cap)]
        ):
            evaluator._run(
                context,
                output_dir=run_root,
                role_aware=True,
                rejection_routing=False,
                candidate_regional_base_path=DISTRICT_BASE,
            )
        runs[cap] = run_root / "active_run"

    summary_rows: list[dict[str, object]] = []
    election_frames: list[pd.DataFrame] = []
    national_frames: list[pd.DataFrame] = []
    for cap, path in runs.items():
        summary_rows.append(
            {
                "contest_regime_log_shift_cap": cap,
                **_metrics(path),
            }
        )
        by_election = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
        by_election["contest_regime_log_shift_cap"] = cap
        election_frames.append(by_election)
        national = pd.read_csv(
            path / "national_predictions.csv", encoding="utf-8-sig"
        )
        national["contest_regime_log_shift_cap"] = cap
        national_frames.append(national)

    summary = pd.DataFrame(summary_rows).sort_values("contest_regime_log_shift_cap")
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
        "experiment": "district_2017_regime_cap_v7",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "active_model_changed": False,
        "diagnostic_only": True,
        "parameter_selection_is_outcome_aware": True,
        "mechanical_rationale": (
            "The previous 0.40 cap truncates the configured 0.50 expansion gain "
            "only when activation exceeds 0.80."
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print()
    print(
        by_election.loc[
            by_election["election_id"].eq("pres_2017"),
            [
                "contest_regime_log_shift_cap",
                "regional_weighted_mae_pp",
                "national_candidate_mae_pp",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
