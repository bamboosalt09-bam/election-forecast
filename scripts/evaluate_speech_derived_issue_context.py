"""Evaluate the speech-derived issue context without changing the active model."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import build_speech_derived_issue_context as context_builder  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "speech_derived_issue_context_v1"
BASELINE_DIR = ROOT / "outputs" / "active_presidential_nested_v16"


@contextmanager
def patched(attributes: list[tuple[object, str, object]]) -> Iterator[None]:
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in attributes]
    for owner, name, value in attributes:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, value in reversed(previous):
            setattr(owner, name, value)


def _run(context: dict[str, object]) -> dict[str, object]:
    assignment_dir = OUTPUT_DIR / "preliminary_slot_assignment"
    run_dir = OUTPUT_DIR / "active_run"
    engines = {active.nested.engine, active.assignment_builder.engine}
    attributes: list[tuple[object, str, object]] = [
        (active, "ASSIGNMENT_DIR", assignment_dir),
        (
            active.nested,
            "ASSIGNMENT_PATH",
            assignment_dir / "candidate_slot_assignments_v2.csv",
        ),
        (active.nested.base_eval, "STANCE_PATH", context["tone"]),
        (active, "CANDIDATE_ISSUE_PROFILE", context["profile"]),
        (active, "CONVERSION_CONTEXT", context["conversion"]),
        (active, "regenerate_issue_seeds", lambda: None),
    ]
    for engine in engines:
        attributes.extend(
            [
                (engine, "CANDIDATE_PARTY_SPEECH_CONTEXT", str(context["speech"])),
                (engine, "CANDIDATE_PARTY_TONE_GAP", str(context["tone"])),
                (engine, "CANDIDATE_PUBLIC_TREATMENT", str(context["treatment"])),
                (engine, "CANDIDATE_VOTE_CONVERSION_CONTEXT", str(context["conversion"])),
                (engine, "AUTO_CANDIDATE_ISSUE_PROFILE", str(context["profile"])),
                (engine, "AUTO_MEGA_ISSUE_AXIS", str(context["axis"])),
                (engine, "AUTO_MEGA_ISSUE_ATTRIBUTION", str(context["attribution"])),
            ]
        )
    with patched(attributes):
        return active.run(output_dir=run_dir)


def _comparison(run_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    baseline = pd.read_csv(BASELINE_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    experiment = pd.read_csv(run_dir / "nested_predictions.csv", encoding="utf-8-sig")
    name = "candidate_name_x" if "candidate_name_x" in baseline else "candidate_name"
    keys = ["election_id", "region_id", name]
    compared = baseline[keys + ["layer_pred"]].merge(
        experiment[keys + ["layer_pred"]],
        on=keys,
        how="inner",
        suffixes=("_baseline", "_speech"),
    )
    compared["difference_pp"] = (
        compared["layer_pred_speech"] - compared["layer_pred_baseline"]
    ) * 100.0
    compared["abs_difference_pp"] = compared["difference_pp"].abs()
    by_election = (
        compared.groupby("election_id", as_index=False)
        .agg(
            rows=("abs_difference_pp", "size"),
            mean_abs_difference_pp=("abs_difference_pp", "mean"),
            max_abs_difference_pp=("abs_difference_pp", "max"),
        )
        .sort_values("election_id")
    )
    summary = {
        "matched_rows": int(len(compared)),
        "changed_rows": int(compared["abs_difference_pp"].gt(1e-10).sum()),
        "mean_abs_difference_pp": float(compared["abs_difference_pp"].mean()),
        "max_abs_difference_pp": float(compared["abs_difference_pp"].max()),
    }
    return by_election, summary


def main() -> None:
    context = context_builder.build_context(OUTPUT_DIR)
    payload = _run(context)
    run_dir = OUTPUT_DIR / "active_run"
    by_election, comparison = _comparison(run_dir)
    baseline = json.loads(
        (BASELINE_DIR / "summary.json").read_text(encoding="utf-8")
    )["metrics"]
    experiment = payload["metrics"]
    decision = {
        "experiment": "speech_derived_issue_context_v1",
        "active_model_changed": False,
        "manual_seed_ancestry_allowed": False,
        "post_2022_outcomes_used": False,
        "baseline_v16_metrics": baseline,
        "speech_derived_metrics": experiment,
        "prediction_comparison": comparison,
        "promotion_status": "not_promoted_pending_review",
    }
    by_election.to_csv(
        OUTPUT_DIR / "prediction_differences_by_election.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
