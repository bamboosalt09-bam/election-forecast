"""Evaluate automatic political roles and third-candidate stature in isolation."""

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
from scripts import evaluate_speech_derived_issue_context as v1_evaluator  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "speech_derived_candidate_context_v2"
ACTIVE_BASELINE_DIR = ROOT / "outputs" / "active_presidential_nested_v16"
SPEECH_V1_DIR = ROOT / "outputs" / "speech_derived_issue_context_v1" / "active_run"


def _write_assignments(assignment_dir: Path, *, role_aware: bool) -> None:
    assignments, audit, summary = active.assignment_builder.build(
        role_aware=role_aware
    )
    assignment_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        assignment_dir / "candidate_slot_assignments_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        assignment_dir / "fold_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (assignment_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run(
    context: dict[str, object],
    *,
    output_dir: Path = OUTPUT_DIR,
    role_aware: bool = True,
    rejection_routing: bool = True,
    third_pressure_path: Path | None = None,
    candidate_regional_base_path: Path | None = None,
    mega_issue_intensity_path: Path | None = None,
) -> dict[str, object]:
    assignment_dir = output_dir / "preliminary_slot_assignment"
    run_dir = output_dir / "active_run"
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
        (
            active,
            "regenerate_assignments",
            lambda: _write_assignments(assignment_dir, role_aware=role_aware),
        ),
    ]
    if candidate_regional_base_path is not None:
        attributes.append(
            (active, "CANDIDATE_REGIONAL_BASE", candidate_regional_base_path)
        )
    if mega_issue_intensity_path is not None:
        attributes.append(
            (active, "MEGA_ISSUE_INTENSITY", mega_issue_intensity_path)
        )
    for engine in engines:
        attributes.extend(
            [
                (engine, "CANDIDATE_PARTY_SPEECH_CONTEXT", str(context["speech"])),
                (engine, "CANDIDATE_PARTY_TONE_GAP", str(context["tone"])),
                (engine, "CANDIDATE_PUBLIC_TREATMENT", str(context["treatment"])),
                (
                    engine,
                    "CANDIDATE_VOTE_CONVERSION_CONTEXT",
                    str(context["conversion"]),
                ),
                (engine, "THIRD_CANDIDATE_PROFILE", str(context["third_profile"])),
                (engine, "AUTO_CANDIDATE_ISSUE_PROFILE", str(context["profile"])),
                (engine, "AUTO_MEGA_ISSUE_AXIS", str(context["axis"])),
                (engine, "AUTO_MEGA_ISSUE_ATTRIBUTION", str(context["attribution"])),
            ]
        )
        if third_pressure_path is not None:
            attributes.append(
                (engine, "THIRD_CANDIDATE_PRESSURE", str(third_pressure_path))
            )
        if candidate_regional_base_path is not None:
            attributes.append(
                (
                    engine,
                    "CANDIDATE_REGIONAL_BASE",
                    str(candidate_regional_base_path),
                )
            )
        if mega_issue_intensity_path is not None:
            attributes.append(
                (
                    engine,
                    "ENHANCED_MEGA_ISSUE_INTENSITY",
                    str(mega_issue_intensity_path),
                )
            )
    with v1_evaluator.patched(attributes):
        payload = active.run(
            output_dir=run_dir,
            rejection_beneficiary_routing_enabled=rejection_routing,
        )
    payload["status"] = "diagnostic_candidate"
    payload["active_model_changed"] = False
    (run_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    context = context_builder.build_context(OUTPUT_DIR)
    payload = _run(context)
    assignments = pd.read_csv(
        OUTPUT_DIR / "preliminary_slot_assignment" / "candidate_slot_assignments_v2.csv",
        encoding="utf-8-sig",
    )
    role_audit_columns = [
        "election_id",
        "source_slot",
        "candidate_name",
        "preliminary_mean_share",
        "preliminary_rank",
        "rank_slot",
        "assigned_slot",
        "political_role",
        "major_party_core_eligible",
        "automatic_third_viability",
        "role_assignment_applied",
        "role_assignment_reason",
    ]
    assignments[role_audit_columns].to_csv(
        OUTPUT_DIR / "candidate_role_assignment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decision = {
        "experiment": "speech_derived_candidate_context_v2",
        "active_model_changed": False,
        "post_2022_outcomes_used": False,
        "manual_issue_seed_ancestry_allowed": False,
        "manual_third_candidate_profile_allowed": False,
        "role_aware_assignment": True,
        "active_v16_metrics": _metrics(ACTIVE_BASELINE_DIR),
        "speech_issue_v1_metrics": _metrics(SPEECH_V1_DIR),
        "speech_candidate_v2_metrics": payload["metrics"],
        "promotion_status": "not_promoted_after_strict_ablation",
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
