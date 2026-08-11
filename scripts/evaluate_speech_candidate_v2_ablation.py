"""Factorial ablation for manual/automatic third profiles and role routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import evaluate_speech_derived_candidate_context_v2 as evaluator  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "speech_candidate_v2_ablation"
ISSUE_V1_DIR = ROOT / "outputs" / "speech_derived_issue_context_v1"
CANDIDATE_V2_DIR = ROOT / "outputs" / "speech_derived_candidate_context_v2"


def _context(base: Path, *, automatic_third: bool) -> dict[str, object]:
    return {
        "speech": base / "candidate_party_speech_context.csv",
        "tone": base / "candidate_party_tone_gap.csv",
        "treatment": base / "candidate_public_treatment.csv",
        "conversion": base / "candidate_vote_conversion_context.csv",
        "profile": base / "auto_issue_seed" / "candidate_issue_profile.csv",
        "axis": base / "auto_issue_seed" / "mega_issue_axis.csv",
        "attribution": base / "auto_issue_seed" / "mega_issue_attribution.csv",
        "third_profile": (
            base / "auto_candidate_role" / "third_candidate_profile.csv"
            if automatic_third
            else ROOT / "data" / "raw" / "third_candidate_profile.csv"
        ),
    }


def main() -> None:
    variants = [
        (
            "manual_profile_rank",
            _context(ISSUE_V1_DIR, automatic_third=False),
            False,
            False,
        ),
        (
            "manual_profile_role",
            _context(ISSUE_V1_DIR, automatic_third=False),
            True,
            False,
        ),
        (
            "auto_profile_rank",
            _context(CANDIDATE_V2_DIR, automatic_third=True),
            False,
            False,
        ),
        (
            "auto_profile_role",
            _context(CANDIDATE_V2_DIR, automatic_third=True),
            True,
            False,
        ),
        (
            "auto_profile_role_routed",
            _context(CANDIDATE_V2_DIR, automatic_third=True),
            True,
            True,
        ),
    ]
    summary_rows: list[dict[str, object]] = []
    election_rows: list[pd.DataFrame] = []
    for name, context, role_aware, routing in variants:
        destination = OUTPUT_DIR / name
        payload = evaluator._run(
            context,
            output_dir=destination,
            role_aware=role_aware,
            rejection_routing=routing,
        )
        summary_rows.append(
            {
                "ablation_variant": name,
                "manual_third_profile": not "auto_profile" in name,
                "role_aware": role_aware,
                "rejection_routing": routing,
                **payload["metrics"],
            }
        )
        by_election = pd.read_csv(destination / "active_run" / "by_election.csv")
        by_election.insert(0, "ablation_variant", name)
        election_rows.append(by_election)

    summary = pd.DataFrame(summary_rows)
    by_election = pd.concat(election_rows, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig"
    )
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    decision = {
        "scope": "strict nested pres_2002 through pres_2022",
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used_by_new_layers": [],
        "variants": summary.to_dict("records"),
        "status": "diagnostic_not_promoted",
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
