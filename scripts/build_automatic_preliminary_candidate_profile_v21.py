"""Build automatic prior-person profiles for dated withdrawn candidates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.preliminary_candidate_registry import (  # noqa: E402
    build_preliminary_candidate_registry,
    derive_prior_candidate_profile,
    merge_preliminary_profile,
)


OUTPUT_DIR = ROOT / "outputs" / "automatic_preliminary_candidate_profile_v21"
ACTIVE_PROFILE = (
    ROOT / "outputs" / "automatic_third_character_v20b" / "third_candidate_profile.csv"
)


def main() -> None:
    coalition = pd.read_csv(
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "coalition_events.csv",
        encoding="utf-8-sig",
    )
    transfers = pd.read_csv(
        ROOT / "data" / "raw" / "withdrawn_candidate_transfers.csv",
        encoding="utf-8-sig",
    )
    landscape = pd.read_csv(
        ROOT / "data" / "raw" / "candidate_political_landscape.csv",
        encoding="utf-8-sig",
    )
    active_profile = pd.read_csv(ACTIVE_PROFILE, encoding="utf-8-sig")
    results = pd.read_csv(
        ROOT
        / "presidential_issue_engine"
        / "fixed_dataset"
        / "presidential_results_standardized.csv",
        encoding="utf-8-sig",
    )
    registry = build_preliminary_candidate_registry(coalition, transfers, landscape)
    automatic, audit = derive_prior_candidate_profile(registry, active_profile, results)
    merged, merge_audit = merge_preliminary_profile(active_profile, automatic)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry.to_csv(
        OUTPUT_DIR / "preliminary_candidate_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    automatic.to_csv(
        OUTPUT_DIR / "automatic_preliminary_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    merged.to_csv(
        OUTPUT_DIR / "third_candidate_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        OUTPUT_DIR / "derivation_audit.csv", index=False, encoding="utf-8-sig"
    )
    merge_audit.to_csv(
        OUTPUT_DIR / "merge_audit.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema": "automatic_preliminary_candidate_profile_v21",
        "registry_rows": int(len(registry)),
        "automatic_rows": int(len(automatic)),
        "merged_rows": int(len(merged)),
        "unmatched_registry_rows": int(len(registry) - len(automatic)),
        "prior_person_evidence_required": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(audit.to_string(index=False))
    print(merge_audit.to_string(index=False))


if __name__ == "__main__":
    main()

