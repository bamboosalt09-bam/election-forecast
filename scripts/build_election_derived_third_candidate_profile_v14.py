"""Build the separated-evidence third-candidate profile for the v14 ablation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.election_derived_third_candidate_profile_v2 import (  # noqa: E402
    build_election_derived_third_profile_v2,
)


OUTPUT_DIR = ROOT / "outputs" / "election_derived_third_candidate_profile_v14"


def main() -> None:
    profile, audit = build_election_derived_third_profile_v2(
        pd.read_csv(
            ROOT
            / "outputs"
            / "speech_derived_candidate_context_v2"
            / "auto_candidate_role"
            / "third_candidate_profile.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT / "data" / "raw" / "candidate_party_speech_context.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT / "data" / "raw" / "official_sources" / "nec_candidate_history.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT
            / "presidential_issue_engine"
            / "fixed_dataset"
            / "presidential_results_standardized.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT
            / "presidential_issue_engine"
            / "fixed_dataset"
            / "bloc_history_results.csv",
            encoding="utf-8-sig",
        ),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(
        OUTPUT_DIR / "third_candidate_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(OUTPUT_DIR / "audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "schema": "election_derived_third_candidate_profile_v14",
        "rows": int(len(profile)),
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "direct_party_and_candidate_ballots_separated": True,
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
