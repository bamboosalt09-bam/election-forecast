"""Build an ablation profile that automates viability only."""

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
    merge_automatic_viability,
)


OUTPUT_DIR = ROOT / "outputs" / "election_derived_third_candidate_profile_v14b"


def main() -> None:
    base = pd.read_csv(
        ROOT / "data" / "raw" / "third_candidate_profile.csv",
        encoding="utf-8-sig",
    )
    automatic = pd.read_csv(
        ROOT
        / "outputs"
        / "election_derived_third_candidate_profile_v14"
        / "third_candidate_profile.csv",
        encoding="utf-8-sig",
    )
    profile, audit = merge_automatic_viability(base, automatic)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(
        OUTPUT_DIR / "third_candidate_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(OUTPUT_DIR / "audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "schema": "election_derived_third_candidate_profile_v14b",
        "rows": int(len(profile)),
        "automatic_viability_rows": int(len(audit)),
        "manual_character_traits_retained": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
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
