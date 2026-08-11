"""Build source-lane pressure from the fully election-derived third profile."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.speech_derived_third_pressure import (  # noqa: E402
    build_automatic_third_candidate_pressure,
)
from scripts import build_speech_derived_issue_context as issue_builder  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "election_derived_third_pressure_v16"


def main() -> None:
    pressure = build_automatic_third_candidate_pressure(
        pd.read_csv(
            ROOT
            / "outputs"
            / "election_derived_third_candidate_profile_v15"
            / "third_candidate_profile.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT / "data" / "raw" / "candidate_party_speech_context.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT / "data" / "raw" / "candidate_political_landscape.csv",
            encoding="utf-8-sig",
        ),
        issue_builder.ELECTION_DATES,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pressure.to_csv(
        OUTPUT_DIR / "third_candidate_pressure.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "schema": "election_derived_third_pressure_v16",
        "rows": int(len(pressure)),
        "profile_source": (
            "outputs/election_derived_third_candidate_profile_v15/"
            "third_candidate_profile.csv"
        ),
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "manual_pressure_read_count": 0,
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pressure.to_string(index=False))


if __name__ == "__main__":
    main()
