"""Build prior-only regional-party candidate alignment for the v10 successor."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.automatic_regional_party_alignment import (  # noqa: E402
    build_automatic_nonmajor_alignment,
    build_full_history_identity_events,
)


OUTPUT_DIR = ROOT / "outputs" / "automatic_regional_party_alignment_v11"


def main() -> None:
    history = pd.read_csv(
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "bloc_history_results.csv",
        encoding="utf-8-sig",
    )
    context = pd.read_csv(
        ROOT / "data" / "raw" / "candidate_party_speech_context.csv",
        encoding="utf-8-sig",
    )
    landscape = pd.read_csv(
        ROOT / "data" / "raw" / "candidate_political_landscape.csv",
        encoding="utf-8-sig",
    )
    bloc_landscape = pd.read_csv(
        ROOT / "data" / "raw" / "assembly15_bloc_political_landscape.csv",
        encoding="utf-8-sig",
    )
    manual = pd.read_csv(
        ROOT / "data" / "raw" / "chungcheong_identity_alignment.csv",
        encoding="utf-8-sig",
    )
    automatic, audit = build_automatic_nonmajor_alignment(
        history, context, landscape, bloc_landscape
    )
    combined = pd.concat([manual, automatic], ignore_index=True)
    combined = combined.sort_values(
        ["election_id", "available_date", "candidate_name"]
    ).drop_duplicates(
        ["election_id", "region_scope", "candidate_name", "evidence_type"],
        keep="last",
    )
    events = build_full_history_identity_events(history)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    automatic.to_csv(
        OUTPUT_DIR / "automatic_alignment.csv", index=False, encoding="utf-8-sig"
    )
    combined.to_csv(
        OUTPUT_DIR / "manual_plus_automatic_alignment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(OUTPUT_DIR / "candidate_audit.csv", index=False, encoding="utf-8-sig")
    events.to_csv(
        OUTPUT_DIR / "full_history_identity_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "schema": "automatic_regional_party_alignment_v11",
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "history_rows": int(len(history)),
        "full_history_event_rows": int(len(events)),
        "automatic_alignment_rows": int(len(automatic)),
        "combined_alignment_rows": int(len(combined)),
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(automatic.to_string(index=False))


if __name__ == "__main__":
    main()
