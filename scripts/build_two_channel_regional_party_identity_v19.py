"""Build lineage-aware party-preference and organization identity events."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.regional_party_channels import (  # noqa: E402
    CHANNEL_TYPE_WEIGHTS,
    build_two_channel_identity_events,
)


OUTPUT_DIR = ROOT / "outputs" / "two_channel_regional_party_identity_v19"


def main() -> None:
    history = pd.read_csv(
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "bloc_history_results.csv",
        encoding="utf-8-sig",
    )
    assembly = pd.read_csv(
        ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv",
        encoding="utf-8-sig",
    )
    events = build_two_channel_identity_events(history, assembly)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(
        OUTPUT_DIR / "identity_events.csv", index=False, encoding="utf-8-sig"
    )
    channel_summary = (
        events.groupby(["evidence_channel", "lineage_specific"], as_index=False)
        .agg(
            rows=("election_id", "size"),
            elections=("election_id", "nunique"),
            mean_identity_excess=("identity_excess", "mean"),
            maximum_identity_excess=("identity_excess", "max"),
        )
    )
    channel_summary.to_csv(
        OUTPUT_DIR / "channel_summary.csv", index=False, encoding="utf-8-sig"
    )
    lineage_elections = (
        events.loc[events["lineage_specific"]]
        .groupby(["election_id", "election_type", "evidence_channel"], as_index=False)
        .agg(rows=("region_id", "size"), maximum_share=("identity_share", "max"))
    )
    lineage_elections.to_csv(
        OUTPUT_DIR / "lineage_elections.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema": "two_channel_regional_party_identity_v19",
        "history_rows": int(len(history)),
        "assembly_candidate_rows": int(len(assembly)),
        "event_rows": int(len(events)),
        "lineage_specific_rows": int(events["lineage_specific"].sum()),
        "type_weights": dict(CHANNEL_TYPE_WEIGHTS),
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(lineage_elections.to_string(index=False))


if __name__ == "__main__":
    main()

