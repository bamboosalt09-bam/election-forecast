"""Build V18-preserving regional identity with party-lineage corroboration."""

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
    build_lineage_corroborated_identity_events,
)


OUTPUT_DIR = ROOT / "outputs" / "lineage_corroborated_identity_v19b"
CORROBORATION_GAIN = 0.25


def main() -> None:
    history = pd.read_csv(
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "bloc_history_results.csv",
        encoding="utf-8-sig",
    )
    assembly = pd.read_csv(
        ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv",
        encoding="utf-8-sig",
    )
    events = build_lineage_corroborated_identity_events(
        history, assembly, corroboration_gain=CORROBORATION_GAIN
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(
        OUTPUT_DIR / "identity_events.csv", index=False, encoding="utf-8-sig"
    )
    audit = events.loc[
        events["lineage_purity"].gt(0.0),
        [
            "election_id",
            "election_type",
            "region_id",
            "evidence_channel",
            "identity_share",
            "lineage_named_share",
            "lineage_purity",
            "base_type_weight",
            "type_weight",
        ],
    ].copy()
    audit.to_csv(
        OUTPUT_DIR / "corroboration_audit.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema": "lineage_corroborated_identity_v19b",
        "event_rows": int(len(events)),
        "corroborated_rows": int(events["lineage_purity"].gt(0.0).sum()),
        "corroboration_gain": CORROBORATION_GAIN,
        "base_identity_excess_preserved": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

