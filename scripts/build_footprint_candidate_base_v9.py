"""Build a footprint-controlled official candidate regional base through 2022."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.district_reconstructed_candidate_base import (  # noqa: E402
    build_district_reconstructed_candidate_base,
)


OFFICIAL_DIR = ROOT / "data" / "raw" / "official_sources"
OUTPUT_DIR = ROOT / "outputs" / "footprint_candidate_base_v9"


def main() -> None:
    history = pd.read_csv(
        OFFICIAL_DIR / "nec_candidate_history.csv", encoding="utf-8-sig"
    )
    district = pd.read_csv(
        OFFICIAL_DIR / "nec_assembly_district_history.csv", encoding="utf-8-sig"
    )
    context = pd.read_csv(
        ROOT / "data" / "raw" / "candidate_party_speech_context.csv",
        encoding="utf-8-sig",
    )
    regional, components = build_district_reconstructed_candidate_base(
        history,
        district,
        context,
        footprint_controlled=True,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regional.to_csv(
        OUTPUT_DIR / "candidate_regional_base.csv", index=False, encoding="utf-8-sig"
    )
    components.to_csv(
        OUTPUT_DIR / "component_audit.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema": "footprint_candidate_base_build_v9",
        "post_2022_outcomes_used": False,
        "target_presidential_outcome_fields_used": [],
        "footprint_controlled": True,
        "lost_executive_candidacies_create_office_base": False,
        "repeat_combination": "bounded_union",
        "district_history_rows": len(district),
        "candidate_history_rows": len(history),
        "component_rows": len(components),
        "regional_base_rows": len(regional),
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
