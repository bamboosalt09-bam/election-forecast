"""Strict nested ablation of party-lineage reliability corroboration."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.regional_party_channels import (  # noqa: E402
    build_lineage_corroborated_identity_events,
)
from scripts import evaluate_two_channel_regional_party_identity_v19 as evaluation  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "lineage_corroborated_identity_v19b_ablation"
CORROBORATION_GAIN = 0.25
ASSEMBLY = pd.read_csv(
    ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv",
    encoding="utf-8-sig",
)


def corroborated_events(history: pd.DataFrame) -> pd.DataFrame:
    return build_lineage_corroborated_identity_events(
        history, ASSEMBLY, corroboration_gain=CORROBORATION_GAIN
    )


def main() -> None:
    evaluation.main(
        output_dir=OUTPUT_DIR,
        event_builder=corroborated_events,
        experiment_name="lineage_corroborated_identity_v19b",
        variant_label="lineage_corroboration_v19b",
        decision_metadata={
            "base_identity_excess_preserved": True,
            "lineage_used_as_reliability_only": True,
            "corroboration_gain": CORROBORATION_GAIN,
            "generic_third_rows_reliability_reduced": False,
        },
    )


if __name__ == "__main__":
    main()

