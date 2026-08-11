"""Build an outcome-free election shock intensity from Assembly issue matches."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine.speech_derived_mega_intensity import (  # noqa: E402
    SCHEMA_VERSION,
    build_automatic_mega_issue_intensity,
    gate_intensity_by_event_class,
)
from scripts import run_active_presidential_model as active  # noqa: E402


SOURCE = (
    ROOT
    / "archives"
    / "experiments"
    / "manual_seed_lineage_v17_rejected_20260728"
    / "artifacts"
    / "assembly_speaker_issue_matches_15_22.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "speech_derived_mega_intensity_v5"
TAXONOMY = ROOT / "data" / "raw" / "mega_issue_taxonomy.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(
        SOURCE,
        usecols=[
            "election_id",
            "period",
            "speaker",
            "issue_name",
            "issue_weight",
            "matched_term_count",
        ],
        encoding="utf-8-sig",
    )
    intensity, diagnostics = build_automatic_mega_issue_intensity(
        matches, active.nested.engine.ELECTION_DATES
    )
    taxonomy = pd.read_csv(
        TAXONOMY,
        usecols=["election_id", "shock_type", "available_date"],
        encoding="utf-8-sig",
    )
    gated_intensity, gated_diagnostics = gate_intensity_by_event_class(
        diagnostics, taxonomy, active.nested.engine.ELECTION_DATES
    )
    intensity_path = output_dir / "mega_issue_intensity.csv"
    diagnostics_path = output_dir / "mega_issue_intensity_diagnostics.csv"
    gated_path = output_dir / "mega_issue_intensity_event_class.csv"
    gated_diagnostics_path = output_dir / "mega_issue_intensity_event_class_diagnostics.csv"
    intensity.to_csv(intensity_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    gated_intensity.to_csv(gated_path, index=False, encoding="utf-8-sig")
    gated_diagnostics.to_csv(
        gated_diagnostics_path, index=False, encoding="utf-8-sig"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "post_2022_outcomes_used": False,
        "outcome_fields_used": [],
        "manual_mega_issue_intensity_read": False,
        "taxonomy_numeric_fields_read": [],
        "taxonomy_categorical_fields_read": [
            "election_id",
            "shock_type",
            "available_date",
        ],
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": _sha256(SOURCE),
        "source_rows": int(len(matches)),
        "output_rows": int(len(intensity)),
        "intensity_path": str(intensity_path),
        "diagnostics_path": str(diagnostics_path),
        "event_class_intensity_path": str(gated_path),
        "event_class_diagnostics_path": str(gated_diagnostics_path),
    }
    (output_dir / "lineage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
