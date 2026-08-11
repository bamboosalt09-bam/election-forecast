"""Build candidate context v3 with automatic third-candidate lane pressure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.speech_derived_third_pressure import (  # noqa: E402
    SCHEMA_VERSION as PRESSURE_SCHEMA_VERSION,
    build_automatic_third_candidate_pressure,
)
from scripts import build_speech_derived_candidate_context_v2 as v2_builder  # noqa: E402
from scripts import build_speech_derived_issue_context as issue_builder  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "speech_derived_candidate_context_v3"
POLITICAL_LANDSCAPE = ROOT / "data" / "raw" / "candidate_political_landscape.csv"
MANUAL_PRESSURE = (ROOT / "data" / "raw" / "third_candidate_pressure.csv").resolve()


def build_context(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    context = v2_builder.build_context(output_dir)
    profile = pd.read_csv(context["third_profile"], encoding="utf-8-sig")
    speech = pd.read_csv(context["speech"], encoding="utf-8-sig")
    landscape = pd.read_csv(POLITICAL_LANDSCAPE, encoding="utf-8-sig")
    pressure = build_automatic_third_candidate_pressure(
        profile,
        speech,
        landscape,
        issue_builder.ELECTION_DATES,
    )
    pressure_path = output_dir / "auto_candidate_role" / "third_candidate_pressure.csv"
    issue_builder._write(pressure, pressure_path)

    manifest = context["manifest"]
    forbidden = set(manifest.get("forbidden_manual_input_paths", []))
    forbidden.add(str(MANUAL_PRESSURE.relative_to(ROOT)).replace("\\", "/"))
    manifest.update(
        {
            "schema_version": "speech_derived_candidate_context_v3",
            "candidate_pressure_schema_version": PRESSURE_SCHEMA_VERSION,
            "manual_third_candidate_pressure_allowed": False,
            "manual_third_candidate_pressure_read_count": 0,
            "candidate_pressure_outcome_fields_used": [],
            "candidate_pressure_formula": (
                "sqrt(centrist_appeal * anti_major_party_appeal) times "
                "normalized equal-weight lane affinity"
            ),
            "forbidden_manual_input_paths": sorted(forbidden),
        }
    )
    manifest["outputs"]["automatic_third_candidate_pressure.csv"] = len(pressure)
    manifest["automatic_third_candidate_pressure"] = {
        "path": str(pressure_path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": pressure_path.stat().st_size,
        "sha256": issue_builder._sha256(pressure_path),
    }
    manifest_path = output_dir / "lineage_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    context["third_pressure"] = pressure_path
    context["manifest"] = manifest
    return context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_context(args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": result["manifest"]["schema_version"],
                "automatic_pressure_rows": result["manifest"]["outputs"][
                    "automatic_third_candidate_pressure.csv"
                ],
                "manual_pressure_reads": result["manifest"][
                    "manual_third_candidate_pressure_read_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
