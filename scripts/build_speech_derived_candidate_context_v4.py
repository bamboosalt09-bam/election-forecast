"""Build candidate context v4 with automatic non-major regional organization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.speech_derived_candidate_regional_base import (  # noqa: E402
    SCHEMA_VERSION as REGIONAL_BASE_SCHEMA_VERSION,
    build_automatic_candidate_regional_base,
)
from scripts import build_speech_derived_candidate_context_v2 as v2_builder  # noqa: E402
from scripts import build_speech_derived_issue_context as issue_builder  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "speech_derived_candidate_context_v4"
BLOC_HISTORY = ROOT / "data" / "raw" / "bloc_history_results.csv"
MANUAL_REGIONAL_BASE = (ROOT / "data" / "raw" / "candidate_regional_base.csv").resolve()


def build_context(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    context = v2_builder.build_context(output_dir)
    speech = pd.read_csv(context["speech"], encoding="utf-8-sig")
    history = pd.read_csv(BLOC_HISTORY, encoding="utf-8-sig")
    regional_base = build_automatic_candidate_regional_base(
        speech,
        history,
        issue_builder.ELECTION_DATES,
    )
    regional_base_path = (
        output_dir / "auto_candidate_role" / "candidate_regional_base.csv"
    )
    issue_builder._write(regional_base, regional_base_path)

    manifest = context["manifest"]
    forbidden = set(manifest.get("forbidden_manual_input_paths", []))
    forbidden.add(str(MANUAL_REGIONAL_BASE.relative_to(ROOT)).replace("\\", "/"))
    manifest.update(
        {
            "schema_version": "speech_derived_candidate_context_v4",
            "candidate_regional_base_schema_version": REGIONAL_BASE_SCHEMA_VERSION,
            "manual_candidate_regional_base_allowed": False,
            "manual_candidate_regional_base_read_count": 0,
            "candidate_regional_base_outcome_fields_used": [],
            "candidate_regional_base_formula": (
                "positive regional excess in latest prior direct-party ballot "
                "for non-major non-independent candidate blocs"
            ),
            "forbidden_manual_input_paths": sorted(forbidden),
        }
    )
    manifest["outputs"]["automatic_candidate_regional_base.csv"] = len(
        regional_base
    )
    manifest["automatic_candidate_regional_base"] = {
        "path": str(regional_base_path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": regional_base_path.stat().st_size,
        "sha256": issue_builder._sha256(regional_base_path),
        "bloc_history_path": str(BLOC_HISTORY.relative_to(ROOT)).replace("\\", "/"),
        "bloc_history_sha256": issue_builder._sha256(BLOC_HISTORY),
    }
    manifest_path = output_dir / "lineage_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    context["candidate_regional_base"] = regional_base_path
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
                "automatic_regional_base_rows": result["manifest"]["outputs"][
                    "automatic_candidate_regional_base.csv"
                ],
                "manual_regional_base_reads": result["manifest"][
                    "manual_candidate_regional_base_read_count"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
