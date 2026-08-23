"""Run V28 without neural inference or the sentence-level stance overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine.external_model_free_runtime import (  # noqa: E402
    assert_external_model_free_manifest,
    external_model_free_runtime,
    strip_external_model_inputs,
)
from scripts import run_active_presidential_model_v24 as v24  # noqa: E402
from scripts import run_active_presidential_model_v27 as v27  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v28"
FINAL_VARIANT = "v28_external_model_free"


def run(output_dir: Path | None = None) -> Path:
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    with external_model_free_runtime():
        v27.run(output_dir=destination)

    manifest_path = destination / "input_manifest.csv"
    strip_external_model_inputs(manifest_path)
    assert_external_model_free_manifest(manifest_path)

    for filename in ("by_election.csv", "national_predictions.csv"):
        path = destination / filename
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["variant"] = FINAL_VARIANT
        v24._atomic_csv_crlf(frame, path)

    summary_path = destination / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["metrics"]["variant"] = FINAL_VARIANT
    payload["policy_version"] = "active_v28_external_model_runtime_free"
    payload["predecessor"] = "v27"
    payload["external_neural_model_runtime"] = False
    payload["external_model_derived_inputs"] = [
        "data/raw/auto_issue_seed/candidate_issue_profile.csv"
    ]
    payload["removed_external_model_derived_inputs"] = [
        "data/raw/assembly_issue_character_overlay.csv",
        "data/raw/auto_issue_seed/mega_issue_axis.csv",
        "data/raw/auto_issue_seed/mega_issue_attribution.csv",
    ]
    payload["external_model_seed_boundary_enforced"] = True
    payload["parliamentary_source_policy"] = (
        "official_records_plus_disclosed_frozen_historical_candidate_issue_aggregate"
    )
    payload["post_2022_outcomes_used"] = False
    v24._atomic_json_crlf(payload, summary_path)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    destination = run(args.output_dir)
    print(v24.report(destination).to_string(index=False))


if __name__ == "__main__":
    main()
