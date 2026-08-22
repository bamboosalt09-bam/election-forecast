"""Run the outcome-free 2025 demonstration without external model features."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine.external_model_free_runtime import (  # noqa: E402
    assert_external_model_free_manifest,
    external_model_free_runtime,
    strip_external_model_inputs,
)
from scripts import run_prospective_forecast_v27 as v27  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "prospective_pres_2025_v28"


def run() -> Path:
    original_output = v27.OUTPUT_DIR
    try:
        v27.OUTPUT_DIR = OUTPUT_DIR
        with external_model_free_runtime():
            v27.run()
    finally:
        v27.OUTPUT_DIR = original_output

    manifest_path = OUTPUT_DIR / "input_manifest.csv"
    strip_external_model_inputs(manifest_path)
    assert_external_model_free_manifest(manifest_path)
    run_manifest_path = OUTPUT_DIR / "run_manifest.json"
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    payload["version"] = "v28"
    payload["predecessor_runtime"] = "v27"
    payload["external_neural_model_runtime"] = False
    payload["external_model_derived_inputs"] = []
    payload["parliamentary_source_policy"] = (
        "official_records_and_deterministic_issue_matching_only"
    )
    payload["model_parameters_changed"] = False
    payload["performance_metrics_computed"] = False
    run_manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return OUTPUT_DIR


if __name__ == "__main__":
    print(run().relative_to(ROOT).as_posix())
