"""Promotion guards for the external-model-free V28 runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V27_HASH = "f40775599dde107abc6cf2312c648ad9c780f33c7a0adc4ccf3d74fd5049c55b"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v28_is_prediction_equivalent_and_external_model_runtime_free() -> None:
    active = ROOT / "outputs/active_presidential_nested_v28"
    assert _sha(active / "nested_predictions.csv") == V27_HASH
    manifest = pd.read_csv(active / "input_manifest.csv")
    paths = manifest.path.astype(str).str.replace("\\", "/", regex=False)
    assert not paths.str.contains("assembly_issue_character_overlay", regex=False).any()
    assert int(
        paths.str.endswith("data/raw/auto_issue_seed/candidate_issue_profile.csv").sum()
    ) == 1
    assert not paths.str.endswith(
        ("mega_issue_axis.csv", "mega_issue_attribution.csv")
    ).any()
    summary = json.loads((active / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["variant"] == "v28_external_model_free"
    assert summary["external_neural_model_runtime"] is False
    assert summary["external_model_derived_inputs"] == [
        "data/raw/auto_issue_seed/candidate_issue_profile.csv"
    ]


def test_v28_prospective_is_outcome_free_and_compositional() -> None:
    output = ROOT / "outputs/prospective_pres_2025_v28"
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["version"] == "v28"
    assert run_manifest["external_neural_model_runtime"] is False
    assert run_manifest["external_model_derived_inputs"] == []
    assert run_manifest["performance_metrics_computed"] is False
    manifest = pd.read_csv(output / "input_manifest.csv")
    assert not manifest.path.astype(str).str.contains(
        "assembly_issue_character_overlay|data/raw/auto_issue_seed/", regex=True
    ).any()
    frame = pd.read_csv(output / "prospective_predictions.csv")
    assert np.allclose(frame.groupby("region_id").predicted_share.sum(), 1.0, atol=1e-10)
