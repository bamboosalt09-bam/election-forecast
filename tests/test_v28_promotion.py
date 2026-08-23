"""Promotion guards for the external-model-free V28 runtime."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V28_HASH = "23d6efd825244caa1f7b06b84e94cf581f00c6184aeb80769d8bb3d4c2a19fba"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v28_is_frozen_and_external_model_runtime_free() -> None:
    active = ROOT / "outputs/active_presidential_nested_v28"
    assert _sha(active / "nested_predictions.csv") == V28_HASH
    manifest = pd.read_csv(active / "input_manifest.csv")
    paths = manifest.path.astype(str).str.replace("\\", "/", regex=False)
    assert not paths.str.contains("assembly_issue_character_overlay", regex=False).any()
    assert int(
        paths.str.endswith("data/raw/auto_issue_seed/candidate_issue_profile.csv").sum()
    ) == 1
    assert not paths.str.endswith(
        ("mega_issue_axis.csv", "mega_issue_attribution.csv")
    ).any()
    assert not paths.str.endswith("kospi_daily.csv").any()
    assert int(paths.str.endswith("kospi_election_context.csv").sum()) == 1
    summary = json.loads((active / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["variant"] == "v28_external_model_free"
    assert summary["external_neural_model_runtime"] is False
    assert summary["external_model_derived_inputs"] == [
        "data/raw/auto_issue_seed/candidate_issue_profile.csv"
    ]
    assert summary["external_model_seed_boundary_enforced"] is True


def test_v28_seed_boundary_survives_nested_config_replacement() -> None:
    from presidential_issue_engine import issue_vote_engine as engine
    from presidential_issue_engine.external_model_free_runtime import (
        ENHANCED_ISSUES_ENV,
        SEED_BLOCK_ENV,
        external_model_free_runtime,
    )

    previous = os.environ.get(SEED_BLOCK_ENV)
    previous_enhanced = os.environ.get(ENHANCED_ISSUES_ENV)
    with external_model_free_runtime():
        from scripts import evaluate_electorate_layers as electorate_evaluation

        assert os.environ[ENHANCED_ISSUES_ENV] == "0"
        assert electorate_evaluation._read_csv(
            electorate_evaluation.OVERLAY_PATH
        ).empty
        assert engine._registered_issue_seed_path(
            engine.ENHANCED_CANDIDATE_ISSUE_PROFILE,
            engine.AUTO_CANDIDATE_ISSUE_PROFILE,
        ) == engine.AUTO_CANDIDATE_ISSUE_PROFILE
        assert engine._registered_issue_seed_path(
            engine.ENHANCED_MEGA_ISSUE_AXIS,
            engine.AUTO_MEGA_ISSUE_AXIS,
        ) == ""
        assert engine._registered_issue_seed_path(
            engine.ENHANCED_MEGA_ISSUE_ATTRIBUTION,
            engine.AUTO_MEGA_ISSUE_ATTRIBUTION,
        ) == ""
    assert os.environ.get(SEED_BLOCK_ENV) == previous
    assert os.environ.get(ENHANCED_ISSUES_ENV) == previous_enhanced


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
