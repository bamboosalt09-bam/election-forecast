"""Promotion guards for core-weighted regional polarization in V27."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The promoted artifacts are committed with this test. Keep an explicit
# existence guard so source-boundary checks remain safe in partial checkouts.
V27_OUTPUT = ROOT / "outputs/active_presidential_nested_v27/nested_predictions.csv"
if not V27_OUTPUT.exists():
    import pytest
    pytest.skip("V27 promoted artifacts are not present", allow_module_level=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v27_remains_in_the_declared_rollback_chain_under_v32() -> None:
    pointer = json.loads((ROOT / "data/config/current_presidential_model.json").read_text(encoding="utf-8"))
    assert pointer["active_version"] == "v32"
    assert pointer["predecessor"] == "v31"
    assert pointer["runner"] == "scripts/run_active_presidential_model_v32.py"
    assert pointer["prospective_runner"] == "scripts/run_prospective_forecast_v32.py"
    assert sha(ROOT / "outputs/active_presidential_nested_v26/nested_predictions.csv") == "9b66b813f97c3c2804a178ebb5b9104fa4a58553c75812f75affbb3b17773dd3"


def test_v27_preserves_national_levels_and_improves_regional_macro() -> None:
    v26 = pd.read_csv(ROOT / "outputs/active_presidential_nested_v26/nested_predictions.csv", low_memory=False)
    v27 = pd.read_csv(ROOT / "outputs/active_presidential_nested_v27/nested_predictions.csv", low_memory=False)
    name = "candidate_name_x"
    for keys, old in v26.groupby(["election_id", name]):
        new = v27.loc[(v27.election_id.eq(keys[0])) & (v27[name].eq(keys[1]))]
        old_level = np.average(old.layer_pred, weights=old.contest_votes)
        new_level = np.average(new.layer_pred, weights=new.contest_votes)
        np.testing.assert_allclose(new_level, old_level, atol=1e-10)
    summary = json.loads((ROOT / "outputs/active_presidential_nested_v27/summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["regional_equal_election_macro_mae_pp"] < 2.7122332621133673
    assert np.isclose(summary["metrics"]["national_equal_election_macro_mae_pp"], 0.7209938807856904, atol=1e-10)


def test_v27_prospective_preserves_national_allocation_and_composition() -> None:
    summary = pd.read_csv(ROOT / "outputs/prospective_pres_2025_v27/national_summary.csv")
    assert np.isclose(summary.predicted_share.sum(), 1.0)
    regional = pd.read_csv(ROOT / "outputs/prospective_pres_2025_v27/prospective_predictions.csv")
    assert np.allclose(regional.groupby("region_id").predicted_share.sum(), 1.0)
    manifest = json.loads((ROOT / "outputs/prospective_pres_2025_v27/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "v27"
    assert manifest["performance_metrics_computed"] is False
    assert manifest["pres_2025_outcome_present"] is False
