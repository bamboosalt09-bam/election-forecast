from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_active_presidential_model_v25 as v25


OUTPUT = Path("outputs/active_presidential_nested_v25")


def test_promoted_v25_preserves_winner_safe_v24_third_candidate_paths() -> None:
    assert "third_candidate_inputs" in v25.AVAILABLE_RUNTIME_REPAIRS
    assert "third_candidate_inputs" not in v25.RUNTIME_REPAIRS

    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["v23_runtime_repairs"]) == set(v25.RUNTIME_REPAIRS)
    assert summary["preserved_v24_runtime_paths"] == {
        "active_conversion_context": "data/raw/candidate_vote_conversion_context.csv",
        "third_candidate_profile": "data/raw/third_candidate_profile.csv",
        "third_candidate_pressure": "data/raw/third_candidate_pressure.csv",
        "reason": "preserve accepted V24 winner-safe weak-C runtime",
    }

    manifest = set(
        pd.read_csv(OUTPUT / "input_manifest.csv", encoding="utf-8-sig")["path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )
    assert "data/raw/third_candidate_profile.csv" in manifest
    assert "data/raw/third_candidate_pressure.csv" in manifest
    assert "outputs/automatic_controls_v23/third_candidate_profile.csv" not in manifest
    assert "outputs/automatic_controls_v23/third_candidate_pressure.csv" not in manifest


def test_v25_uses_only_the_accepted_prediction_tilted_weak_c_route() -> None:
    audit = pd.read_csv(
        OUTPUT / "weak_same_lane_refusal_audit.csv", encoding="utf-8-sig"
    )
    assert set(audit["recipient_weight_mode"].astype(str)) == {"prediction_tilted"}
    assert set(audit["floor_mode"].astype(str)) == {"theoretical"}
    assert np.allclose(pd.to_numeric(audit["gain"], errors="raise"), 0.50)
    assert len(audit.loc[audit["election_id"].eq("pres_2022")]) == 17


def test_v25_historical_artifact_contract_and_winner_gate() -> None:
    predictions = pd.read_csv(
        OUTPUT / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    assert len(predictions) == 232
    assert "pres_2025" not in set(predictions["election_id"].astype(str))
    assert np.allclose(
        predictions.groupby(["election_id", "region_id"])["layer_pred"].sum(),
        1.0,
    )

    report = v25.v24.report(OUTPUT)
    assert int(report["winner_correct"].sum()) == 4
    assert len(report) == 5
    assert bool(
        report.loc[report["election"].astype(str).eq("2022"), "winner_correct"].iloc[0]
    )
