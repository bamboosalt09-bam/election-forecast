from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_prospective_forecast as prospective


def test_target_base_preserves_merged_candidate_names(monkeypatch) -> None:
    target = pd.DataFrame(
        {
            "election_id": ["pres_2025"],
            "region_id": ["sido_11"],
            "slot": ["A"],
            "candidate_name": ["candidate_a"],
        }
    )
    historical = pd.DataFrame(
        columns=[
            "election_id",
            "region_id",
            "slot",
            "candidate_name_x",
            "candidate_name_y",
            "contest_votes",
            "actual",
        ]
    )
    monkeypatch.setattr(
        prospective,
        "_prior_region_volume",
        lambda: pd.Series({"sido_11": 100.0}),
    )

    result = prospective._target_base(target, historical)

    assert result.loc[0, "candidate_name_x"] == "candidate_a"
    assert result.loc[0, "candidate_name_y"] == "candidate_a"
    assert np.isnan(result.loc[0, "actual"])


def test_committed_prospective_output_is_forecast_only() -> None:
    output = prospective.ROOT / "outputs/prospective_pres_2025_v23"
    predictions = pd.read_csv(
        output / "prospective_predictions.csv", encoding="utf-8-sig"
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

    assert predictions.columns.tolist() == list(prospective.OUTPUT_COLUMNS)
    assert len(predictions) == 51
    assert predictions["candidate_name"].astype(str).ne("0.0").all()
    assert np.allclose(
        predictions.groupby(["election_id", "region_id"])["predicted_share"].sum(),
        1.0,
    )
    assert manifest["forecast_cutoff"] == "2025-06-02"
    assert manifest["training_latest_election"] == "pres_2022"
    assert manifest["outcome_columns_used"] == []
    assert manifest["performance_metrics_computed"] is False
    assert manifest["pres_2025_outcome_present"] is False


def test_v24_requires_human_promoted_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prospective, "V24_CONFIG", tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="no human-promoted config"):
        prospective._config_path("v24")
