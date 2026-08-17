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


def test_candidate_strength_prefers_direct_speech_context(monkeypatch, tmp_path) -> None:
    context = tmp_path / "candidate_vote_conversion_context.csv"
    rows = []
    for election_id, name, slot, weight in [
        ("pres_2022", "old", "A", 0.4),
        ("pres_2025", "candidate_a", "A", 0.7),
        ("pres_2025", "candidate_b", "B", 0.6),
        ("pres_2025", "candidate_c", "C", 0.3),
    ]:
        rows.append(
            {
                "election_id": election_id,
                "slot": slot,
                "candidate_name": name,
                "candidate_weight": weight,
                "confidence": 0.5,
                "available_date": "2025-06-02" if election_id == "pres_2025" else "2022-03-08",
            }
        )
    pd.DataFrame(rows).to_csv(context, index=False)
    monkeypatch.setattr(prospective, "CANDIDATE_CONVERSION_HISTORY", context)
    selected = pd.DataFrame(
        {
            "candidate_id": ["id_b", "id_a", "id_c"],
            "candidate_name": ["candidate_b", "candidate_a", "candidate_c"],
            "slot": ["A", "B", "C"],
        }
    )

    combined, diagnostics = prospective._candidate_strength_context(
        selected,
        pd.DataFrame(),
    )

    assert diagnostics["method"] == "direct_speech_derived_candidate_context"
    target = combined.loc[combined["election_id"].eq("pres_2025")]
    assert target.set_index("candidate_name")["slot"].to_dict() == {
        "candidate_b": "A",
        "candidate_a": "B",
        "candidate_c": "C",
    }
