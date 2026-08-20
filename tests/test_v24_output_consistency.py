from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
V23_PREDICTIONS = ROOT / "outputs" / "active_presidential_nested_v23" / "nested_predictions.csv"
V24_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v24"
V24_SENSITIVITY = ROOT / "outputs" / "v24_structural_residual_hypotheses"
V24_FLOOR_SENSITIVITY = ROOT / "outputs" / "v24_floor_recalibration_hypotheses"
V23_FROZEN_SHA256 = "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b"


def test_v23_frozen_prediction_artifact_is_unchanged() -> None:
    assert hashlib.sha256(V23_PREDICTIONS.read_bytes()).hexdigest() == V23_FROZEN_SHA256


def test_v24_public_metrics_are_computed_after_v24_extensions() -> None:
    payload = json.loads((V24_OUTPUT / "summary.json").read_text(encoding="utf-8"))
    by_election = pd.read_csv(V24_OUTPUT / "by_election.csv", encoding="utf-8-sig")
    national = pd.read_csv(V24_OUTPUT / "national_predictions.csv", encoding="utf-8-sig")
    stages = pd.read_csv(V24_OUTPUT / "candidate_stage_summary.csv", encoding="utf-8-sig")

    assert payload["metrics"]["variant"] == "v24_structural_residual"
    assert set(by_election["variant"]) == {"v24_structural_residual"}
    assert set(national["variant"]) == {"v24_structural_residual"}
    assert (stages["variant"] == "v24_structural_residual").sum() == 1
    assert payload["metrics"]["regional_equal_election_macro_mae_pp"] == pytest.approx(
        by_election["regional_weighted_mae_pp"].mean(), abs=1e-12
    )
    assert payload["metrics"]["national_equal_election_macro_mae_pp"] == pytest.approx(
        by_election["national_candidate_mae_pp"].mean(), abs=1e-12
    )
    assert payload["third_candidate_lineage_ceiling"]["audit_rows"] == 12
    assert payload["third_candidate_lineage_ceiling"]["affected_elections"] == ["pres_2002"]
    assert payload["rejection_beneficiary_routing_restored_from_v23"] is True
    assert payload["strong_incumbent_veto"]["audit_rows"] == 33
    assert payload["strong_incumbent_veto"]["affected_elections"] == [
        "pres_2007",
        "pres_2017",
    ]
    assert payload["strong_incumbent_veto"]["projected_margin_threshold"] == 0.10
    assert payload["strong_incumbent_veto"]["gain"] == 1.00
    assert payload["strong_incumbent_veto"]["rupture_floor_erosion_enabled"] is True
    assert payload["strong_incumbent_veto"]["theoretical_floor"] == 0.01
    assert payload["strong_incumbent_veto"]["outcome_fields_used"] == []
    assert payload["weak_same_lane_refusal"]["audit_rows"] == 33
    assert payload["weak_same_lane_refusal"]["affected_elections"] == [
        "pres_2002",
        "pres_2022",
    ]
    assert payload["weak_same_lane_refusal"]["gain"] == 0.50
    assert payload["weak_same_lane_refusal"]["floor_mode"] == "theoretical"
    assert payload["weak_same_lane_refusal"]["theoretical_floor"] == 0.01
    assert (
        payload["weak_same_lane_refusal"]["recipient_weight_mode"]
        == "prediction_tilted"
    )
    assert payload["weak_same_lane_refusal"]["outcome_fields_used"] == []


def test_v24_final_predictions_remain_compositional() -> None:
    frame = pd.read_csv(
        V24_OUTPUT / "nested_predictions.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    totals = frame.groupby(["election_id", "region_id"])["layer_pred"].sum()

    assert len(frame) == 232
    assert not frame["election_id"].astype(str).str.contains("2025").any()
    assert (totals - 1.0).abs().max() < 1e-12


def test_v24_hypothesis_sensitivity_is_recorded_without_2025() -> None:
    manifest = json.loads(
        (V24_SENSITIVITY / "manifest.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(V24_SENSITIVITY / "metrics.csv", encoding="utf-8-sig")
    by_election = pd.read_csv(
        V24_SENSITIVITY / "by_election.csv",
        encoding="utf-8-sig",
    )
    transfers = pd.read_csv(
        V24_SENSITIVITY / "transfer_audit.csv",
        encoding="utf-8-sig",
    )

    assert manifest["purpose"] == "sensitivity record, not coefficient optimisation"
    assert manifest["prospective_elections_excluded"] is True
    assert manifest["declared_primary"] == {
        "strong_incumbent_veto_gain": 0.50,
        "weak_same_lane_gain": 0.25,
    }
    assert len(metrics) == 16
    primary = metrics.loc[metrics["declared_primary"]]
    assert len(primary) == 1
    assert float(primary.iloc[0]["strong_incumbent_veto_gain"]) == 0.50
    assert float(primary.iloc[0]["weak_same_lane_gain"]) == 0.25
    for frame in (by_election, transfers):
        assert not frame["election_id"].astype(str).str.contains("2025").any()


def test_v24_floor_recalibration_records_original_failure_and_followup() -> None:
    manifest = json.loads(
        (V24_FLOOR_SENSITIVITY / "manifest.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(
        V24_FLOOR_SENSITIVITY / "metrics.csv",
        encoding="utf-8-sig",
    )
    by_election = pd.read_csv(
        V24_FLOOR_SENSITIVITY / "by_election.csv",
        encoding="utf-8-sig",
    )
    national = pd.read_csv(
        V24_FLOOR_SENSITIVITY / "national_predictions.csv",
        encoding="utf-8-sig",
    )
    transfers = pd.read_csv(
        V24_FLOOR_SENSITIVITY / "transfer_audit.csv",
        encoding="utf-8-sig",
    )
    summary = json.loads((V24_OUTPUT / "summary.json").read_text(encoding="utf-8"))

    assert manifest["variant_count"] == 48
    assert manifest["prospective_elections_excluded"] is True
    assert len(metrics) == 48
    original = metrics.loc[metrics["declared_primary"]]
    followup = metrics.loc[metrics["followup_structural_candidate"]]
    assert len(original) == 1
    assert len(followup) == 1
    assert float(original.iloc[0]["winner_accuracy"]) == 0.60
    assert float(followup.iloc[0]["winner_accuracy"]) == 0.80
    assert followup.iloc[0]["recipient_weight_mode"] == "prediction_tilted"
    assert float(followup.iloc[0]["regional_equal_election_macro_mae_pp"]) == pytest.approx(
        summary["metrics"]["regional_equal_election_macro_mae_pp"], abs=1e-12
    )
    for frame in (by_election, national, transfers):
        assert not frame["election_id"].astype(str).str.contains("2025").any()
