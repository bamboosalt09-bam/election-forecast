"""Guards for the V29 promotion.

V29 differs from V28 by one transform. The promotion rests on three claims that
are properties of its form, not of the panel: the national level is conserved,
an election without a third candidate is untouched, and the gain is the
parameter-free value rather than the better-scoring swept one. These tests pin
all three against the shipped artifact, pin that V28 did not move, and pin that
the 2025 demonstration is still disclosed as not regenerated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from presidential_issue_engine.third_share_dispersion_expansion import DEFAULT_GAIN

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v29"
V28_DIR = ROOT / "outputs" / "active_presidential_nested_v28"
V28_SHA256 = "23d6efd825244caa1f7b06b84e94cf581f00c6184aeb80769d8bb3d4c2a19fba"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer() -> dict[str, object]:
    return json.loads(POINTER.read_text(encoding="utf-8"))


def _metrics(directory: Path) -> dict[str, float]:
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))["metrics"]


def test_the_pointer_records_the_promotion_and_its_rollback() -> None:
    pointer = _pointer()
    assert pointer["active_version"] == "v29"
    assert pointer["predecessor"] == "v28"
    assert pointer["rollback_pointer"] == (
        "outputs/active_presidential_nested_v28/finalization_manifest.json"
    )
    assert pointer["prediction_sha256"] == _sha256(ACTIVE_DIR / "nested_predictions.csv")


def test_promoting_v29_did_not_move_v28() -> None:
    assert _sha256(V28_DIR / "nested_predictions.csv") == V28_SHA256


def test_the_regional_metric_improves_and_the_national_one_does_not_move() -> None:
    """The national figure is conserved by construction, not by tuning."""

    v28, v29 = _metrics(V28_DIR), _metrics(ACTIVE_DIR)
    assert v29["regional_equal_election_macro_mae_pp"] < v28[
        "regional_equal_election_macro_mae_pp"
    ]
    assert v29["national_equal_election_macro_mae_pp"] == pytest.approx(
        v28["national_equal_election_macro_mae_pp"], abs=1e-9
    )
    assert v29["winner_accuracy"] == v28["winner_accuracy"]
    assert v29["rows"] == v28["rows"]


def test_the_gain_stays_the_parameter_free_value() -> None:
    """0.50 scores better on the panel; adopting it would be a fitted constant."""

    assert DEFAULT_GAIN == 1.0
    finalization = json.loads(
        (ACTIVE_DIR / "finalization_manifest.json").read_text(encoding="utf-8")
    )
    expansion = finalization["third_share_dispersion_expansion"]
    assert expansion["gain"] == 1.0
    assert expansion["gain_selection"] == "parameter_free_unit_gain_not_swept"
    assert expansion["outcome_fields_used"] == []


def test_the_promotion_record_discloses_the_better_scoring_rejected_gain() -> None:
    promotion = json.loads(
        (ACTIVE_DIR / "promotion_manifest.json").read_text(encoding="utf-8")
    )
    assert promotion["active_version"] == "v29"
    assert promotion["post_2022_outcomes_used"] is False
    assert any("0_50" in scope for scope in promotion["rejected_scope"])
    disclosure = str(promotion["selection_disclosure"])
    assert "2.555129" in disclosure, "the rejected gain's better figure must be stated"
    assert "in-sample" in disclosure


def test_no_candidate_national_level_moved(
) -> None:
    audit = pd.read_csv(
        ACTIVE_DIR / "third_share_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    assert (audit["max_candidate_level_shift_pp"].abs() < 1e-9).all()


def test_the_two_candidate_election_is_untouched() -> None:
    """2012 has no third candidate, so the transform must not reach it."""

    audit = pd.read_csv(
        ACTIVE_DIR / "third_share_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    row = audit.loc[audit.election_id.eq("pres_2012")].iloc[0]
    assert float(row["predicted_third_share"]) == 0.0
    assert float(row["applied_factor"]) == 1.0

    v28 = pd.read_csv(V28_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False)
    v29 = pd.read_csv(ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False)
    keys = ["election_id", "region_id", "slot"]
    merged = v28.loc[v28.election_id.eq("pres_2012"), keys + ["layer_pred"]].merge(
        v29.loc[v29.election_id.eq("pres_2012"), keys + ["layer_pred"]],
        on=keys,
        suffixes=("_v28", "_v29"),
    )
    assert not merged.empty
    assert (merged.layer_pred_v28 - merged.layer_pred_v29).abs().max() < 1e-12


def test_2017_is_capped_at_the_feasible_factor() -> None:
    """Its nominal factor would drive two 홍준표 regions below zero."""

    audit = pd.read_csv(
        ACTIVE_DIR / "third_share_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    row = audit.loc[audit.election_id.eq("pres_2017")].iloc[0]
    assert bool(row["feasibility_capped"])
    assert float(row["applied_factor"]) < float(row["expansion_factor"])

    frame = pd.read_csv(ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False)
    assert float(frame.layer_pred.min()) >= 0.0


def test_the_2025_demonstration_is_disclosed_as_not_regenerated() -> None:
    """Shipping V29 beside a V28-era forecast must not be silent."""

    demonstration = _pointer()["prospective_demonstration"]
    assert demonstration["regenerated_for_v29"] is False
    assert demonstration["artifact"] == "outputs/prospective_pres_2025_v28"
    blocker = ROOT / str(demonstration["blocked_by"])
    assert blocker.is_file(), "the pointer must name a diagnosis that exists"
