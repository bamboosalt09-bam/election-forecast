"""Guards for the V31 promotion.

V31 replaces V29's additive dispersion expansion with a multiplicative one. The
claim is about what the transform *can* emit, not about what it scored: an
additive expansion capped at the first region to reach zero publishes that
region at exactly zero, and a multiplicative one cannot reach zero at all.

So these tests pin the property, not the metric. The regional figure improved
and the national one got worse, and neither is asserted as a condition of the
promotion being correct — V30's audit made that mistake and this one does not
repeat it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from presidential_issue_engine import multiplicative_dispersion_expansion as expansion

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v31"
V30_DIR = ROOT / "outputs" / "active_presidential_nested_v30"
V30_SHA256 = "afee25e582e201873f1785c7123004336f4dfb892791c30c4e6f3f7ab9d3049e"
V29_SHA256 = "fed959cdba1e127f91c2ab640a378d1f44a4a3e79b4c4a76893cf8d7c6153904"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer() -> dict[str, object]:
    return json.loads(POINTER.read_text(encoding="utf-8"))


def _active() -> pd.DataFrame:
    return pd.read_csv(
        ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )


def _audit() -> pd.DataFrame:
    return pd.read_csv(
        ACTIVE_DIR / "multiplicative_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )


def test_the_pointer_records_the_promotion_and_its_rollback() -> None:
    pointer = _pointer()
    assert pointer["active_version"] == "v31"
    assert pointer["predecessor"] == "v30"
    assert pointer["rollback_pointer"] == (
        "outputs/active_presidential_nested_v30/finalization_manifest.json"
    )
    assert pointer["prediction_sha256"] == _sha256(ACTIVE_DIR / "nested_predictions.csv")


def test_promoting_v31_moved_neither_frozen_predecessor() -> None:
    assert _sha256(V30_DIR / "nested_predictions.csv") == V30_SHA256
    assert _sha256(ROOT / "outputs/active_presidential_nested_v29/nested_predictions.csv") == V29_SHA256


def test_no_predicted_share_is_zero() -> None:
    """The whole promotion, stated as a property of the artifact."""

    frame = _active()
    assert float(frame["layer_pred"].min()) > 0.0
    # and not merely above zero: the value that motivated V31 was 0.0001%
    assert float(frame["layer_pred"].min()) > 0.005


def test_the_row_that_motivated_the_change() -> None:
    """2017 홍준표 광주 was published at exactly the cap's floor."""

    frame = _active()
    row = frame[
        frame.election_id.eq("pres_2017")
        & frame.region_id.eq("sido_29")
        & frame.slot.eq("B")
    ]
    assert len(row) == 1
    assert float(row["layer_pred"].iloc[0]) > 0.015
    previous = pd.read_csv(
        V30_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    was = previous[
        previous.election_id.eq("pres_2017")
        & previous.region_id.eq("sido_29")
        & previous.slot.eq("B")
    ]
    assert float(was["layer_pred"].iloc[0]) < 1e-5, "V30 published this row at the floor"


def test_the_feasibility_cap_is_gone_rather_than_widened() -> None:
    audit = _audit()
    assert "feasibility_capped" not in audit.columns
    assert "applied_factor" not in audit.columns
    # the factor is the nominal one everywhere, because nothing constrains it
    shares = audit["predicted_third_share"].astype(float)
    np.testing.assert_allclose(audit["expansion_factor"], 1.0 + shares, rtol=0, atol=1e-12)


def test_the_levels_are_reconciled_not_merely_expanded() -> None:
    """The multiplicative form alone moves levels by up to 0.465pp."""

    audit = _audit()
    assert (audit["max_candidate_level_shift_pp"].abs() < 1e-9).all()
    assert (audit["reconciliation_rounds"] >= 1).all()
    assert (audit["reconciliation_rounds"] < expansion.MAX_ROUNDS).all()


def test_every_region_still_sums_to_one() -> None:
    frame = _active()
    totals = frame.groupby(["election_id", "region_id"])["layer_pred"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, rtol=0, atol=1e-12)


def test_the_gain_stays_the_parameter_free_value() -> None:
    assert expansion.DEFAULT_GAIN == 1.0
    finalization = json.loads(
        (ACTIVE_DIR / "finalization_manifest.json").read_text(encoding="utf-8")
    )
    record = finalization["multiplicative_dispersion_expansion"]
    assert record["gain"] == 1.0
    assert record["gain_selection"] == "parameter_free_unit_gain_not_swept"
    assert record["outcome_fields_used"] == []


def test_the_transform_refuses_a_non_positive_input() -> None:
    """The log form is undefined at zero, so it must raise rather than nudge."""

    frame = _active().head(40).copy()
    frame.loc[frame.index[0], "layer_pred"] = 0.0
    try:
        expansion.apply_multiplicative_dispersion_expansion(
            frame, weight_column="forecast_time_region_weight"
        )
    except ValueError as error:
        assert "non-positive" in str(error)
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("a zero input was accepted")


def test_the_runner_refuses_a_missing_warmup_table(tmp_path: Path) -> None:
    """V30's shared module falls back to equal regions; this version does not."""

    from presidential_issue_engine import forecast_time_region_weights as weights
    from scripts import run_active_presidential_model_v31 as v31

    original = weights.WARMUP_TURNOUT
    try:
        weights.WARMUP_TURNOUT = tmp_path / "absent.csv"
        try:
            v31.require_forecast_time_inputs(_active())
        except FileNotFoundError as error:
            assert "warmup" in str(error) or "absent.csv" in str(error)
        else:  # pragma: no cover
            raise AssertionError("a missing warmup table was accepted")
    finally:
        weights.WARMUP_TURNOUT = original


def test_the_runner_refuses_inconsistent_regional_volumes() -> None:
    from scripts import run_active_presidential_model_v31 as v31

    frame = _active().copy()
    frame.loc[frame.index[0], "contest_votes"] = float(frame["contest_votes"].iloc[0]) + 1.0
    try:
        v31.require_forecast_time_inputs(frame)
    except ValueError as error:
        assert "disagrees" in str(error)
    else:  # pragma: no cover
        raise AssertionError("inconsistent turnout was accepted")


def test_the_promotion_record_states_the_property_not_the_score() -> None:
    promotion = json.loads(
        (ACTIVE_DIR / "promotion_manifest.json").read_text(encoding="utf-8")
    )
    assert promotion["active_version"] == "v31"
    assert promotion["post_2022_outcomes_used"] is False
    disclosure = str(promotion["selection_disclosure"])
    assert "cannot reach zero" in disclosure
    assert "0.724291" in disclosure, "the worse national figure must be stated"


def test_the_2025_demonstration_lost_its_zero_and_kept_its_levels() -> None:
    demonstration = _pointer()["prospective_demonstration"]
    assert demonstration["artifact"] == "outputs/prospective_pres_2025_v31"
    assert demonstration["regenerated_for_v31"] is True

    change = demonstration["change_from_published_v30_artifact"]
    assert change["winner_unchanged"] is True
    assert change["national_max_pp"] < 1e-9, "the correction must live inside regions"

    forecast = pd.read_csv(
        ROOT / "outputs/prospective_pres_2025_v31/prospective_predictions.csv",
        encoding="utf-8-sig",
    )
    assert float(forecast["predicted_share"].min()) > 0.015
    totals = forecast.groupby("region_id")["predicted_share"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, rtol=0, atol=1e-10)

    gwangju = forecast[forecast.region_id.eq("sido_29") & forecast.slot.eq("B")]
    assert float(gwangju["predicted_share"].iloc[0]) > 0.015

    manifest = json.loads(
        (ROOT / "outputs/prospective_pres_2025_v31/run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == "v31"
    assert manifest["performance_metrics_computed"] is False
