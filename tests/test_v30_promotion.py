"""Guards for the V30 promotion.

V30 differs from V29 by one thing: the weight the two terminal transforms use to
locate each candidate's national level. It was ``contest_votes``, the target
election's own regional turnout, which exists only after the count. It is now
the previous election's regional valid votes.

The claims worth pinning are therefore about availability, not about score: no
scored election reads its own turnout as a weight, every scored election has a
predecessor to read instead (2002 reads 1997 from a shipped warmup table), and
the transforms V30 inherits still behave as V29 promised. The metric moved in
V30's favour and that is deliberately *not* what these tests assert.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from presidential_issue_engine import forecast_time_region_weights as weights
from presidential_issue_engine.third_share_dispersion_expansion import DEFAULT_GAIN

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v30"
V29_DIR = ROOT / "outputs" / "active_presidential_nested_v29"
V28_DIR = ROOT / "outputs" / "active_presidential_nested_v28"
V29_SHA256 = "fed959cdba1e127f91c2ab640a378d1f44a4a3e79b4c4a76893cf8d7c6153904"
V28_SHA256 = "23d6efd825244caa1f7b06b84e94cf581f00c6184aeb80769d8bb3d4c2a19fba"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer() -> dict[str, object]:
    return json.loads(POINTER.read_text(encoding="utf-8"))


def _active() -> pd.DataFrame:
    return pd.read_csv(
        ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )


def test_the_pointer_records_the_promotion_and_its_rollback() -> None:
    pointer = _pointer()
    assert pointer["active_version"] == "v30"
    assert pointer["predecessor"] == "v29"
    assert pointer["rollback_pointer"] == (
        "outputs/active_presidential_nested_v29/finalization_manifest.json"
    )
    assert pointer["prediction_sha256"] == _sha256(ACTIVE_DIR / "nested_predictions.csv")


def test_promoting_v30_moved_neither_frozen_predecessor() -> None:
    assert _sha256(V29_DIR / "nested_predictions.csv") == V29_SHA256
    assert _sha256(V28_DIR / "nested_predictions.csv") == V28_SHA256


def test_no_scored_election_is_weighted_by_its_own_turnout() -> None:
    """The whole point of the promotion, stated as a property of the artifact."""

    frame = _active()
    assert weights.WEIGHT_COLUMN in frame.columns
    for election, group in frame.groupby("election_id"):
        own = group.groupby("region_id")["contest_votes"].first()
        used = group.groupby("region_id")[weights.WEIGHT_COLUMN].first()
        shared = own.index.intersection(used.index)
        assert not own.loc[shared].equals(used.loc[shared]), (
            f"{election} still weights by its own turnout"
        )


def test_every_scored_election_weights_by_its_predecessor() -> None:
    frame = _active()
    volumes = {
        str(election): group.groupby("region_id")["contest_votes"].first()
        for election, group in frame.groupby("election_id")
    }
    warmup = pd.read_csv(weights.WARMUP_TURNOUT, encoding="utf-8-sig")
    volumes["pres_1997"] = warmup.set_index("region_id")["valid_votes"].astype(float)

    order = weights.SCORED_ORDER
    for index, election in enumerate(order):
        previous = "pres_1997" if index == 0 else order[index - 1]
        used = frame.loc[frame.election_id.eq(election)].groupby("region_id")[
            weights.WEIGHT_COLUMN
        ].first()
        source = volumes[previous]
        shared = used.index.intersection(source.index)
        assert len(shared) >= 15
        pd.testing.assert_series_equal(
            used.loc[shared].astype(float),
            source.loc[shared].astype(float),
            check_names=False,
        )
        # a region the predecessor did not have (세종 before 2012) takes its mean
        for region in used.index.difference(source.index):
            assert used[region] == float(source.mean())


def test_the_1997_warmup_table_reproduces_the_published_totals() -> None:
    """The transcription was checked by summation, so the check ships with it."""

    warmup = pd.read_csv(weights.WARMUP_TURNOUT, encoding="utf-8-sig")
    assert len(warmup) == 16
    assert int(warmup["electorate"].sum()) == 32_290_416
    assert int(warmup["votes_cast"].sum()) == 26_042_633
    assert int(warmup["valid_votes"].sum()) == 25_642_438


def test_the_error_columns_describe_the_shipped_prediction() -> None:
    """They described ``official_pred``, a pre-layer baseline, on all 232 rows."""

    frame = _active()
    expected = (frame["layer_pred"] - frame["actual"]) * 100.0
    assert (frame["err_pp"] - expected).abs().max() < 1e-9
    assert (frame["abs_err_pp"] - expected.abs()).abs().max() < 1e-9
    assert "baseline_pre_layer_pred" in frame.columns
    assert "baseline_pre_layer_err_pp" in frame.columns


def test_the_promotion_record_states_availability_not_score() -> None:
    promotion = json.loads(
        (ACTIVE_DIR / "promotion_manifest.json").read_text(encoding="utf-8")
    )
    assert promotion["active_version"] == "v30"
    assert promotion["post_2022_outcomes_used"] is False
    disclosure = str(promotion["selection_disclosure"])
    assert "contest_votes" in disclosure
    assert "1997" in disclosure


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


def test_no_candidate_national_level_moved() -> None:
    audit = pd.read_csv(
        ACTIVE_DIR / "third_share_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    assert (audit["max_candidate_level_shift_pp"].abs() < 1e-9).all()


def test_the_two_candidate_election_is_still_untouched_by_the_expansion() -> None:
    audit = pd.read_csv(
        ACTIVE_DIR / "third_share_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    row = audit.loc[audit.election_id.eq("pres_2012")].iloc[0]
    assert float(row["predicted_third_share"]) == 0.0
    assert float(row["applied_factor"]) == 1.0


def test_2017_is_capped_at_the_feasible_factor() -> None:
    """Its nominal factor would drive two 홍준표 regions below zero."""

    audit = pd.read_csv(
        ACTIVE_DIR / "third_share_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    row = audit.loc[audit.election_id.eq("pres_2017")].iloc[0]
    assert bool(row["feasibility_capped"])
    assert float(row["applied_factor"]) < float(row["expansion_factor"])
    assert float(_active()["layer_pred"].min()) >= 0.0


def test_the_2025_demonstration_is_unchanged_by_the_reweighting() -> None:
    """The prospective path already refused the target's turnout.

    That refusal is the admission V30 acts on, so the 2025 artifact must come
    out identical. If it ever moves, the two paths have diverged again.
    """

    demonstration = _pointer()["prospective_demonstration"]
    assert demonstration["artifact"] == "outputs/prospective_pres_2025_v30"
    assert demonstration["regenerated_for_v30"] is True
    assert demonstration["identical_to_v29_artifact"] is True
    assert demonstration["change_from_published_v28_artifact"]["winner_unchanged"] is True

    keys = ["region_id", "slot"]
    older = pd.read_csv(
        ROOT / "outputs/prospective_pres_2025_v29/prospective_predictions.csv",
        encoding="utf-8-sig",
    )
    newer = pd.read_csv(
        ROOT / "outputs/prospective_pres_2025_v30/prospective_predictions.csv",
        encoding="utf-8-sig",
    )
    merged = older[keys + ["predicted_share"]].merge(
        newer[keys + ["predicted_share"]], on=keys, suffixes=("_v29", "_v30")
    )
    assert not merged.empty
    assert (merged.predicted_share_v29 - merged.predicted_share_v30).abs().max() == 0.0


def test_the_2025_artifact_is_compositional_and_outcome_free() -> None:
    forecast = ROOT / "outputs/prospective_pres_2025_v30"
    predictions = pd.read_csv(forecast / "prospective_predictions.csv", encoding="utf-8-sig")
    totals = predictions.groupby("region_id")["predicted_share"].sum()
    assert totals.round(10).eq(1.0).all()
    assert float(predictions["predicted_share"].min()) >= 0.0

    manifest = json.loads((forecast / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "v30"
    assert manifest["performance_metrics_computed"] is False

    audit = pd.read_csv(
        forecast / "third_share_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    assert (audit["max_candidate_level_shift_pp"].abs() < 1e-9).all()
    assert bool(audit.iloc[0]["feasibility_capped"]), "2025 hits the feasibility cap"


def test_the_boundary_history_reference_is_not_a_promoted_model() -> None:
    """The harness's baseline must never be mistaken for a shipped artifact."""

    manifest = json.loads(
        (ROOT / "outputs/external_model_free_v25_baseline/baseline_manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["promoted"] is False
    assert manifest["scored"] is False
    assert manifest["runtime"] == "external_model_free"
    assert _pointer()["output"] != "outputs/external_model_free_v25_baseline"
