"""Guards for the V32 promotion.

V32 is the version that cannot be argued for with a metric. Its scored artifact
is byte-identical to V31's, so every macro figure is unchanged by construction
and no comparison can support or refuse it. The claim being pinned here is
therefore about the *contract*: what the prospective assembly is now required to
do, and what it is no longer allowed to do quietly.

That makes one test more important than the rest — the byte comparison. A
version whose whole story is "the scored panel cannot move" has to fail loudly
the moment it does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v32"
V31_DIR = ROOT / "outputs" / "active_presidential_nested_v31"
FORECAST_DIR = ROOT / "outputs" / "prospective_pres_2025_v32"
V31_SHA256 = "969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069"

if not (ACTIVE_DIR / "nested_predictions.csv").exists():
    pytest.skip("V32 promoted artifacts are not present", allow_module_level=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer() -> dict:
    return json.loads(POINTER.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(
        (ACTIVE_DIR / "finalization_manifest.json").read_text(encoding="utf-8")
    )


def test_the_pointer_records_the_promotion_and_its_rollback() -> None:
    pointer = _pointer()
    assert pointer["active_version"] == "v32"
    assert pointer["predecessor"] == "v31"
    assert pointer["rollback_pointer"] == (
        "outputs/active_presidential_nested_v31/finalization_manifest.json"
    )
    assert pointer["prediction_sha256"] == _sha256(ACTIVE_DIR / "nested_predictions.csv")
    assert pointer["canonical_document"] == "docs/FINAL_MODEL_V32_20260826.md"


def test_the_scored_panel_is_byte_identical_to_v31() -> None:
    """The promotion's central claim, as bytes rather than as a tolerance.

    A tolerance is the form of this that could later be loosened to accommodate
    a change that was supposed to be impossible.
    """

    assert _sha256(ACTIVE_DIR / "nested_predictions.csv") == V31_SHA256
    assert _sha256(V31_DIR / "nested_predictions.csv") == V31_SHA256
    assert _manifest()["verification"]["scored_predictions_byte_identical_to_v31"] is True


def test_both_macros_change_by_exactly_zero() -> None:
    verification = _manifest()["verification"]
    assert verification["national_macro_change_vs_v31_pp"] == 0.0
    assert verification["regional_macro_change_vs_v31_pp"] == 0.0
    v31 = json.loads((V31_DIR / "summary.json").read_text(encoding="utf-8"))["metrics"]
    v32 = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))["metrics"]
    for key in (
        "national_equal_election_macro_mae_pp",
        "regional_equal_election_macro_mae_pp",
        "winner_accuracy",
        "rows",
    ):
        assert v32[key] == v31[key], key
    # the one metric that must differ, so the two artifacts stay distinguishable
    assert v32["variant"] != v31["variant"]


def test_the_promotion_record_rests_on_the_contract_not_on_a_score() -> None:
    promotion = json.loads(
        (ACTIVE_DIR / "promotion_manifest.json").read_text(encoding="utf-8")
    )
    assert promotion["active_version"] == "v32"
    assert promotion["post_2022_outcomes_used"] is False
    assert "any_change_evaluated_against_the_2025_outcome" in promotion["rejected_scope"]
    disclosure = str(promotion["selection_disclosure"])
    assert "byte-identical" in disclosure
    assert "not on a score" in disclosure


def test_no_column_may_be_satisfied_by_a_zero() -> None:
    """The contract's invariant, exercised rather than described."""

    from presidential_issue_engine import prospective_feature_contract as contract

    column = "a_column_no_version_ever_declared"
    assert contract.classify(column)[0] == "UNCLASSIFIED"
    frame = pd.DataFrame({"region_id": ["sido_11"], "slot": ["A"]})
    with pytest.raises(contract.ProspectiveFeatureError) as raised:
        contract.resolve(frame, ["region_id", "slot", column], site="test")
    assert column in str(raised.value)


def test_an_outcome_only_column_is_never_filled_with_zero() -> None:
    """A zero here is a fabricated result downstream code cannot distinguish."""

    from presidential_issue_engine import prospective_feature_contract as contract

    for column in ("err_pp", "abs_err_pp"):
        kind, _ = contract.classify(column)
        assert kind == contract.OUTCOME_ONLY, column
    frame = pd.DataFrame({"region_id": ["sido_11"], "slot": ["A"]})
    filled = contract.resolve(frame, ["region_id", "slot", "err_pp"], site="test")
    assert filled["err_pp"].isna().all()


def test_the_five_model_active_families_are_built_for_the_target() -> None:
    """Every one of them was identically zero in the published V31 forecast."""

    artifact = FORECAST_DIR / "prediction_stage_audit.csv"
    if not artifact.is_file():
        pytest.skip("the V32 prospective artifact is not present in this tree")
    frame = pd.read_csv(artifact, encoding="utf-8-sig", low_memory=False)

    for column in (
        "regional_accent_reliability",
        "major_party_core_eligible",
        "lineage_identity_score",
        "wasted_vote_resistance",
        "strategic_transfer_confidence",
    ):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        assert bool((values != 0.0).any()), f"{column} is dead in the target frame"


def test_the_manifest_names_all_five_families() -> None:
    contract = _manifest()["prospective_feature_contract"]
    assert set(contract["model_active_families_recovered"]) == {
        "regional_accent_*",
        "major_party_core_eligible",
        "lineage_identity_*",
        "wasted_vote_resistance",
        "strategic_transfer_confidence",
    }
    assert contract["outcome_only_fill"] == "NaN"
    assert contract["required_derived_without_builder"] == "hard failure, never a zero"
    assert contract["outcome_fields_used"] == []


def test_the_external_model_derived_tables_are_refused_by_name() -> None:
    """The old rule matched a directory prefix and missed the temp-dir copy."""

    refusal = _manifest()["external_model_derived_input_refusal"]
    assert refusal["matched_by"] == "file_name"
    assert set(refusal["refused"]) == {
        "assembly_issue_character_overlay.csv",
        "mega_issue_axis.csv",
        "mega_issue_attribution.csv",
    }
    trace = pd.read_csv(FORECAST_DIR / "raw_input_read_trace.csv", encoding="utf-8-sig")
    honoured = trace.loc[trace["reader"].astype(str).ne("refused_by_v32"), "path"]
    normalized = honoured.astype(str).str.replace("\\", "/", regex=False)
    assert not normalized.str.endswith(tuple(refusal["refused"])).any()


def test_every_calibration_call_met_its_tolerance() -> None:
    from presidential_issue_engine import calibration_guard

    audit = pd.read_csv(
        ACTIVE_DIR / "calibration_acceptance_audit.csv", encoding="utf-8-sig"
    )
    assert len(audit) > 0
    assert bool(audit["converged"].astype(bool).all())
    assert bool(
        (audit["max_candidate_residual"].abs() <= calibration_guard.CALIBRATION_ABS_TOL).all()
    )
    assert bool(
        (audit["max_region_sum_residual"].abs() <= calibration_guard.CALIBRATION_ABS_TOL).all()
    )
    # running the full budget is not a success condition, and the record says so
    assert _manifest()["calibration_acceptance"][
        "budget_exhaustion_is_not_a_success_condition"
    ] is True


def test_promoting_v32_moved_no_frozen_predecessor() -> None:
    for version, expected in (
        ("v23", "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b"),
        ("v30", "afee25e582e201873f1785c7123004336f4dfb892791c30c4e6f3f7ab9d3049e"),
        ("v31", V31_SHA256),
    ):
        path = ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv"
        assert _sha256(path) == expected, version


def test_the_2025_forecast_moved_and_the_manifest_measured_it() -> None:
    """The figures are read off the two artifacts, not transcribed."""

    change = _pointer()["prospective_demonstration"]["change_from_published_v31_artifact"]
    before = pd.read_csv(
        ROOT / "outputs/prospective_pres_2025_v31/prospective_predictions.csv",
        encoding="utf-8-sig",
    )
    after = pd.read_csv(
        FORECAST_DIR / "prospective_predictions.csv", encoding="utf-8-sig"
    )
    merged = before.merge(after, on=["region_id", "slot"], suffixes=("_v31", "_v32"))
    worst = float(
        (merged["predicted_share_v32"] - merged["predicted_share_v31"]).abs().max() * 100
    )
    assert abs(worst - float(change["regional_max_pp"])) < 1e-9
    assert change["winner_unchanged"] is True
    assert change["ranking_unchanged"] is True

    totals = after.groupby("region_id")["predicted_share"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, rtol=0, atol=1e-10)
    # V31's own promotion property must survive V32
    assert float(after["predicted_share"].min()) > 0.015


def test_the_forecast_still_computes_no_performance_metric() -> None:
    manifest = json.loads(
        (FORECAST_DIR / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == "v32"
    assert manifest["performance_metrics_computed"] is False
    assert manifest["forecast_cutoff"] == "2025-06-02"
    assert manifest["candidate_selection_outcome_fields_used"] == []
