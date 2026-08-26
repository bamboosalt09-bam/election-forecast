"""Promote and freeze V32: prospective/historical feature-contract parity.

V32 is the version whose scored panel cannot move. Everything it changes is
on the path that builds the target election's features, and that path is not
exercised by the five scored elections. So the manifest below records a
measured zero rather than an improvement, and the case for the version is
made on what the assembly is now required to do, not on a score.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import calibration_guard  # noqa: E402
from presidential_issue_engine import multiplicative_dispersion_expansion  # noqa: E402
from scripts import finalize_active_presidential_model_v25 as shared  # noqa: E402

ACTIVE_DIR = ROOT / "outputs/active_presidential_nested_v32"
POINTERS = (
    ROOT / "data/config/current_presidential_model.json",
    ROOT / "data/config/active_presidential_model.json",
)
V31_SHA256 = "969e63fe5239462c9f26a73ff8b97a196d543063821ba0577d1b6563ff2dd069"
#: V31's published macros, recorded so the manifest can state the change
#: rather than assert a direction.
V31_NATIONAL_MACRO_PP = 0.7242913678028117
V31_REGIONAL_MACRO_PP = 2.5007010072077227

PROSPECTIVE_ARTIFACT = "outputs/prospective_pres_2025_v32"
SUPERSEDED_PROSPECTIVE_ARTIFACT = "outputs/prospective_pres_2025_v31"


def _prospective_change() -> dict[str, object]:
    """Measure V32's 2025 forecast against V31's rather than restate it.

    The previous manifest hardcoded three conclusions about the transform and
    two of them had gone false. Anything here that can be read off the two
    artifacts is read off the two artifacts.
    """

    def load(directory: str, name: str) -> pd.DataFrame:
        return pd.read_csv(ROOT / directory / name, encoding="utf-8-sig")

    keys = ["region_id", "slot"]
    before = load(SUPERSEDED_PROSPECTIVE_ARTIFACT, "prospective_predictions.csv")
    after = load(PROSPECTIVE_ARTIFACT, "prospective_predictions.csv")
    merged = before.merge(after, on=keys, suffixes=("_v31", "_v32"))
    if len(merged) != len(after):
        raise RuntimeError("the two prospective artifacts do not align row for row")
    regional = (merged["predicted_share_v32"] - merged["predicted_share_v31"]).abs() * 100

    national_before = load(SUPERSEDED_PROSPECTIVE_ARTIFACT, "national_summary.csv")
    national_after = load(PROSPECTIVE_ARTIFACT, "national_summary.csv")
    national = national_before.merge(national_after, on="slot", suffixes=("_v31", "_v32"))
    national_shift = (
        national["predicted_share_v32"] - national["predicted_share_v31"]
    ) * 100

    def ordering(frame: pd.DataFrame, column: str) -> list[str]:
        return list(frame.sort_values(column, ascending=False)["slot"])

    return {
        "artifact": PROSPECTIVE_ARTIFACT,
        "regenerated_for_v32": True,
        "history_reference": "outputs/external_model_free_v25_baseline",
        "supersedes": SUPERSEDED_PROSPECTIVE_ARTIFACT,
        "change_from_published_v31_artifact": {
            "regional_max_pp": float(regional.max()),
            "regional_mean_pp": float(regional.mean()),
            "national_max_abs_pp": float(national_shift.abs().max()),
            "national_shift_pp_by_slot": {
                str(row.slot): float(
                    (row.predicted_share_v32 - row.predicted_share_v31) * 100
                )
                for row in national.itertuples(index=False)
            },
            "winner_unchanged": bool(
                ordering(national_before, "predicted_share")[0]
                == ordering(national_after, "predicted_share")[0]
            ),
            "ranking_unchanged": bool(
                ordering(national_before, "predicted_share")
                == ordering(national_after, "predicted_share")
            ),
        },
        "cause_of_change": (
            "Five feature families that the prospective assembly had been "
            "filling with zero are now built for the target: the 27 "
            "regional_accent columns, major_party_core_eligible, the five "
            "lineage_identity columns, wasted_vote_resistance and "
            "strategic_transfer_confidence. The three external-model-derived "
            "seed tables are refused rather than read. No 2025 outcome was used, "
            "and the size of the move was measured after the change was decided, "
            "not used to decide it."
        ),
    }


SELECTION_DISCLOSURE = (
    "V32 makes the prospective feature assembly obey the same contract as the "
    "historical one. Until now it closed the gap between the two frames with a "
    "blanket `out[column] = 0.0`, under a comment asserting that anything "
    "missing was diagnostic-only. A sweep of the shipped 2025 artifact found 40 "
    "columns identically zero across all 51 rows while populated for every "
    "scored election, and five families among them were model-active - the "
    "27-column regional accent layer, major_party_core_eligible, the "
    "lineage_identity family, wasted_vote_resistance and "
    "strategic_transfer_confidence. Zero is a legal value everywhere they "
    "landed, so none of it surfaced in the output. Every column the target "
    "lacks is now classified REQUIRED_DERIVED, EXPLICIT_ZERO, OUTCOME_ONLY or "
    "DIAGNOSTIC_ONLY; a required column with no builder is a hard failure, an "
    "outcome-only column becomes NaN rather than a fabricated zero, and an "
    "unclassified column stops the run. "
    "No parameter was selected and no threshold was tuned. The scored panel "
    "cannot move - V32's nested_predictions.csv is byte-identical to V31's, "
    "both macros change by exactly 0.000000pp - because the scored elections "
    "never take the assembly path that was changed. The version was therefore "
    "decided on the correctness of the feature contract, not on a score, and "
    "the 2025 outcome was not consulted at any point. The three "
    "external-model-derived seed tables are refused at the reader and the "
    "refusal is recorded, replacing a rule that matched on directory path and "
    "so missed the same table once it had been copied to a temporary directory."
)


def main() -> None:
    rollback = ROOT / "outputs/active_presidential_nested_v31/nested_predictions.csv"
    if shared._sha256(rollback) != V31_SHA256:
        raise RuntimeError("V31 rollback prediction drift")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    intervals = json.loads(
        (ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    prediction_hash = shared._sha256(ACTIVE_DIR / "nested_predictions.csv")
    # read what the run actually produced, so the manifest reports rather
    # than asserts
    predictions = pd.read_csv(
        ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    expansion_audit = pd.read_csv(
        ACTIVE_DIR / "multiplicative_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    calibration_audit = pd.read_csv(
        ACTIVE_DIR / "calibration_acceptance_audit.csv", encoding="utf-8-sig"
    )
    region_sums_hold = bool(
        predictions.groupby(["election_id", "region_id"])["layer_pred"]
        .sum()
        .sub(1.0)
        .abs()
        .lt(1e-12)
        .all()
    )
    artifacts = [
        "scripts/run_active_presidential_model_v32.py",
        "scripts/run_prospective_forecast_v32.py",
        "scripts/build_active_v32_predictive_intervals.py",
        "scripts/evaluate_electorate_layers.py",
        "scripts/finalize_active_presidential_model_v32.py",
        "scripts/audit_public_active_presidential_model_v32.py",
        "scripts/verify_v32_clean_reproduction.py",
        "scripts/verify_v32_prospective_reproduction.py",
        "scripts/build_external_model_free_v25_baseline.py",
        "presidential_issue_engine/multiplicative_dispersion_expansion.py",
        "presidential_issue_engine/forecast_time_region_weights.py",
        "presidential_issue_engine/party_regionalism_dispersion.py",
        "presidential_issue_engine/fixed_dataset/pres_1997_regional_turnout.csv",
        "presidential_issue_engine/external_model_free_runtime.py",
        "presidential_issue_engine/issue_vote_engine.py",
        "presidential_issue_engine/fixed_dataset/kospi_election_context.csv",
        "presidential_issue_engine/prospective_feature_contract.py",
        "presidential_issue_engine/calibration_guard.py",
        "presidential_issue_engine/raw_input_read_trace.py",
        "presidential_issue_engine/unified_lineage_identity.py",
        "docs/FINAL_MODEL_V32_20260826.md",
        "docs/EXPERIMENT_V32_PROSPECTIVE_FEATURE_CONTRACT_20260826.md",
        "docs/PROSPECTIVE_FEATURE_CONTRACT_20260826.md",
        "docs/DIAGNOSIS_INPUT_BOUNDARY_AND_CALIBRATION_20260826.md",
        "outputs/active_presidential_nested_v32/nested_predictions.csv",
        "outputs/active_presidential_nested_v32/summary.json",
        "outputs/active_presidential_nested_v32/by_election.csv",
        "outputs/active_presidential_nested_v32/national_predictions.csv",
        "outputs/active_presidential_nested_v32/input_manifest.csv",
        "outputs/active_presidential_nested_v32/multiplicative_dispersion_expansion_audit.csv",
        "outputs/active_presidential_nested_v32/national_predictive_intervals.csv",
        "outputs/active_presidential_nested_v32/predictive_interval_summary.csv",
        "outputs/active_presidential_nested_v32/predictive_interval_components.csv",
        "outputs/active_presidential_nested_v32/predictive_interval_manifest.json",
        "outputs/active_presidential_nested_v32/calibration_acceptance_audit.csv",
        "outputs/active_presidential_nested_v32/raw_input_read_trace.csv",
        "outputs/prospective_pres_2025_v32/prospective_predictions.csv",
        "outputs/prospective_pres_2025_v32/national_summary.csv",
        "outputs/prospective_pres_2025_v32/target_feature_audit.csv",
        "outputs/prospective_pres_2025_v32/raw_input_read_trace.csv",
        "outputs/prospective_pres_2025_v32/run_manifest.json",
    ]
    promotion = {
        "schema": "presidential_model_promotion_v1",
        "status": "promoted_prospective_feature_contract",
        "active_version": "v32",
        "predecessor": "v31",
        "post_2022_outcomes_used": False,
        "point_metrics": summary["metrics"],
        "accepted_scope": [
            "prospective_historical_feature_contract_parity",
            "runtime_lineage_routing_through_the_central_registry",
            "external_model_derived_input_refusal_matched_by_file_name",
            "dispersion_calibration_acceptance_tolerance",
        ],
        "rejected_scope": [
            "any_change_evaluated_against_the_2025_outcome",
            "tuning_the_accent_gain_map_to_move_the_2025_forecast",
        ],
        "selection_disclosure": SELECTION_DISCLOSURE,
        "prospective_demonstration": _prospective_change(),
        "rollback": {"version": "v31", "prediction_sha256": V31_SHA256},
    }
    shared._atomic_json(promotion, ACTIVE_DIR / "promotion_manifest.json")
    artifacts.append("outputs/active_presidential_nested_v32/promotion_manifest.json")

    finalization = {
        "schema": "presidential_model_finalization_v1",
        "status": "frozen_prospective_feature_contract",
        "active_version": "v32",
        "base_config_version": "v23",
        "scored_development_elections": [
            "pres_2002",
            "pres_2007",
            "pres_2012",
            "pres_2017",
            "pres_2022",
        ],
        "post_2022_outcomes_used": False,
        "untouched_historical_holdout": False,
        "external_neural_model_runtime": False,
        "external_model_derived_inputs": [
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        ],
        "metrics": summary["metrics"],
        # V31's transform, inherited unchanged and re-measured here rather
        # than carried forward as a claim.
        "multiplicative_dispersion_expansion": {
            "gain": multiplicative_dispersion_expansion.DEFAULT_GAIN,
            "gain_selection": "parameter_free_unit_gain_not_swept",
            "index": "model_predicted_third_placed_national_level",
            # measured from the shipped audit, not asserted. The previous
            # version of this block hardcoded three claims, and two of them
            # were false for V32: it listed pres_2017 as feasibility-capped
            # when removing the cap is the reason V32 exists, and it declared
            # the national macro not worse when V32 accepts a +0.0039pp cost
            # on purpose. A record that states its conclusions as literals
            # cannot go stale loudly.
            "candidate_national_level_preserved": bool(
                expansion_audit["max_candidate_level_shift_pp"].abs().lt(1e-9).all()
            ),
            "worst_candidate_level_shift_pp": float(
                expansion_audit["max_candidate_level_shift_pp"].abs().max()
            ),
            "regional_composition_preserved": bool(region_sums_hold),
            "feasibility_cap": (
                "not applicable; the multiplicative form cannot reach zero from a "
                "positive input, so no cap exists to bind"
            ),
            "feasibility_capped_elections": [],
            "minimum_predicted_share": float(predictions["layer_pred"].min()),
            "reconciliation_rounds": {
                str(row.election_id): int(row.reconciliation_rounds)
                for row in expansion_audit.itertuples(index=False)
            },
            "outcome_fields_used": [],
        },
        "prospective_feature_contract": {
            "replaces": "blanket zero-fill of every column the target lacked",
            "classes": [
                "REQUIRED_DERIVED",
                "EXPLICIT_ZERO",
                "OUTCOME_ONLY",
                "DIAGNOSTIC_ONLY",
            ],
            "unclassified_column_behaviour": "raises ProspectiveFeatureError",
            "required_derived_without_builder": "hard failure, never a zero",
            "outcome_only_fill": "NaN",
            "model_active_families_recovered": [
                "regional_accent_*",
                "major_party_core_eligible",
                "lineage_identity_*",
                "wasted_vote_resistance",
                "strategic_transfer_confidence",
            ],
            "present_but_dead_check": "audit_required_derived",
            "scored_path_exercised": False,
            "outcome_fields_used": [],
            "classification_record": "docs/PROSPECTIVE_FEATURE_CONTRACT_20260826.md",
        },
        "external_model_derived_input_refusal": {
            "matched_by": "file_name",
            "previous_rule": "directory path prefix",
            "why_it_changed": (
                "the prospective runner copies seed tables into a temporary "
                "directory, where a path-prefix rule stopped matching and the "
                "trace showed the table opened three times"
            ),
            "refused": [
                "assembly_issue_character_overlay.csv",
                "mega_issue_axis.csv",
                "mega_issue_attribution.csv",
            ],
            "retained_and_disclosed": [
                "data/raw/auto_issue_seed/candidate_issue_profile.csv"
            ],
            "trace": "raw_input_read_trace.csv, written where the read happens",
        },
        "calibration_acceptance": {
            "tolerance_share": calibration_guard.CALIBRATION_ABS_TOL,
            "numerical_impact_bound_pp": calibration_guard.NUMERICAL_IMPACT_BOUND_PP,
            "worst_candidate_residual": float(
                calibration_audit["max_candidate_residual"].abs().max()
            ),
            "worst_region_sum_residual": float(
                calibration_audit["max_region_sum_residual"].abs().max()
            ),
            "all_calls_converged": bool(
                calibration_audit["converged"].astype(bool).all()
            ),
            "calls": int(len(calibration_audit)),
            "budget_exhaustion_is_not_a_success_condition": True,
            "plateau_root_cause": "unresolved numerical observation, not a defect",
        },
        "predictive_intervals": {
            "status": intervals["status"],
            "levels": intervals["levels"],
            "post_2022_outcomes_used": False,
        },
        "verification": {
            # this one is genuinely checked above, and raises before reaching here
            "v31_rollback_hash_match": True,
            "v32_prediction_hash": prediction_hash,
            # Not "no material change" - the same bytes. V32 changes the
            # path that builds the target election's features, and the five
            # scored elections do not take it.
            "scored_predictions_byte_identical_to_v31": bool(
                prediction_hash == V31_SHA256
            ),
            # the measured change, not a boolean asserting a direction. Both
            # are expected to be exactly zero here, and they are computed
            # rather than written as zero, so a future run that does move the
            # panel says so in the manifest instead of passing quietly.
            "national_macro_change_vs_v31_pp": float(
                summary["metrics"]["national_equal_election_macro_mae_pp"]
                - V31_NATIONAL_MACRO_PP
            ),
            "regional_macro_change_vs_v31_pp": float(
                summary["metrics"]["regional_equal_election_macro_mae_pp"]
                - V31_REGIONAL_MACRO_PP
            ),
            "predecessor_score_was_not_a_promotion_condition": True,
        },
        "artifacts": [shared._record(path) for path in artifacts],
        "rollback": {
            "version": "v31",
            "prediction_sha256": V31_SHA256,
            "finalization_manifest": (
                "outputs/active_presidential_nested_v31/finalization_manifest.json"
            ),
        },
        "change_policy": (
            "Do not modify V23 through V32 in place; use a new versioned experiment."
        ),
        "known_open": [
            {
                "item": "calibration residual plateau root cause",
                "grade": "unresolved observation, not a defect",
            },
            {
                "item": (
                    "the 1e-11 condition inside _calibrate is unchanged, so "
                    "the three plateau calls still run the full 200 rounds"
                ),
                "grade": "P3 performance; no effect on any published figure",
            },
            {
                "item": (
                    "the scored-path manifest check still inspects a file the "
                    "line above it rewrote; only the read trace is written "
                    "where the read happens"
                ),
                "grade": "P2, carried forward",
            },
            {
                "item": (
                    "the Windows CI runner produces one of two stable scored "
                    "artifacts, differing by 1.388e-13; why there are two "
                    "rather than one is not established"
                ),
                "grade": (
                    "unresolved observation, not a defect; both results sit "
                    "inside the published 1e-12 reproduction tolerance and the "
                    "runner records the magnitude when it recurs"
                ),
            },
        ],
    }
    shared._atomic_json(finalization, ACTIVE_DIR / "finalization_manifest.json")

    metrics = summary["metrics"]
    pointer = {
        "schema": "current_presidential_model_pointer_v1",
        "active_version": "v32",
        "lifecycle_status": "frozen_prospective_feature_contract",
        "canonical_document": "docs/FINAL_MODEL_V32_20260826.md",
        "finalization_manifest": (
            "outputs/active_presidential_nested_v32/finalization_manifest.json"
        ),
        "runner": "scripts/run_active_presidential_model_v32.py",
        "prospective_runner": "scripts/run_prospective_forecast_v32.py",
        "config": "data/config/active_presidential_model_v23.json",
        "version_wrapper": "scripts/run_active_presidential_model_v32.py",
        "base_config_version": "v23",
        "output": "outputs/active_presidential_nested_v32",
        "predecessor": "v31",
        "rollback_pointer": (
            "outputs/active_presidential_nested_v31/finalization_manifest.json"
        ),
        "regional_equal_election_macro_mae_pp": metrics[
            "regional_equal_election_macro_mae_pp"
        ],
        "national_equal_election_macro_mae_pp": metrics[
            "national_equal_election_macro_mae_pp"
        ],
        "winner_accuracy": metrics["winner_accuracy"],
        "prediction_rows": metrics["rows"],
        "prediction_sha256": prediction_hash,
        "predictive_intervals": (
            "outputs/active_presidential_nested_v32/predictive_interval_manifest.json"
        ),
        "predictive_interval_levels": intervals["levels"],
        "predictive_interval_status": intervals["status"],
        "external_neural_model_runtime": False,
        "external_model_derived_inputs": [
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        ],
        "post_2022_outcomes_used": False,
        "prospective_demonstration": _prospective_change(),
    }
    for path in POINTERS:
        shared._atomic_json(pointer, path)
    _refresh_github_baseline(prediction_hash)


def _refresh_github_baseline(prediction_hash: str) -> None:
    """Keep the boundary baseline's pinned hashes in step with the pointer.

    The baseline pins the pointer files by hash, and finalizing rewrites those
    files, so every re-freeze left the boundary audit failing until someone
    refreshed the pins by hand. Doing it here means the only thing that can
    change the pointer is also the thing that records it.
    """

    import hashlib

    baseline_path = ROOT / "docs/GITHUB_BASELINE_V32_20260826.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for pointer_path in POINTERS:
        relative = pointer_path.relative_to(ROOT).as_posix()
        baseline["expected_hashes"][relative] = hashlib.sha256(
            pointer_path.read_bytes()
        ).hexdigest()
    baseline["expected_hashes"][
        "outputs/active_presidential_nested_v32/nested_predictions.csv"
    ] = prediction_hash
    baseline_path.write_bytes(
        (json.dumps(baseline, ensure_ascii=False, indent=2) + chr(10)).encode("utf-8")
    )


if __name__ == "__main__":
    main()
