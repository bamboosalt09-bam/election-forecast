"""Run isolated V24 defect ablations without changing or promoting V23."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import automatic_contest_response  # noqa: E402
from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine import unified_lineage_identity  # noqa: E402
from presidential_issue_engine import v24_defect_ablation  # noqa: E402
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402
from scripts import run_active_presidential_model_v22 as active_v22  # noqa: E402


V23_CONFIG = ROOT / "data" / "config" / "active_presidential_model_v23.json"
CONFIG_DIR = ROOT / "data" / "config" / "v24_defect_ablation"
OUTPUT_DIR = ROOT / "outputs" / "v24_defect_ablation"
V23_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v23"
ASSIGNMENT_ROOT = OUTPUT_DIR / "assignments"
AUTOMATIC_DIR = ROOT / "outputs" / "automatic_controls_v23"
FOOTPRINT_BASE = ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
ALIGNMENT = AUTOMATIC_DIR / "regional_alignment_with_policy.csv"
THIRD_PROFILE = AUTOMATIC_DIR / "third_candidate_profile.csv"
THIRD_PRESSURE = AUTOMATIC_DIR / "third_candidate_pressure.csv"
THIRD_LANDSCAPE = AUTOMATIC_DIR / "candidate_political_landscape.csv"
MEGA_INTENSITY = AUTOMATIC_DIR / "mega_issue_intensity.csv"
MEGA_TAXONOMY = AUTOMATIC_DIR / "mega_issue_taxonomy.csv"
ECONOMIC_ALIGNMENT = AUTOMATIC_DIR / "economic_slot_alignment.csv"
HOUSING_ALIGNMENT = AUTOMATIC_DIR / "housing_slot_alignment.csv"
REGISTRY = AUTOMATIC_DIR / "withdrawal_transfer_registry.csv"
GENERATION = AUTOMATIC_DIR / "election_generation_weights.csv"
HISTORY = ROOT / "presidential_issue_engine" / "fixed_dataset" / "bloc_history_results.csv"
ASSEMBLY = ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv"
PARTY_TRANSITIONS = ROOT / "data" / "raw" / "party_lineage_transitions.csv"
PRESIDENTIAL_RESULTS = (
    ROOT / "presidential_issue_engine" / "fixed_dataset" / "presidential_results_standardized.csv"
)
EXPECTED_V23_HASH = "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b"

VARIANTS = {
    "reference": {},
    "a1_remove_rif": {"remove_rif": True},
    "b1_enforce_config_caps": {"fixed_contest_caps": True},
    "b2_gain_is_cap": {"gain_is_cap": True},
    "c2_remove_preference_floor": {"preference_floor": 0.0},
    "d1_disable_dead_general_identity": {"disable_general_identity": True},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_configs() -> dict[str, Path]:
    base = json.loads(V23_CONFIG.read_text(encoding="utf-8"))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, changes in VARIANTS.items():
        payload = deepcopy(base)
        payload["active"] = False
        payload["policy_version"] = f"v24_defect_ablation_{name}"
        payload["promotion"] = {
            **payload.get("promotion", {}),
            "status": "experimental_not_active",
            "source_active_version": "v23",
            "active_pointer_changed": False,
            "single_change": name,
        }
        if changes.get("remove_rif"):
            payload["predictors"] = [
                predictor for predictor in payload["predictors"] if predictor != "rif"
            ]
        if changes.get("fixed_contest_caps"):
            payload["postprocess"]["contest_regime_cap_policy"] = "fixed_config_caps"
        if changes.get("gain_is_cap"):
            payload["postprocess"].pop("contest_regime_log_shift_cap", None)
            payload["postprocess"].pop("contest_regime_swing_log_shift_cap", None)
            payload["postprocess"]["contest_regime_cap_policy"] = (
                "selected_gain_is_log_shift_cap_and_1.25_times_gain_is_swing_cap"
            )
        if "preference_floor" in changes:
            payload["structural_layers"]["electorate_response"][
                "preference_gain_floor"
            ] = float(changes["preference_floor"])
        if changes.get("disable_general_identity"):
            identity = payload["structural_layers"]["general_regional_identity"]
            identity["enabled"] = False
            identity["gain"] = 0.0
            identity["disabled_reason"] = "superseded_by_unified_exact_lineage_identity"
        path = CONFIG_DIR / f"{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths[name] = path
    return paths


def _runtime_policy_validator(name: str, original_validator):
    changes = VARIANTS[name]

    def validate(policy: dict[str, object]) -> dict[str, object]:
        original = deepcopy(policy)
        normalized = deepcopy(policy)
        normalized["active"] = True
        if changes.get("gain_is_cap"):
            normalized["postprocess"]["contest_regime_log_shift_cap"] = 0.40
            normalized["postprocess"]["contest_regime_swing_log_shift_cap"] = 0.50
        if "preference_floor" in changes:
            normalized["structural_layers"]["electorate_response"][
                "preference_gain_floor"
            ] = 0.04
        if changes.get("disable_general_identity"):
            identity = normalized["structural_layers"]["general_regional_identity"]
            identity["enabled"] = True
            identity["gain"] = 0.10
        original_validator(normalized)

        # Compatibility fields are derived, not stored, for the b2 execution.
        if changes.get("gain_is_cap"):
            gain = float(original["postprocess"]["contest_regime_expansion_gain"])
            original["postprocess"]["contest_regime_log_shift_cap"] = gain
            original["postprocess"]["contest_regime_swing_log_shift_cap"] = 1.25 * gain
        return original

    return validate


def _static_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(HISTORY, encoding="utf-8-sig")
    assembly = pd.read_csv(ASSEMBLY, encoding="utf-8-sig")
    events = unified_lineage_identity.build_exact_lineage_events(history, assembly)
    transitions = pd.read_csv(PARTY_TRANSITIONS, encoding="utf-8-sig")
    candidates = (
        pd.read_csv(
            PRESIDENTIAL_RESULTS,
            encoding="utf-8-sig",
            usecols=["election_id", "slot", "candidate_name", "party_name"],
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return history, events, transitions, candidates


def run_variant(
    name: str,
    config_path: Path,
    *,
    events: pd.DataFrame,
    transitions: pd.DataFrame,
    candidate_parties: pd.DataFrame,
) -> Path:
    changes = VARIANTS[name]
    original_apply = contest_regime.apply_contest_regime_response
    response_audit: dict[str, pd.DataFrame] = {}
    lineage_reliability: list[pd.DataFrame] = []

    def automatic_apply(
        frame,
        regimes,
        *,
        prediction_column,
        slot_column="source_slot",
        output_column=None,
        expansion_gain=0.50,
        log_shift_cap=0.40,
        critical_elasticity=0.75,
        swing_elasticity=1.25,
        swing_log_shift_cap=0.50,
    ):
        del expansion_gain
        if changes.get("fixed_contest_caps"):
            result, audit_frame = (
                v24_defect_ablation.apply_prior_selected_response_with_fixed_caps(
                    frame,
                    regimes,
                    prediction_column=prediction_column,
                    apply_response=original_apply,
                    election_order=active.nested.ELECTIONS,
                    slot_column=slot_column,
                    output_column=output_column,
                    critical_elasticity=critical_elasticity,
                    swing_elasticity=swing_elasticity,
                    log_shift_cap=log_shift_cap,
                    swing_log_shift_cap=swing_log_shift_cap,
                )
            )
        else:
            result, audit_frame = automatic_contest_response.apply_prior_selected_contest_response(
                frame,
                regimes,
                prediction_column=prediction_column,
                apply_response=original_apply,
                election_order=active.nested.ELECTIONS,
                slot_column=slot_column,
                output_column=output_column,
                critical_elasticity=critical_elasticity,
                swing_elasticity=swing_elasticity,
            )
        response_audit["audit"] = audit_frame
        return result

    def unified_apply(
        frame,
        candidate_events,
        candidate_regional_base,
        alignment,
        *,
        prediction_column,
        gain,
        shift_cap=0.08,
        half_life_years=12.0,
        prior_strength=1.5,
    ):
        adjusted, audit_frame, reliability = unified_lineage_identity.apply_unified_lineage_routing(
            frame,
            candidate_events,
            candidate_regional_base,
            alignment,
            candidate_parties,
            transitions,
            prediction_column=prediction_column,
            gain=gain,
            shift_cap=shift_cap,
            half_life_years=half_life_years,
            prior_strength=prior_strength,
            include_direct_lineage_score=True,
            direct_lineage_scope="non_major",
        )
        lineage_reliability.append(reliability)
        return adjusted, audit_frame

    def unified_attach_prior(frame, ignored_history, election_order):
        del ignored_history
        return unified_lineage_identity.attach_lineage_projected_prior(
            frame, events, candidate_parties, election_order
        )

    def no_general_events(ignored_history):
        del ignored_history
        return pd.DataFrame()

    def no_general_apply(frame, *args, **kwargs):
        del args, kwargs
        return frame, pd.DataFrame()

    original_validator = active.validate_policy
    base_predictors = tuple(active.nested.BASE_PREDICTORS)
    variant_predictors = tuple(
        predictor for predictor in base_predictors if not (
            changes.get("remove_rif") and predictor == "rif"
        )
    )
    variant_map = dict(active.nested.VARIANTS)
    variant_map[active.EXPECTED_VARIANT] = variant_predictors
    engines = {active.nested.engine, active.assignment_builder.engine}
    attributes: list[tuple[object, str, object]] = [
        (active, "validate_policy", _runtime_policy_validator(name, original_validator)),
        (active.nested, "BASE_PREDICTORS", variant_predictors),
        (active.nested, "VARIANTS", variant_map),
        (active.contest_regime, "apply_contest_regime_response", automatic_apply),
        (active.nested.engine, "attach_bloc_prior", unified_attach_prior),
        (active.assignment_builder.engine, "attach_bloc_prior", unified_attach_prior),
        (active.chungcheong_identity, "build_identity_events", lambda _: events),
        (active.chungcheong_identity, "apply_identity_routing", unified_apply),
        (active.regional_identity, "build_distinctiveness_events", no_general_events),
        (active.regional_identity, "apply_regional_identity_routing", no_general_apply),
        (active, "MEGA_ISSUE_INTENSITY", MEGA_INTENSITY),
    ]
    for engine in engines:
        attributes.extend(
            [
                (engine, "WITHDRAWAL_TRANSFER_REGISTRY", str(REGISTRY)),
                (engine, "ELECTION_GENERATION_WEIGHTS", str(GENERATION)),
                (engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(MEGA_INTENSITY)),
                (engine, "MEGA_ISSUE_TAXONOMY", str(MEGA_TAXONOMY)),
                (engine, "ECONOMIC_SLOT_ALIGNMENT", str(ECONOMIC_ALIGNMENT)),
                (engine, "HOUSING_SLOT_ALIGNMENT", str(HOUSING_ALIGNMENT)),
                (engine, "CANDIDATE_POLITICAL_LANDSCAPE", str(THIRD_LANDSCAPE)),
            ]
        )

    run_dir = OUTPUT_DIR / name / "active_run"
    assignment_dir = ASSIGNMENT_ROOT / name
    with patching.patched(attributes):
        clean._run_variant(
            name,
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=ALIGNMENT,
            third_profile_path=THIRD_PROFILE,
            third_pressure_path=THIRD_PRESSURE,
            config_path=config_path,
            run_dir_override=run_dir,
            assignment_dir_override=assignment_dir,
            regenerate_issue_seeds_enabled=False,
            output_root=OUTPUT_DIR,
        )

    response_audit.get("audit", pd.DataFrame()).to_csv(
        run_dir / "automatic_response_gain_audit.csv", index=False, encoding="utf-8-sig"
    )
    reliability = pd.concat(lineage_reliability, ignore_index=True) if lineage_reliability else pd.DataFrame()
    reliability.to_csv(
        run_dir / "lineage_type_reliability_audit.csv", index=False, encoding="utf-8-sig"
    )
    return run_dir


def _variant_metrics(path: Path) -> dict[str, object]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    by_election = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
    fold_audit = pd.read_csv(path / "fold_audit.csv", encoding="utf-8-sig")
    predictors = set()
    for value in fold_audit["predictors"].astype(str):
        predictors.update(item for item in value.split("|") if item)
    pit_pass = bool(
        fold_audit["target_excluded_from_fit"].astype(bool).all()
        and fold_audit["consistent_scored_denominator"].astype(bool).all()
        and not (path / "input_manifest.csv").read_text(encoding="utf-8-sig").__contains__("pres_2025")
    )
    slot_pass = bool(
        not fold_audit["old_slot_predictors_used"].astype(bool).any()
        and not ({"slot_A", "slot_B", "slotA_prior", "slotB_prior"} & predictors)
    )
    return {
        "regional_equal_election_macro_mae_pp": float(
            summary["metrics"]["regional_equal_election_macro_mae_pp"]
        ),
        "national_equal_election_macro_mae_pp": float(
            summary["metrics"]["national_equal_election_macro_mae_pp"]
        ),
        "winner_accuracy": float(summary["metrics"]["winner_accuracy"]),
        "prediction_sha256": _sha256(path / "nested_predictions.csv"),
        "pit_audit_pass": pit_pass,
        "slot_leakage_audit_pass": slot_pass,
        "by_election": [
            {
                "election_id": str(row.election_id),
                "regional_weighted_mae_pp": float(row.regional_weighted_mae_pp),
                "national_candidate_mae_pp": float(row.national_candidate_mae_pp),
            }
            for row in by_election.itertuples(index=False)
        ],
    }


def main() -> None:
    configs = build_configs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, events, transitions, candidate_parties = _static_inputs()
    results: dict[str, dict[str, object]] = {}
    for name in VARIANTS:
        run_dir = OUTPUT_DIR / name / "active_run"
        complete = all(
            (run_dir / filename).exists()
            for filename in (
                "summary.json",
                "nested_predictions.csv",
                "by_election.csv",
                "fold_audit.csv",
                "input_manifest.csv",
                "automatic_response_gain_audit.csv",
            )
        )
        if complete:
            print(f"[v24 ablation] reusing complete {name}", flush=True)
        else:
            print(f"[v24 ablation] running {name}", flush=True)
            run_dir = run_variant(
                name,
                configs[name],
                events=events,
                transitions=transitions,
                candidate_parties=candidate_parties,
            )
        results[name] = _variant_metrics(run_dir)
        if name == "reference" and results[name]["prediction_sha256"] != EXPECTED_V23_HASH:
            raise RuntimeError(
                "V24 harness failed exact V23 reproduction: "
                f"{results[name]['prediction_sha256']} != {EXPECTED_V23_HASH}"
            )

    reference = results["reference"]
    modifications: dict[str, object] = {}
    for name, metrics in results.items():
        if name == "reference":
            continue
        modifications[name] = {
            **metrics,
            "delta_regional_mae_pp": float(
                metrics["regional_equal_election_macro_mae_pp"]
                - reference["regional_equal_election_macro_mae_pp"]
            ),
            "delta_national_mae_pp": float(
                metrics["national_equal_election_macro_mae_pp"]
                - reference["national_equal_election_macro_mae_pp"]
            ),
            "delta_winner_accuracy": float(
                metrics["winner_accuracy"] - reference["winner_accuracy"]
            ),
            "numerically_invariant": bool(
                metrics["prediction_sha256"] == reference["prediction_sha256"]
            ),
        }

    decision = {
        "schema": "v24_defect_ablation_decision_v1",
        "status": "measurement_only_not_promoted",
        "active_pointer_changed": False,
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "reference": reference,
        "a2_dated_regional_issue_sensitivity": {
            "status": "not_implemented_missing_source",
            "reason": (
                "data/raw/region_issue_sensitivity.csv has only a header; the dated template "
                "is not observational evidence and the curated source has no available_date"
            ),
        },
        "c1_documentation_matches_active_floor": {
            "status": "implemented_documentation_only",
            "numerically_invariant": True,
        },
        "individual_modifications": modifications,
        "numerically_invariant_modifications": [
            name for name, row in modifications.items() if row["numerically_invariant"]
        ] + ["c1_documentation_matches_active_floor"],
        "numerically_changing_modifications": [
            name for name, row in modifications.items() if not row["numerically_invariant"]
        ],
        "combination_selected": False,
        "promotion_decision": "human_review_required",
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "variant": name,
                "regional_equal_election_macro_mae_pp": metrics[
                    "regional_equal_election_macro_mae_pp"
                ],
                "national_equal_election_macro_mae_pp": metrics[
                    "national_equal_election_macro_mae_pp"
                ],
                "winner_accuracy": metrics["winner_accuracy"],
                "prediction_sha256": metrics["prediction_sha256"],
                "pit_audit_pass": metrics["pit_audit_pass"],
                "slot_leakage_audit_pass": metrics["slot_leakage_audit_pass"],
            }
        )
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
