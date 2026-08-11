"""Strict nested ablation of the V22 automatic control compilers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import automatic_contest_response  # noqa: E402
from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine import unified_lineage_identity  # noqa: E402
from scripts import build_automatic_controls_v22 as builder  # noqa: E402
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_controls_v22_ablation"
BASELINE_DIR = ROOT / "outputs" / "active_presidential_nested_v21"
AUTOMATIC_DIR = ROOT / "outputs" / "automatic_controls_v22"
CONFIG = ROOT / "data" / "config" / "active_presidential_model_v21.json"
FOOTPRINT_BASE = ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
BASE_ALIGNMENT = ROOT / "outputs" / "automatic_regional_party_alignment_v11" / "automatic_alignment.csv"
BASE_THIRD_PROFILE = ROOT / "outputs" / "automatic_third_character_v20b" / "third_candidate_profile.csv"
BASE_THIRD_PRESSURE = ROOT / "data" / "raw" / "third_candidate_pressure.csv"
BASE_TRANSITIONS = ROOT / "data" / "raw" / "party_lineage_transitions.csv"
ASSEMBLY = pd.read_csv(
    ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv",
    encoding="utf-8-sig",
)
CANDIDATE_PARTIES = (
    pd.read_csv(
        ROOT / "presidential_issue_engine" / "fixed_dataset" / "presidential_results_standardized.csv",
        encoding="utf-8-sig",
        usecols=["election_id", "slot", "candidate_name", "party_name"],
    )
    .drop_duplicates()
    .reset_index(drop=True)
)


VARIANTS = {
    "policy_commitment_only": {"policy": True},
    "automatic_mega_only": {"mega": True},
    "automatic_responsibility_only": {"responsibility": True},
    "automatic_generation_only": {"generation": True},
    "automatic_third_profile_only": {"third_profile": True},
    "automatic_third_pressure_only": {"third_pressure": True},
    "automatic_third_context_only": {"third": True},
    "automatic_genealogy_retention_only": {"genealogy": True},
    "priority_policy_mega_pressure": {
        "policy": True,
        "mega": True,
        "third_pressure": True,
    },
    "priority_policy_mega_third": {"policy": True, "mega": True, "third": True},
    "priority_policy_mega_third_generation": {
        "policy": True,
        "mega": True,
        "third": True,
        "generation": True,
    },
    "priority_policy_mega_third_responsibility": {
        "policy": True,
        "mega": True,
        "third": True,
        "responsibility": True,
    },
    "priority_policy_mega_third_generation_responsibility": {
        "policy": True,
        "mega": True,
        "third": True,
        "generation": True,
        "responsibility": True,
    },
    "full_automatic_controls": {
        "policy": True,
        "mega": True,
        "responsibility": True,
        "generation": True,
        "third": True,
        "genealogy": True,
    },
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _run_variant(label: str, switches: dict[str, bool], exact_events: pd.DataFrame) -> Path:
    variant_root = OUTPUT_DIR / label
    alignment_path = (
        AUTOMATIC_DIR / "regional_alignment_with_policy.csv"
        if switches.get("policy")
        else BASE_ALIGNMENT
    )
    third_profile_path = (
        AUTOMATIC_DIR / "third_candidate_profile.csv"
        if switches.get("third") or switches.get("third_profile")
        else BASE_THIRD_PROFILE
    )
    third_pressure_path = (
        AUTOMATIC_DIR / "third_candidate_pressure.csv"
        if switches.get("third")
        else (
            AUTOMATIC_DIR / "third_candidate_pressure_active_context.csv"
            if switches.get("third_pressure")
            else BASE_THIRD_PRESSURE
        )
    )
    transitions = _read(
        AUTOMATIC_DIR / "party_lineage_transitions_behavioral.csv"
        if switches.get("genealogy")
        else BASE_TRANSITIONS
    )
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
        del expansion_gain, log_shift_cap, swing_log_shift_cap
        result, audit = automatic_contest_response.apply_prior_selected_contest_response(
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
        response_audit["audit"] = audit
        return result

    def unified_apply(
        frame,
        events,
        candidate_regional_base,
        alignment,
        *,
        prediction_column,
        gain,
        shift_cap=0.08,
        half_life_years=12.0,
        prior_strength=1.5,
    ):
        adjusted, audit, reliability = unified_lineage_identity.apply_unified_lineage_routing(
            frame,
            events,
            candidate_regional_base,
            alignment,
            CANDIDATE_PARTIES,
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
        return adjusted, audit

    def unified_attach_prior(frame, ignored_history, election_order):
        del ignored_history
        return unified_lineage_identity.attach_lineage_projected_prior(
            frame, exact_events, CANDIDATE_PARTIES, election_order
        )

    def no_general_events(ignored_history):
        del ignored_history
        return pd.DataFrame()

    def no_general_apply(
        frame,
        events,
        candidate_regional_base,
        *,
        prediction_column,
        gain,
        shift_cap=0.04,
        half_life_years=12.0,
        prior_strength=1.5,
    ):
        del events, candidate_regional_base, prediction_column, gain
        del shift_cap, half_life_years, prior_strength
        return frame, pd.DataFrame()

    engines = {active.nested.engine, active.assignment_builder.engine}
    attributes: list[tuple[object, str, object]] = [
        (active.contest_regime, "apply_contest_regime_response", automatic_apply),
        (active.nested.engine, "attach_bloc_prior", unified_attach_prior),
        (active.assignment_builder.engine, "attach_bloc_prior", unified_attach_prior),
        (active.chungcheong_identity, "build_identity_events", lambda _: exact_events),
        (active.chungcheong_identity, "apply_identity_routing", unified_apply),
        (active.regional_identity, "build_distinctiveness_events", no_general_events),
        (active.regional_identity, "apply_regional_identity_routing", no_general_apply),
    ]
    if switches.get("mega"):
        attributes.append((active, "MEGA_ISSUE_INTENSITY", AUTOMATIC_DIR / "mega_issue_intensity.csv"))
    for engine in engines:
        if switches.get("mega"):
            attributes.extend(
                [
                    (engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(AUTOMATIC_DIR / "mega_issue_intensity.csv")),
                    (engine, "MEGA_ISSUE_TAXONOMY", str(AUTOMATIC_DIR / "mega_issue_taxonomy.csv")),
                ]
            )
        if switches.get("responsibility"):
            attributes.extend(
                [
                    (engine, "ECONOMIC_SLOT_ALIGNMENT", str(AUTOMATIC_DIR / "economic_slot_alignment.csv")),
                    (engine, "HOUSING_SLOT_ALIGNMENT", str(AUTOMATIC_DIR / "housing_slot_alignment.csv")),
                ]
            )
        if switches.get("generation"):
            attributes.append(
                (engine, "ELECTION_GENERATION_WEIGHTS", str(AUTOMATIC_DIR / "election_generation_weights.csv"))
            )
        if switches.get("third") or switches.get("third_profile"):
            attributes.append(
                (engine, "CANDIDATE_POLITICAL_LANDSCAPE", str(AUTOMATIC_DIR / "candidate_political_landscape.csv"))
            )

    with patching.patched(attributes):
        run_dir = clean._run_variant(
            label,
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=alignment_path,
            third_profile_path=third_profile_path,
            third_pressure_path=third_pressure_path,
            config_path=CONFIG,
            run_dir_override=variant_root / "active_run",
            assignment_dir_override=variant_root / "slot_assignment",
            regenerate_issue_seeds_enabled=False,
            output_root=OUTPUT_DIR,
        )
    response_audit.get("audit", pd.DataFrame()).to_csv(
        variant_root / "automatic_response_gain_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability = pd.concat(lineage_reliability, ignore_index=True) if lineage_reliability else pd.DataFrame()
    reliability.to_csv(
        variant_root / "lineage_type_reliability_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return run_dir


def main() -> None:
    builder.build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = _read(active.nested.base_eval.HISTORY_PATH)
    exact_events = unified_lineage_identity.build_exact_lineage_events(history, ASSEMBLY)
    requested = {
        value.strip()
        for value in os.environ.get("V22_VARIANTS", "").split(",")
        if value.strip()
    }
    selected_variants = (
        {label: switches for label, switches in VARIANTS.items() if label in requested}
        if requested
        else VARIANTS
    )
    unknown = requested - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown V22 variants: {sorted(unknown)}")
    runs = {
        label: _run_variant(label, switches, exact_events)
        for label, switches in selected_variants.items()
    }

    for label in VARIANTS:
        run_dir = OUTPUT_DIR / label / "active_run"
        if label not in runs and (run_dir / "summary.json").exists():
            runs[label] = run_dir

    baseline_metrics = _metrics(BASELINE_DIR)
    baseline_by = _read(BASELINE_DIR / "by_election.csv")
    baseline_lookup = baseline_by.set_index("election_id")["regional_weighted_mae_pp"]
    summary_rows = [{"variant_label": "active_v21", **baseline_metrics}]
    by_frames = [baseline_by.assign(variant_label="active_v21")]
    national_frames = [_read(BASELINE_DIR / "national_predictions.csv").assign(variant_label="active_v21")]
    decision_rows: list[dict[str, object]] = []
    priority_elections = {"pres_2002", "pres_2007", "pres_2017"}
    baseline_priority = float(
        baseline_by.loc[
            baseline_by["election_id"].isin(priority_elections),
            "regional_weighted_mae_pp",
        ].mean()
    )
    for label, run_dir in runs.items():
        metrics = _metrics(run_dir)
        summary_rows.append({"variant_label": label, **metrics})
        by = _read(run_dir / "by_election.csv").assign(variant_label=label)
        by["regional_change_vs_v21_pp"] = (
            by["regional_weighted_mae_pp"] - by["election_id"].map(baseline_lookup)
        )
        by_frames.append(by)
        national_frames.append(
            _read(run_dir / "national_predictions.csv").assign(variant_label=label)
        )
        priority_mae = float(
            by.loc[
                by["election_id"].isin(priority_elections),
                "regional_weighted_mae_pp",
            ].mean()
        )
        regional_change = float(
            metrics["regional_equal_election_macro_mae_pp"]
            - baseline_metrics["regional_equal_election_macro_mae_pp"]
        )
        national_change = float(
            metrics["national_equal_election_macro_mae_pp"]
            - baseline_metrics["national_equal_election_macro_mae_pp"]
        )
        max_regression = float(by["regional_change_vs_v21_pp"].max())
        gate = bool(
            metrics["winner_accuracy"] >= baseline_metrics["winner_accuracy"]
            and regional_change <= 0.10
            and national_change <= 0.10
            and max_regression <= 0.25
        )
        decision_rows.append(
            {
                "variant_label": label,
                "regional_change_vs_v21_pp": regional_change,
                "national_change_vs_v21_pp": national_change,
                "maximum_election_regression_pp": max_regression,
                "priority_2002_2007_2017_mae_pp": priority_mae,
                "priority_change_vs_v21_pp": priority_mae - baseline_priority,
                "winner_accuracy": metrics["winner_accuracy"],
                "equivalence_gate": "pass" if gate else "fail",
            }
        )

    summary = pd.DataFrame(summary_rows)
    by_election = pd.concat(by_frames, ignore_index=True)
    national = pd.concat(national_frames, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)
    passing = decisions.loc[decisions["equivalence_gate"].eq("pass")].sort_values(
        ["priority_2002_2007_2017_mae_pp", "regional_change_vs_v21_pp"]
    )
    candidate = str(passing.iloc[0]["variant_label"]) if not passing.empty else "none"
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    national.to_csv(OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(OUTPUT_DIR / "decision_table.csv", index=False, encoding="utf-8-sig")
    decision = {
        "experiment": "automatic_controls_v22",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "active_model_changed": False,
        "selection_is_development_outcome_aware": True,
        "equivalence_gate": {
            "regional_degradation_cap_pp": 0.10,
            "national_degradation_cap_pp": 0.10,
            "maximum_election_regression_cap_pp": 0.25,
            "winner_accuracy_no_regression": True,
        },
        "best_passing_shadow_candidate": candidate,
        "promotion_status": "not_promoted_pending_layer_review",
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print()
    print(decisions.to_string(index=False))
    print()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
