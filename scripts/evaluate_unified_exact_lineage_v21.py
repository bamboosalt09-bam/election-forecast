"""Strict nested experiment for one exact-lineage regional identity layer."""

from __future__ import annotations

import json
import hashlib
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
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "unified_exact_lineage_v21_ablation"
REFERENCE_RUN = ROOT / "outputs" / "active_presidential_nested_v20"
CONFIG = ROOT / "data" / "config" / "active_presidential_model_v20.json"
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)
ALIGNMENT = (
    ROOT
    / "outputs"
    / "automatic_regional_party_alignment_v11"
    / "automatic_alignment.csv"
)
THIRD_PROFILE = (
    ROOT
    / "outputs"
    / "automatic_third_character_v20b"
    / "third_candidate_profile.csv"
)
ASSEMBLY = pd.read_csv(
    ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv",
    encoding="utf-8-sig",
)
PARTY_TRANSITIONS = pd.read_csv(
    ROOT / "data" / "raw" / "party_lineage_transitions.csv",
    encoding="utf-8-sig",
)
CANDIDATE_PARTIES = (
    pd.read_csv(
        ROOT
        / "presidential_issue_engine"
        / "fixed_dataset"
        / "presidential_results_standardized.csv",
        encoding="utf-8-sig",
        usecols=["election_id", "slot", "candidate_name", "party_name"],
    )
    .drop_duplicates()
    .reset_index(drop=True)
)


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _augment_input_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "input_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    extra = pd.DataFrame(
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (
                ROOT
                / "data"
                / "raw"
                / "official_sources"
                / "nec_assembly_district_history.csv",
                ROOT / "data" / "raw" / "party_lineage_transitions.csv",
            )
        ]
    )
    manifest = (
        pd.concat([manifest, extra], ignore_index=True)
        .drop_duplicates("path", keep="last")
        .sort_values("path")
        .reset_index(drop=True)
    )
    active._atomic_csv(manifest, manifest_path)


def exact_events(history: pd.DataFrame) -> pd.DataFrame:
    return unified_lineage_identity.build_exact_lineage_events(history, ASSEMBLY)


def main() -> None:
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
        adjusted, audit, reliability = (
            unified_lineage_identity.apply_unified_lineage_routing(
                frame,
                events,
                candidate_regional_base,
                alignment,
                CANDIDATE_PARTIES,
                PARTY_TRANSITIONS,
                prediction_column=prediction_column,
                gain=gain,
                shift_cap=shift_cap,
                half_life_years=half_life_years,
                prior_strength=prior_strength,
                include_direct_lineage_score=True,
                direct_lineage_scope="non_major",
            )
        )
        lineage_reliability.append(reliability)
        return adjusted, audit

    def unified_attach_prior(frame, history, election_order):
        del history
        return unified_lineage_identity.attach_lineage_projected_prior(
            frame,
            exact_events_source,
            CANDIDATE_PARTIES,
            election_order,
        )

    def no_general_events(history: pd.DataFrame) -> pd.DataFrame:
        del history
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

    history_source = pd.read_csv(
        active.nested.base_eval.HISTORY_PATH, encoding="utf-8-sig"
    )
    exact_events_source = exact_events(history_source)

    with patching.patched(
        [
            (active.contest_regime, "apply_contest_regime_response", automatic_apply),
            (active.nested.engine, "attach_bloc_prior", unified_attach_prior),
            (
                active.assignment_builder.engine,
                "attach_bloc_prior",
                unified_attach_prior,
            ),
            (active.chungcheong_identity, "build_identity_events", exact_events),
            (active.chungcheong_identity, "apply_identity_routing", unified_apply),
            (active.regional_identity, "build_distinctiveness_events", no_general_events),
            (active.regional_identity, "apply_regional_identity_routing", no_general_apply),
        ]
    ):
        run = clean._run_variant(
            "unified_exact_lineage_v21",
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=ALIGNMENT,
            third_profile_path=THIRD_PROFILE,
            config_path=CONFIG,
            run_dir_override=OUTPUT_DIR / "active_run",
            assignment_dir_override=OUTPUT_DIR / "slot_assignment",
            regenerate_issue_seeds_enabled=False,
            output_root=OUTPUT_DIR,
        )

    response_audit["audit"].to_csv(
        OUTPUT_DIR / "automatic_response_gain_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability = (
        pd.concat(lineage_reliability, ignore_index=True)
        if lineage_reliability
        else pd.DataFrame()
    )
    reliability.to_csv(
        OUTPUT_DIR / "lineage_type_reliability_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _augment_input_manifest(run)
    summary = pd.DataFrame(
        [
            {"variant_label": "v20_reference", **_metrics(REFERENCE_RUN)},
            {"variant_label": "unified_exact_lineage_v21", **_metrics(run)},
        ]
    )
    reference_by = pd.read_csv(REFERENCE_RUN / "by_election.csv", encoding="utf-8-sig")
    reference_by["variant_label"] = "v20_reference"
    candidate_by = pd.read_csv(run / "by_election.csv", encoding="utf-8-sig")
    candidate_by["variant_label"] = "unified_exact_lineage_v21"
    by_election = pd.concat([reference_by, candidate_by], ignore_index=True)
    reference = reference_by.set_index("election_id")["regional_weighted_mae_pp"]
    by_election["regional_change_vs_v20_pp"] = (
        by_election["regional_weighted_mae_pp"]
        - by_election["election_id"].map(reference)
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    candidate = summary.loc[
        summary["variant_label"].eq("unified_exact_lineage_v21")
    ].iloc[0]
    reference_metrics = summary.loc[
        summary["variant_label"].eq("v20_reference")
    ].iloc[0]
    changes = by_election.loc[
        by_election["variant_label"].eq("unified_exact_lineage_v21")
    ]
    regional_change = float(
        candidate["regional_equal_election_macro_mae_pp"]
        - reference_metrics["regional_equal_election_macro_mae_pp"]
    )
    national_change = float(
        candidate["national_equal_election_macro_mae_pp"]
        - reference_metrics["national_equal_election_macro_mae_pp"]
    )
    maximum_regression = float(changes["regional_change_vs_v20_pp"].max())
    # Exact genealogy is a measurement correction, not an outcome-selected
    # performance feature. These bounds reject only a clearly broken runtime;
    # they do not require the corrected representation to beat V20.
    integration_safe = bool(
        float(candidate["winner_accuracy"])
        >= float(reference_metrics["winner_accuracy"])
        and regional_change <= 0.50
        and national_change <= 0.50
        and maximum_regression <= 0.75
    )
    decision = {
        "experiment": "unified_exact_lineage_v21",
        "strict_nested": True,
        "reference": "active_v20_v10_successor",
        "changed_system": "unified_exact_lineage_prior_and_identity",
        "ridge_partisan_prior_source": "final_bloc_projection_of_same_exact_lineage_ledger",
        "exact_party_name_preserved_before_lineage": True,
        "assembly_constituency_exact_party_restored": True,
        "same_formula_all_regions": True,
        "special_chungcheong_estimator_removed": True,
        "manual_candidate_alignment_rows_used": False,
        "direct_lineage_score_scope_after_ridge": "non_major_only",
        "candidate_ballot_reliability_source": "prior same-date direct-party agreement",
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "regional_mae_pp": float(candidate["regional_equal_election_macro_mae_pp"]),
        "national_mae_pp": float(candidate["national_equal_election_macro_mae_pp"]),
        "regional_change_vs_v20_pp": regional_change,
        "national_change_vs_v20_pp": national_change,
        "maximum_election_regression_pp": maximum_regression,
        "methodology_gate": "pass",
        "performance_tradeoff_policy": (
            "adopt_consistent_exact_genealogy_unless_integration_is_broken"
        ),
        "integration_safety_gate": "pass" if integration_safe else "fail",
        "promotion_candidate": integration_safe,
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print()
    print(by_election.to_string(index=False))
    print()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
