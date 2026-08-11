"""Run active V21 with one point-in-time exact-party lineage ledger."""

from __future__ import annotations

import hashlib
import json
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
from scripts import build_automatic_third_character_v20b as character_v20b  # noqa: E402
from scripts import build_election_derived_third_candidate_profile_v14 as profile_v14  # noqa: E402
from scripts import build_election_derived_third_candidate_profile_v14b as profile_v14b  # noqa: E402
from scripts import build_election_derived_third_candidate_profile_v15 as profile_v15  # noqa: E402
from scripts import build_unified_exact_lineage_v21 as lineage_builder  # noqa: E402
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


CONFIG = ROOT / "data" / "config" / "active_presidential_model_v21.json"
OUTPUT_DIR = ROOT / "outputs" / "active_presidential_nested_v21"
ASSIGNMENT_DIR = ROOT / "outputs" / "preliminary_slot_assignment_v21"
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)
ALIGNMENT = (
    ROOT
    / "outputs"
    / "automatic_regional_party_alignment_v11"
    / "automatic_alignment.csv"
)
THIRD_PROFILE = character_v20b.OUTPUT_DIR / "third_candidate_profile.csv"
HISTORY = (
    ROOT
    / "presidential_issue_engine"
    / "fixed_dataset"
    / "bloc_history_results.csv"
)
ASSEMBLY = (
    ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv"
)
PARTY_TRANSITIONS = ROOT / "data" / "raw" / "party_lineage_transitions.csv"
PRESIDENTIAL_RESULTS = (
    ROOT
    / "presidential_issue_engine"
    / "fixed_dataset"
    / "presidential_results_standardized.csv"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _augment_input_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "input_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    extra = pd.DataFrame(
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (ASSEMBLY, PARTY_TRANSITIONS)
        ]
    )
    manifest = (
        pd.concat([manifest, extra], ignore_index=True)
        .drop_duplicates("path", keep="last")
        .sort_values("path")
        .reset_index(drop=True)
    )
    active._atomic_csv(manifest, manifest_path)


def main() -> None:
    profile_v14.main()
    profile_v14b.main()
    profile_v15.main()
    character_v20b.main()
    lineage_builder.main()

    history = pd.read_csv(HISTORY, encoding="utf-8-sig")
    assembly = pd.read_csv(ASSEMBLY, encoding="utf-8-sig")
    exact_events = unified_lineage_identity.build_exact_lineage_events(
        history, assembly
    )
    transitions = pd.read_csv(PARTY_TRANSITIONS, encoding="utf-8-sig")
    candidate_parties = (
        pd.read_csv(
            PRESIDENTIAL_RESULTS,
            encoding="utf-8-sig",
            usecols=["election_id", "slot", "candidate_name", "party_name"],
        )
        .drop_duplicates()
        .reset_index(drop=True)
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
        adjusted, audit, reliability = (
            unified_lineage_identity.apply_unified_lineage_routing(
                frame,
                events,
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
        )
        lineage_reliability.append(reliability)
        return adjusted, audit

    def unified_attach_prior(frame, ignored_history, election_order):
        del ignored_history
        return unified_lineage_identity.attach_lineage_projected_prior(
            frame,
            exact_events,
            candidate_parties,
            election_order,
        )

    def no_general_events(ignored_history: pd.DataFrame) -> pd.DataFrame:
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

    with patching.patched(
        [
            (active.contest_regime, "apply_contest_regime_response", automatic_apply),
            (active.nested.engine, "attach_bloc_prior", unified_attach_prior),
            (active.assignment_builder.engine, "attach_bloc_prior", unified_attach_prior),
            (active.chungcheong_identity, "build_identity_events", lambda _: exact_events),
            (active.chungcheong_identity, "apply_identity_routing", unified_apply),
            (active.regional_identity, "build_distinctiveness_events", no_general_events),
            (active.regional_identity, "apply_regional_identity_routing", no_general_apply),
        ]
    ):
        run_dir = clean._run_variant(
            "active_v21",
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=ALIGNMENT,
            third_profile_path=THIRD_PROFILE,
            config_path=CONFIG,
            run_dir_override=OUTPUT_DIR,
            assignment_dir_override=ASSIGNMENT_DIR,
            regenerate_issue_seeds_enabled=False,
            output_root=ROOT / "outputs",
        )

    response_audit["audit"].to_csv(
        run_dir / "automatic_response_gain_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability = (
        pd.concat(lineage_reliability, ignore_index=True)
        if lineage_reliability
        else pd.DataFrame()
    )
    reliability.to_csv(
        run_dir / "lineage_type_reliability_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _augment_input_manifest(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    promotion = {
        "schema": "active_presidential_model_promotion_v21",
        "status": "active",
        "predecessor": "active_v20",
        "experiment_lineage": [
            "active_v20",
            "unified_exact_lineage_v21",
            "dated_party_genealogy_v1",
        ],
        "methodology_priority": (
            "consistent exact genealogy over small retrospective MAE advantage"
        ),
        "same_formula_all_regions": True,
        "manual_candidate_alignment_rows_used": False,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used_by_new_layers": [],
        "performance_improvement_claim": False,
        "config_sha256": _sha256(CONFIG),
        "lineage_events_sha256": _sha256(
            lineage_builder.OUTPUT_DIR / "exact_lineage_events.csv"
        ),
        "party_transitions_sha256": _sha256(PARTY_TRANSITIONS),
        "predictions_sha256": _sha256(run_dir / "nested_predictions.csv"),
        "metrics": summary["metrics"],
        "rollback_checkpoint": "backups/model_checkpoints/20260802_pre_active_v21",
    }
    (run_dir / "promotion_manifest.json").write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(promotion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
