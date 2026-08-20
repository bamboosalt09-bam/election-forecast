"""Run corrected V25: V23 runtime repairs plus V24 structural extensions.

V24 accidentally called the pre-V22 generic active runner directly.  That
bypassed V23's unified lineage prior, prior-selected contest response, disabled
general-identity duplicate, and automatic-control paths.  V25 preserves the
published V24 ballot panel and its three structural postprocesses.  It restores
the omitted V23 paths except the V23 third-candidate profile/pressure pair,
which would duplicate V24's accepted weak-C response and fails the pre-existing
winner-safety gate.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterator

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from presidential_issue_engine import automatic_contest_response  # noqa: E402
from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine import unified_lineage_identity  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model_v24 as v24  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v25"
CONFIG_PATH = ROOT / "data" / "config" / "active_presidential_model_v23.json"
AUTOMATIC_DIR = ROOT / "outputs" / "automatic_controls_v23"
FOOTPRINT_BASE = ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
HISTORY = ROOT / "presidential_issue_engine" / "fixed_dataset" / "bloc_history_results.csv"
ASSEMBLY = ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv"
PARTY_TRANSITIONS = ROOT / "data" / "raw" / "party_lineage_transitions.csv"
FINAL_VARIANT = "v25_corrected_v23_lineage_v24_structural"
CORE_RUNTIME_REPAIRS = frozenset(
    {
        "policy_binding",
        "automatic_contest_response",
        "unified_prior",
        "unified_identity",
        "disable_general_identity",
    }
)
AVAILABLE_AUTOMATIC_INPUT_REPAIRS = frozenset(
    {
        "freeze_issue_seeds",
        "candidate_regional_base",
        "regional_alignment",
        "mega_issue_inputs",
        "generation_weights",
        "economic_housing_alignment",
        "political_landscape",
        "third_candidate_inputs",
    }
)
AVAILABLE_RUNTIME_REPAIRS = CORE_RUNTIME_REPAIRS | AVAILABLE_AUTOMATIC_INPUT_REPAIRS
# V24's accepted weak-C mechanism was selected with the existing generic
# third-candidate profile/pressure paths.  Rebinding those two paths to the V23
# automatic controls duplicates third-candidate pressure and fails the existing
# 2022 winner-safety gate.  Preserve the accepted V24 routing and its inputs.
RUNTIME_REPAIRS = AVAILABLE_RUNTIME_REPAIRS - {"third_candidate_inputs"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _augment_input_manifest(destination: Path, repairs: frozenset[str]) -> None:
    path = destination / "input_manifest.csv"
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    explicit = [
        CONFIG_PATH,
        HISTORY,
        ASSEMBLY,
        PARTY_TRANSITIONS,
        v24.V24_BASELINE,
        v24.V24_DATA / "presidential_results_standardized.csv",
        v24.V24_DATA / "coalition_events.csv",
        v24.V24_DATA / "candidate_slot_assignments_v2.csv",
        v24.V24_DATA / "candidate_party_speech_context.csv",
        v24.V24_DATA / "candidate_vote_conversion_context.csv",
        v24.V24_DATA / "scored_contest_scope.csv",
        v24.V24_DATA / "third_candidate_lineage.csv",
    ]
    if "candidate_regional_base" in repairs:
        explicit.append(FOOTPRINT_BASE)
    automatic_paths = {
        "regional_alignment": [AUTOMATIC_DIR / "regional_alignment_with_policy.csv"],
        "mega_issue_inputs": [
            AUTOMATIC_DIR / "mega_issue_intensity.csv",
            AUTOMATIC_DIR / "mega_issue_taxonomy.csv",
        ],
        "generation_weights": [AUTOMATIC_DIR / "election_generation_weights.csv"],
        "economic_housing_alignment": [
            AUTOMATIC_DIR / "economic_slot_alignment.csv",
            AUTOMATIC_DIR / "housing_slot_alignment.csv",
        ],
        "political_landscape": [AUTOMATIC_DIR / "candidate_political_landscape.csv"],
        "third_candidate_inputs": [
            AUTOMATIC_DIR / "third_candidate_profile.csv",
            AUTOMATIC_DIR / "third_candidate_pressure.csv",
        ],
    }
    for repair, paths in automatic_paths.items():
        if repair in repairs:
            explicit.extend(paths)
    extra = pd.DataFrame(
        [
            {
                "path": source.relative_to(ROOT).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
            for source in explicit
        ]
    )
    combined = (
        pd.concat([manifest, extra], ignore_index=True)
        .drop_duplicates("path", keep="last")
        .sort_values("path")
        .reset_index(drop=True)
    )
    v24._atomic_csv_crlf(combined, path)


@contextmanager
def corrected_runtime(
    active,
    builder,
    nested,
    base_eval,
    *,
    repairs: frozenset[str] = RUNTIME_REPAIRS,
) -> Iterator[dict[str, object]]:
    """Install selected V23 runtime repairs on the versioned V24 ballot panel."""

    unknown = set(repairs) - set(AVAILABLE_RUNTIME_REPAIRS)
    if unknown:
        raise ValueError(f"unknown V25 runtime repairs: {sorted(unknown)}")

    history = pd.read_csv(HISTORY, encoding="utf-8-sig")
    assembly = pd.read_csv(ASSEMBLY, encoding="utf-8-sig")
    exact_events = unified_lineage_identity.build_exact_lineage_events(history, assembly)
    transitions = pd.read_csv(PARTY_TRANSITIONS, encoding="utf-8-sig")
    candidate_parties = (
        pd.read_csv(
            v24.V24_DATA / "presidential_results_standardized.csv",
            encoding="utf-8-sig",
            usecols=["election_id", "slot", "candidate_name", "party_name"],
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    original_response = contest_regime.apply_contest_regime_response
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
        adjusted, audit = automatic_contest_response.apply_prior_selected_contest_response(
            frame,
            regimes,
            prediction_column=prediction_column,
            apply_response=original_response,
            election_order=nested.ELECTIONS,
            slot_column=slot_column,
            output_column=output_column,
            critical_elasticity=critical_elasticity,
            swing_elasticity=swing_elasticity,
        )
        response_audit["audit"] = audit
        return adjusted

    def unified_prior(frame, ignored_history, election_order):
        del ignored_history
        return unified_lineage_identity.attach_lineage_projected_prior(
            frame,
            exact_events,
            candidate_parties,
            election_order,
        )

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
        del events
        adjusted, audit, reliability = unified_lineage_identity.apply_unified_lineage_routing(
            frame,
            exact_events,
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
        return adjusted, audit

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

    original_load_policy = active.load_policy

    def load_v23_policy(path=CONFIG_PATH):
        del path
        return original_load_policy(CONFIG_PATH)

    exclusions = v24.scored_exclusions()
    engines = {
        active.nested.engine,
        active.assignment_builder.engine,
        base_eval.engine,
        builder.engine,
    }
    attributes: list[tuple[object, str, object]] = [
        (active, "CONFIG_PATH", CONFIG_PATH),
        (active, "regenerate_assignments", lambda: None),
        (nested, "ASSIGNMENT_PATH", v24.V24_DATA / "candidate_slot_assignments_v2.csv"),
        (base_eval, "BASELINE_PATH", v24.V24_BASELINE),
    ]
    if "policy_binding" in repairs:
        attributes.append((active, "load_policy", load_v23_policy))
    if "automatic_contest_response" in repairs:
        attributes.append(
            (active.contest_regime, "apply_contest_regime_response", automatic_apply)
        )
    if "unified_prior" in repairs:
        attributes.extend(
            [
                (active.nested.engine, "attach_bloc_prior", unified_prior),
                (active.assignment_builder.engine, "attach_bloc_prior", unified_prior),
            ]
        )
    if "unified_identity" in repairs:
        attributes.extend(
            [
                (active.chungcheong_identity, "build_identity_events", lambda _: exact_events),
                (active.chungcheong_identity, "apply_identity_routing", unified_apply),
            ]
        )
    if "disable_general_identity" in repairs:
        attributes.extend(
            [
                (active.regional_identity, "build_distinctiveness_events", no_general_events),
                (active.regional_identity, "apply_regional_identity_routing", no_general_apply),
            ]
        )
    if "freeze_issue_seeds" in repairs:
        attributes.append((active, "regenerate_issue_seeds", lambda: None))
    if "candidate_regional_base" in repairs:
        attributes.append((active, "CANDIDATE_REGIONAL_BASE", FOOTPRINT_BASE))
    if "regional_alignment" in repairs:
        attributes.append(
            (
                active,
                "CHUNGCHEONG_ALIGNMENT",
                AUTOMATIC_DIR / "regional_alignment_with_policy.csv",
            )
        )
    if "mega_issue_inputs" in repairs:
        attributes.append(
            (active, "MEGA_ISSUE_INTENSITY", AUTOMATIC_DIR / "mega_issue_intensity.csv")
        )
    for engine in engines:
        engine_attributes = [
                (engine, "RESULTS", str(v24.V24_DATA / "presidential_results_standardized.csv")),
                (engine, "COALITION_EVENTS", str(v24.V24_DATA / "coalition_events.csv")),
                (engine, "CANDIDATE_PARTY_SPEECH_CONTEXT", str(v24.V24_DATA / "candidate_party_speech_context.csv")),
                (engine, "CANDIDATE_VOTE_CONVERSION_CONTEXT", str(v24.V24_DATA / "candidate_vote_conversion_context.csv")),
                # V24's ballot-faithful correction supersedes V23's slot-keyed
                # withdrawal registry.  Re-enabling it removes the real 2022
                # slot-C ballot candidate from the assembled panel.
                (engine, "_load_scored_contest_scope_exclusions", lambda _excluded=exclusions: set(_excluded)),
            ]
        if "candidate_regional_base" in repairs:
            engine_attributes.append(
                (engine, "CANDIDATE_REGIONAL_BASE", str(FOOTPRINT_BASE))
            )
        if "mega_issue_inputs" in repairs:
            engine_attributes.extend(
                [
                    (engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(AUTOMATIC_DIR / "mega_issue_intensity.csv")),
                    (engine, "MEGA_ISSUE_TAXONOMY", str(AUTOMATIC_DIR / "mega_issue_taxonomy.csv")),
                ]
            )
        if "generation_weights" in repairs:
            engine_attributes.append(
                (engine, "ELECTION_GENERATION_WEIGHTS", str(AUTOMATIC_DIR / "election_generation_weights.csv"))
            )
        if "economic_housing_alignment" in repairs:
            engine_attributes.extend(
                [
                    (engine, "ECONOMIC_SLOT_ALIGNMENT", str(AUTOMATIC_DIR / "economic_slot_alignment.csv")),
                    (engine, "HOUSING_SLOT_ALIGNMENT", str(AUTOMATIC_DIR / "housing_slot_alignment.csv")),
                ]
            )
        if "political_landscape" in repairs:
            engine_attributes.append(
                    (engine, "CANDIDATE_POLITICAL_LANDSCAPE", str(AUTOMATIC_DIR / "candidate_political_landscape.csv")),
            )
        if "third_candidate_inputs" in repairs:
            engine_attributes.extend(
                [
                    (engine, "THIRD_CANDIDATE_PROFILE", str(AUTOMATIC_DIR / "third_candidate_profile.csv")),
                    (engine, "THIRD_CANDIDATE_PRESSURE", str(AUTOMATIC_DIR / "third_candidate_pressure.csv")),
                ]
            )
        attributes.extend(engine_attributes)

    original_rows = builder._all_ballot_rows
    original_redistribution = builder._apply_withdrawal_redistribution
    v24.install_ballot_patches(builder)
    try:
        with patching.patched(attributes):
            yield {
                "response_audit": response_audit,
                "lineage_reliability": lineage_reliability,
            }
    finally:
        builder._all_ballot_rows = original_rows
        builder._apply_withdrawal_redistribution = original_redistribution


def _synchronise(
    destination: Path,
    predictions: pd.DataFrame,
    audits: dict[str, pd.DataFrame],
    active,
    nested,
    repairs: frozenset[str],
) -> None:
    summary, by_election, national = nested._metrics(predictions, "layer_pred", FINAL_VARIANT)
    v24._atomic_csv_crlf(by_election, destination / "by_election.csv")
    v24._atomic_csv_crlf(national, destination / "national_predictions.csv")

    for filename, final_rows in {
        "candidate_stage_summary.csv": pd.DataFrame([summary]),
        "candidate_stage_by_election.csv": by_election,
        "candidate_stage_national.csv": national,
    }.items():
        path = destination / filename
        existing = pd.read_csv(path, encoding="utf-8-sig")
        existing = existing.loc[existing["variant"].astype(str).ne(FINAL_VARIANT)]
        v24._atomic_csv_crlf(pd.concat([existing, final_rows], ignore_index=True), path)

    summary_path = destination / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["pre_v25_extension_metrics"] = payload.get("metrics")
    payload["metrics"] = summary
    payload["policy_version"] = "active_v25_corrected_v23_lineage_v24_structural"
    payload["predecessor"] = "v24_with_runtime_lineage_defect"
    payload["v24_runtime_lineage_defect_corrected"] = True
    payload["v23_runtime_repairs"] = sorted(repairs)
    payload["preserved_v24_runtime_paths"] = {
        "active_conversion_context": "data/raw/candidate_vote_conversion_context.csv",
        "third_candidate_profile": "data/raw/third_candidate_profile.csv",
        "third_candidate_pressure": "data/raw/third_candidate_pressure.csv",
        "reason": "preserve accepted V24 winner-safe weak-C runtime",
    }
    payload["v24_structural_extensions"] = [
        "strong_incumbent_veto",
        "third_candidate_lineage_ceiling",
        "weak_same_lane_refusal",
    ]
    payload["post_2022_outcomes_used"] = False
    payload["v25_extension_audit_rows"] = {name: int(len(frame)) for name, frame in audits.items()}
    v24._atomic_json_crlf(payload, summary_path)


def run(
    output_dir: Path | None = None,
    *,
    repairs: frozenset[str] = RUNTIME_REPAIRS,
) -> Path:
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)

    from scripts import build_preliminary_slot_assignments as builder
    from scripts import run_active_presidential_model as active
    import evaluate_electorate_layers as base_eval
    from presidential_issue_engine import strong_incumbent_veto
    from presidential_issue_engine import third_candidate_lineage_constraint
    from presidential_issue_engine import weak_same_lane_refusal

    with corrected_runtime(
        active, builder, active.nested, base_eval, repairs=repairs
    ) as runtime:
        active.run(output_dir=destination, rejection_beneficiary_routing_enabled=True)

    path = destination / "nested_predictions.csv"
    predictions = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "candidate_name" not in predictions.columns and "candidate_name_x" in predictions.columns:
        predictions["candidate_name"] = predictions["candidate_name_x"]
    predictions, veto_audit = strong_incumbent_veto.apply_strong_incumbent_veto(predictions)
    predictions, lineage_audit = third_candidate_lineage_constraint.apply_lineage_ceiling(predictions)
    predictions, refusal_audit = weak_same_lane_refusal.apply_weak_same_lane_refusal(predictions)
    audits = {
        "strong_incumbent_veto": veto_audit,
        "third_candidate_lineage_ceiling": lineage_audit,
        "weak_same_lane_refusal": refusal_audit,
    }
    v24._atomic_csv_crlf(predictions, path)
    for name, audit in audits.items():
        v24._atomic_csv_crlf(audit, destination / f"{name}_audit.csv")
    response = runtime["response_audit"].get("audit", pd.DataFrame())
    reliability_parts = runtime["lineage_reliability"]
    reliability = pd.concat(reliability_parts, ignore_index=True) if reliability_parts else pd.DataFrame()
    v24._atomic_csv_crlf(response, destination / "automatic_response_gain_audit.csv")
    v24._atomic_csv_crlf(reliability, destination / "lineage_type_reliability_audit.csv")
    _synchronise(destination, predictions, audits, active, active.nested, repairs)
    _augment_input_manifest(destination, repairs)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--repairs",
        nargs="*",
        choices=sorted(AVAILABLE_RUNTIME_REPAIRS),
        default=sorted(RUNTIME_REPAIRS),
        help="internal ablation control; omitted means the complete corrected runtime",
    )
    args = parser.parse_args()
    destination = run(args.output_dir, repairs=frozenset(args.repairs))
    table = v24.report(destination)
    print(table.to_string(index=False))
    print(
        "macro regional row %.3f | regional weighted %.3f | macro level %.3f | winner %d/%d"
        % (
            table.regional_row_mae_pp.mean(),
            table.regional_weighted_mae_pp.mean(),
            table.level_mae_pp.mean(),
            table.winner_correct.sum(),
            len(table),
        )
    )


if __name__ == "__main__":
    main()
