"""Strict nested v11-based ablation of automatic third-candidate inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import automatic_contest_response  # noqa: E402
from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine.automatic_regional_party_alignment import (  # noqa: E402
    build_full_history_identity_events,
)
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_third_candidate_v12_ablation"
V11_RUN = (
    ROOT
    / "outputs"
    / "automatic_regional_party_alignment_v11_ablation"
    / "supplemental_full_history"
    / "active_run"
)
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)
ALIGNMENT = (
    ROOT
    / "outputs"
    / "automatic_regional_party_alignment_v11"
    / "manual_plus_automatic_alignment.csv"
)
AUTO_PROFILE = (
    ROOT
    / "outputs"
    / "speech_derived_candidate_context_v2"
    / "auto_candidate_role"
    / "third_candidate_profile.csv"
)
AUTO_PRESSURE = (
    ROOT
    / "outputs"
    / "speech_derived_candidate_context_v3"
    / "auto_candidate_role"
    / "third_candidate_pressure.csv"
)
EMPTY_PRESSURE = (
    ROOT
    / "outputs"
    / "speech_derived_candidate_context_v3"
    / "auto_candidate_role"
    / "empty_third_pressure.csv"
)
MANUAL_PRESSURE = ROOT / "data" / "raw" / "third_candidate_pressure.csv"


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _run_variant(label: str, pressure_path: Path) -> tuple[Path, pd.DataFrame]:
    original_apply = contest_regime.apply_contest_regime_response
    audit_holder: dict[str, pd.DataFrame] = {}

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
        audit_holder["audit"] = audit
        return result

    with patching.patched(
        [
            (active.contest_regime, "apply_contest_regime_response", automatic_apply),
            (
                active.chungcheong_identity,
                "build_identity_events",
                build_full_history_identity_events,
            ),
        ]
    ):
        run = clean._run_variant(
            label,
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=ALIGNMENT,
            third_profile_path=AUTO_PROFILE,
            third_pressure_path=pressure_path,
            output_root=OUTPUT_DIR,
        )
    return run, audit_holder["audit"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", dest="selected_variants")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = {
        "auto_profile_manual_pressure": MANUAL_PRESSURE,
        "auto_profile_no_pressure": EMPTY_PRESSURE,
        "auto_profile_auto_pressure": AUTO_PRESSURE,
    }
    selected = set(args.selected_variants or variants)
    unknown = selected - set(variants)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    runs: dict[str, Path] = {}
    for label, pressure_path in variants.items():
        run = OUTPUT_DIR / label / "active_run"
        audit_path = OUTPUT_DIR / label / "automatic_response_gain_audit.csv"
        if label not in selected and not (run / "summary.json").exists():
            continue
        if args.rerun or not (run / "summary.json").exists():
            run, audit = _run_variant(label, pressure_path)
            audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
        runs[label] = run

    summary_rows = [{"variant_label": "v11_reference", **_metrics(V11_RUN)}]
    by_election_frames: list[pd.DataFrame] = []
    national_frames: list[pd.DataFrame] = []
    for label, path in [("v11_reference", V11_RUN), *runs.items()]:
        if label != "v11_reference":
            summary_rows.append({"variant_label": label, **_metrics(path)})
        by_election = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
        by_election["variant_label"] = label
        by_election_frames.append(by_election)
        national = pd.read_csv(path / "national_predictions.csv", encoding="utf-8-sig")
        national["variant_label"] = label
        national_frames.append(national)
    summary = pd.DataFrame(summary_rows)
    by_election = pd.concat(by_election_frames, ignore_index=True)
    national = pd.concat(national_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    national.to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    reference = by_election.loc[
        by_election["variant_label"].eq("v11_reference"),
        ["election_id", "regional_weighted_mae_pp"],
    ].set_index("election_id")["regional_weighted_mae_pp"]
    comparison = by_election.copy()
    comparison["v11_regional_mae_pp"] = comparison["election_id"].map(reference)
    comparison["regional_change_vs_v11_pp"] = (
        comparison["regional_weighted_mae_pp"] - comparison["v11_regional_mae_pp"]
    )
    comparison.to_csv(
        OUTPUT_DIR / "comparison_vs_v11.csv", index=False, encoding="utf-8-sig"
    )
    best = summary.sort_values("regional_equal_election_macro_mae_pp").iloc[0]
    decision = {
        "experiment": "automatic_third_candidate_v12",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "manual_third_profile_used_by_automatic_variants": False,
        "best_aggregate_variant": str(best["variant_label"]),
        "completed_variants": sorted(runs),
        "missing_variants": sorted(set(variants) - set(runs)),
        "active_model_changed": False,
        "promotion_decision": (
            "experiment_only_pending_review"
            if set(runs) == set(variants)
            else "partial_experiment_resume_required"
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print()
    print(
        comparison[
            [
                "variant_label",
                "election_id",
                "regional_weighted_mae_pp",
                "national_candidate_mae_pp",
                "regional_change_vs_v11_pp",
            ]
        ].sort_values(["election_id", "variant_label"]).to_string(index=False)
    )


if __name__ == "__main__":
    main()
