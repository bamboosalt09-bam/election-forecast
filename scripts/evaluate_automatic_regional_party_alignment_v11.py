"""Strict nested ablation of automatic regional-party candidate routing."""

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


OUTPUT_DIR = ROOT / "outputs" / "automatic_regional_party_alignment_v11_ablation"
V10_DIR = ROOT / "outputs" / "automatic_contest_response_v10_ablation"
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)
ALIGNMENT_DIR = ROOT / "outputs" / "automatic_regional_party_alignment_v11"


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _run_variant(
    label: str,
    *,
    alignment_path: Path,
    full_history_reservoir: bool,
) -> tuple[Path, pd.DataFrame]:
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

    attributes: list[tuple[object, str, object]] = [
        (active.contest_regime, "apply_contest_regime_response", automatic_apply)
    ]
    if full_history_reservoir:
        attributes.append(
            (
                active.chungcheong_identity,
                "build_identity_events",
                build_full_history_identity_events,
            )
        )
    with patching.patched(attributes):
        run_dir = clean._run_variant(
            label,
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=alignment_path,
            output_root=OUTPUT_DIR,
        )
    return run_dir, audit_holder["audit"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", dest="selected_variants")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = {
        "supplemental_direct_party": {
            "alignment_path": ALIGNMENT_DIR / "manual_plus_automatic_alignment.csv",
            "full_history_reservoir": False,
        },
        "supplemental_full_history": {
            "alignment_path": ALIGNMENT_DIR / "manual_plus_automatic_alignment.csv",
            "full_history_reservoir": True,
        },
        "automatic_only_full_history": {
            "alignment_path": ALIGNMENT_DIR / "automatic_alignment.csv",
            "full_history_reservoir": True,
        },
    }
    runs: dict[str, Path] = {}
    response_audits: list[pd.DataFrame] = []
    selected = set(args.selected_variants or variants)
    unknown = selected - set(variants)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    for label, config in variants.items():
        run = OUTPUT_DIR / label / "active_run"
        audit_path = OUTPUT_DIR / label / "automatic_response_gain_audit.csv"
        if label not in selected and not (run / "summary.json").exists():
            continue
        if args.rerun or not (run / "summary.json").exists():
            run, audit = _run_variant(label, **config)
            audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
        elif audit_path.exists():
            audit = pd.read_csv(audit_path, encoding="utf-8-sig")
        else:
            audit = pd.DataFrame()
        runs[label] = run
        if not audit.empty:
            audit["variant_label"] = label
            response_audits.append(audit)

    v10_run = (
        V10_DIR
        / "footprint_prior_selected_routed"
        / "active_run"
    )
    summary_rows = [{"variant_label": "v10_reference", **_metrics(v10_run)}]
    election_frames: list[pd.DataFrame] = []
    national_frames: list[pd.DataFrame] = []
    for label, path in [("v10_reference", v10_run), *runs.items()]:
        if label != "v10_reference":
            summary_rows.append({"variant_label": label, **_metrics(path)})
        by_election = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
        by_election["variant_label"] = label
        election_frames.append(by_election)
        national = pd.read_csv(path / "national_predictions.csv", encoding="utf-8-sig")
        national["variant_label"] = label
        national_frames.append(national)

    summary = pd.DataFrame(summary_rows)
    by_election = pd.concat(election_frames, ignore_index=True)
    national = pd.concat(national_frames, ignore_index=True)
    response_audit = (
        pd.concat(response_audits, ignore_index=True)
        if response_audits
        else pd.DataFrame()
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    national.to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    if not response_audit.empty:
        response_audit.to_csv(
            OUTPUT_DIR / "automatic_response_gain_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )

    reference = by_election.loc[
        by_election["variant_label"].eq("v10_reference"),
        ["election_id", "regional_weighted_mae_pp"],
    ].set_index("election_id")["regional_weighted_mae_pp"]
    comparison = by_election.copy()
    comparison["v10_regional_mae_pp"] = comparison["election_id"].map(reference)
    comparison["regional_change_vs_v10_pp"] = (
        comparison["regional_weighted_mae_pp"] - comparison["v10_regional_mae_pp"]
    )
    comparison.to_csv(
        OUTPUT_DIR / "comparison_vs_v10.csv", index=False, encoding="utf-8-sig"
    )
    best_label = str(
        summary.sort_values("regional_equal_election_macro_mae_pp").iloc[0][
            "variant_label"
        ]
    )
    best_changes = comparison.loc[comparison["variant_label"].eq(best_label)]
    maximum_regression = float(best_changes["regional_change_vs_v10_pp"].max())
    decision = {
        "experiment": "automatic_regional_party_alignment_v11",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "active_model_changed": False,
        "best_aggregate_variant": best_label,
        "best_variant_maximum_election_regression_pp": maximum_regression,
        "automatic_only_is_reported_separately": True,
        "completed_variants": sorted(runs),
        "missing_variants": sorted(set(variants) - set(runs)),
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
                "regional_change_vs_v10_pp",
            ]
        ].sort_values(["election_id", "variant_label"]).to_string(index=False)
    )


if __name__ == "__main__":
    main()
