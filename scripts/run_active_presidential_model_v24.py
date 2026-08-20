"""Run the V24 lineage: 1%p scored floor, ballot-faithful withdrawals, lineage ceiling.

V24 keeps every V23 coefficient, gain, threshold, and predictor list. It changes
only which rows are scored and how three structural facts are represented:

1. the scored contest floor is a declared uniform 1%p national vote share,
   replacing per-election exclusions;
2. a withdrawn candidate no longer overwrites the ballot slot of the candidate
   who actually appeared on the ballot, and the slot-keyed duplicates of the
   candidate-keyed withdrawal registry are removed;
3. organisation strength for party-backed non-major candidates is continuous in
   strictly prior party-list evidence, and a third candidate without
   major-party split lineage is capped at its own bloc's direct party evidence.

The frozen V23 artefacts are never read for writing and never modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

V24_DATA = ROOT / "presidential_issue_engine" / "fixed_dataset" / "v24"
V24_BASELINE = (
    ROOT / "presidential_issue_engine" / "report" / "tables" / "v24"
    / "issue_vote_engine_nested_outer_predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v24"
CONFIG_PATH = ROOT / "data" / "config" / "active_presidential_model_v23.json"
WITHDRAWN_SLOT = "W"
FINAL_VARIANT = "v24_structural_residual"


def _atomic_csv_crlf(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\r\n")
    os.replace(temporary, path)


def _atomic_json_crlf(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        rendered = json.dumps(payload, ensure_ascii=False, indent=2).replace("\n", "\r\n")
        handle.write(rendered + "\r\n")
    os.replace(temporary, path)


def scored_exclusions() -> set[tuple[str, str]]:
    """(election_id, slot) pairs excluded by the declared 1%p floor."""

    frame = pd.read_csv(V24_DATA / "scored_contest_scope.csv", encoding="utf-8-sig")
    excluded = frame.loc[
        ~frame["include_in_scored_contest"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    ]
    return set(zip(excluded["election_id"].astype(str), excluded["slot"].astype(str)))


def install_ballot_patches(builder) -> None:
    """Keep the real ballot candidate in slot C; route withdrawals through slot W."""

    original_rows = builder._all_ballot_rows
    empty = pd.DataFrame(
        columns=["election_id", "source_slot", "candidate_name", "latent_candidate_weight"]
    )

    def all_ballot_rows(withdrawal_profiles=None):
        frame = original_rows(empty)
        profiles = (
            builder._withdrawal_profiles() if withdrawal_profiles is None else withdrawal_profiles
        )
        extra = []
        for profile in profiles.itertuples(index=False):
            mask = frame["election_id"].eq(profile.election_id) & frame["slot"].eq(
                profile.source_slot
            )
            added = frame.loc[mask].copy()
            if added.empty:
                continue
            added["slot"] = WITHDRAWN_SLOT
            added["candidate_name"] = profile.candidate_name
            added["candidate_weight"] = profile.latent_candidate_weight
            added["latent_withdrawn_candidate"] = 1.0
            extra.append(added)
        if extra:
            frame = pd.concat([frame, *extra], ignore_index=True, sort=False)
        if "latent_withdrawn_candidate" not in frame.columns:
            frame["latent_withdrawn_candidate"] = 0.0
        frame["latent_withdrawn_candidate"] = pd.to_numeric(
            frame["latent_withdrawn_candidate"], errors="coerce"
        ).fillna(0.0)
        return frame

    original_redistribution = builder._apply_withdrawal_redistribution

    def redistribution(candidate, withdrawal_profiles):
        profiles = withdrawal_profiles.copy()
        if len(profiles):
            profiles["source_slot"] = WITHDRAWN_SLOT
        return original_redistribution(candidate, profiles)

    builder._all_ballot_rows = all_ballot_rows
    builder._apply_withdrawal_redistribution = redistribution


def run(output_dir: Path | None = None, *, rebuild_assignments: bool = False) -> Path:
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)

    from scripts import run_active_presidential_model as active
    from scripts import build_preliminary_slot_assignments as builder
    from scripts import evaluate_preliminary_slot_shadow_nested as nested
    import evaluate_electorate_layers as base_eval
    from presidential_issue_engine import strong_incumbent_veto as incumbent_veto
    from presidential_issue_engine import third_candidate_lineage_constraint as lineage
    from presidential_issue_engine import weak_same_lane_refusal as lane_refusal

    exclusions = scored_exclusions()
    engines = {
        active.nested.engine,
        active.assignment_builder.engine,
        base_eval.engine,
        builder.engine,
    }
    for engine in engines:
        engine.RESULTS = str(V24_DATA / "presidential_results_standardized.csv")
        engine.COALITION_EVENTS = str(V24_DATA / "coalition_events.csv")
        engine.CANDIDATE_PARTY_SPEECH_CONTEXT = str(V24_DATA / "candidate_party_speech_context.csv")
        engine.CANDIDATE_VOTE_CONVERSION_CONTEXT = str(
            V24_DATA / "candidate_vote_conversion_context.csv"
        )
        engine._load_scored_contest_scope_exclusions = (
            lambda _excluded=exclusions: set(_excluded)
        )

    install_ballot_patches(builder)
    assignment_path = V24_DATA / "candidate_slot_assignments_v2.csv"
    if rebuild_assignments:
        assignments, _, _ = builder.build()
        assignments.to_csv(assignment_path, index=False, encoding="utf-8-sig")

    base_eval.BASELINE_PATH = V24_BASELINE
    nested.ASSIGNMENT_PATH = assignment_path
    active.CONFIG_PATH = CONFIG_PATH
    # V24 reads its versioned assignment table above.  The generic active runner
    # otherwise rebuilds into outputs/preliminary_slot_assignment, which is a
    # tracked V23/shared artefact and leaves an unrelated dirty worktree.
    original_regenerate_assignments = active.regenerate_assignments
    active.regenerate_assignments = lambda: None
    try:
        active.run(
            output_dir=destination,
            rejection_beneficiary_routing_enabled=True,
        )
    finally:
        active.regenerate_assignments = original_regenerate_assignments

    predictions_path = destination / "nested_predictions.csv"
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig", low_memory=False)
    if "candidate_name" not in predictions.columns and "candidate_name_x" in predictions.columns:
        predictions = predictions.copy()
        predictions["candidate_name"] = predictions["candidate_name_x"]
    veto_adjusted, veto_audit = incumbent_veto.apply_strong_incumbent_veto(predictions)
    lineage_adjusted, audit = lineage.apply_lineage_ceiling(veto_adjusted)
    adjusted, refusal_audit = lane_refusal.apply_weak_same_lane_refusal(
        lineage_adjusted
    )
    adjusted.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    audit.to_csv(destination / "third_candidate_lineage_audit.csv", index=False, encoding="utf-8-sig")
    veto_audit.to_csv(
        destination / "strong_incumbent_veto_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    refusal_audit.to_csv(
        destination / "weak_same_lane_refusal_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _synchronise_final_artifacts(
        destination,
        adjusted,
        audit,
        veto_audit,
        refusal_audit,
        active,
        nested,
        lineage,
        incumbent_veto,
        lane_refusal,
    )
    return destination


def _synchronise_final_artifacts(
    destination: Path,
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    veto_audit: pd.DataFrame,
    refusal_audit: pd.DataFrame,
    active,
    nested,
    lineage,
    incumbent_veto,
    lane_refusal,
) -> None:
    """Rebuild every metric artefact from the post-ceiling predictions.

    The generic active runner writes its reports before the V24-only ceiling is
    applied.  Keeping those reports would make one output directory advertise
    two different forecasts.  Preserve the pre-ceiling metrics explicitly, then
    replace the public metric artefacts with their final V24 equivalents.
    """

    summary, by_election, national = nested._metrics(
        predictions, "layer_pred", FINAL_VARIANT
    )
    _atomic_csv_crlf(by_election, destination / "by_election.csv")
    _atomic_csv_crlf(national, destination / "national_predictions.csv")

    stage_files = {
        "candidate_stage_summary.csv": pd.DataFrame([summary]),
        "candidate_stage_by_election.csv": by_election,
        "candidate_stage_national.csv": national,
    }
    for filename, final_rows in stage_files.items():
        path = destination / filename
        existing = pd.read_csv(path, encoding="utf-8-sig")
        existing = existing.loc[existing["variant"].astype(str).ne(FINAL_VARIANT)]
        _atomic_csv_crlf(
            pd.concat([existing, final_rows], ignore_index=True),
            path,
        )

    summary_path = destination / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    previous = payload.get("metrics")
    if previous and previous.get("variant") != FINAL_VARIANT:
        payload["pre_lineage_ceiling_metrics"] = previous
    payload["metrics"] = summary
    payload["pre_v24_extension_metrics"] = payload.pop(
        "pre_lineage_ceiling_metrics", previous
    )
    payload["rejection_beneficiary_routing_restored_from_v23"] = True
    payload["strong_incumbent_veto"] = {
        "applied": True,
        "audit_rows": int(len(veto_audit)),
        "affected_elections": sorted(
            veto_audit["election_id"].astype(str).unique().tolist()
        )
        if not veto_audit.empty
        else [],
        "projected_margin_threshold": float(
            incumbent_veto.DEFAULT_PROJECTED_MARGIN_THRESHOLD
        ),
        "gain": float(incumbent_veto.DEFAULT_GAIN),
        "rupture_floor_erosion_enabled": bool(
            incumbent_veto.DEFAULT_RUPTURE_FLOOR_EROSION_ENABLED
        ),
        "theoretical_floor": float(incumbent_veto.DEFAULT_THEORETICAL_FLOOR),
        "rupture_inputs": [
            "mega_issue_intensity_response",
            "direct_mega_score",
            "government_negative_share",
            "government_rejection_breadth",
        ],
        "outcome_fields_used": [],
    }
    payload["third_candidate_lineage_ceiling"] = {
        "applied": True,
        "audit_rows": int(len(audit)),
        "affected_elections": sorted(audit["election_id"].astype(str).unique().tolist())
        if not audit.empty
        else [],
        "defection_floor_share_of_chamber": float(lineage.DEFAULT_DEFECTION_FLOOR),
        "ceiling_column": "direct_party_recent_base",
    }
    payload["weak_same_lane_refusal"] = {
        "applied": True,
        "audit_rows": int(len(refusal_audit)),
        "affected_elections": sorted(
            refusal_audit["election_id"].astype(str).unique().tolist()
        )
        if not refusal_audit.empty
        else [],
        "gain": float(lane_refusal.DEFAULT_GAIN),
        "affinity_power": float(lane_refusal.DEFAULT_AFFINITY_POWER),
        "floor_mode": lane_refusal.DEFAULT_FLOOR_MODE,
        "theoretical_floor": float(lane_refusal.DEFAULT_THEORETICAL_FLOOR),
        "recipient_weight_mode": lane_refusal.DEFAULT_RECIPIENT_WEIGHT_MODE,
        "lineage_gate": "no_major_party_split_mass",
        "outcome_fields_used": [],
    }
    _atomic_json_crlf(payload, summary_path)


def report(destination: Path) -> pd.DataFrame:
    frame = pd.read_csv(destination / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False)
    weighted = pd.read_csv(destination / "by_election.csv", encoding="utf-8-sig").set_index(
        "election_id"
    )
    name = "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"
    rows = []
    for election_id, group in frame.groupby("election_id"):
        actual = group.groupby(name).apply(
            lambda x: np.average(x["actual"], weights=x["contest_votes"])
        )
        predicted = group.groupby(name).apply(
            lambda x: np.average(x["layer_pred"], weights=x["contest_votes"])
        )
        rows.append(
            {
                "election": election_id.replace("pres_", ""),
                "candidates": group[name].nunique(),
                "regional_row_mae_pp": round(
                    (group["layer_pred"] - group["actual"]).abs().mean() * 100, 3
                ),
                "regional_weighted_mae_pp": round(
                    float(weighted.at[election_id, "regional_weighted_mae_pp"]), 3
                ),
                "level_mae_pp": round((predicted - actual).abs().mean() * 100, 3),
                "winner_correct": bool(predicted.idxmax() == actual.idxmax()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rebuild-assignments", action="store_true")
    args = parser.parse_args()
    destination = run(args.output_dir, rebuild_assignments=args.rebuild_assignments)
    table = report(destination)
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
