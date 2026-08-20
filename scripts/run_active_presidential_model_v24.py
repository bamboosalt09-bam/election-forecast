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
    from presidential_issue_engine import third_candidate_lineage_constraint as lineage

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
    active.run(output_dir=destination)

    predictions_path = destination / "nested_predictions.csv"
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig", low_memory=False)
    if "candidate_name" not in predictions.columns and "candidate_name_x" in predictions.columns:
        predictions["candidate_name"] = predictions["candidate_name_x"]
    adjusted, audit = lineage.apply_lineage_ceiling(predictions)
    adjusted.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    audit.to_csv(destination / "third_candidate_lineage_audit.csv", index=False, encoding="utf-8-sig")
    return destination


def report(destination: Path) -> pd.DataFrame:
    frame = pd.read_csv(destination / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False)
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
                "regional_mae_pp": round((group["layer_pred"] - group["actual"]).abs().mean() * 100, 3),
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
        "macro regional %.3f | macro level %.3f | winner %d/%d"
        % (
            table.regional_mae_pp.mean(),
            table.level_mae_pp.mean(),
            table.winner_correct.sum(),
            len(table),
        )
    )


if __name__ == "__main__":
    main()
