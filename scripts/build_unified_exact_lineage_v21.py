"""Build the exact-party lineage ledger and point-in-time reliability audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from election_forecast.features.region_bloc_prior import election_date  # noqa: E402
from presidential_issue_engine.unified_lineage_identity import (  # noqa: E402
    build_exact_lineage_events,
    fit_lineage_profiles,
)


OUTPUT_DIR = ROOT / "outputs" / "unified_exact_lineage_v21"
HISTORY = (
    ROOT
    / "presidential_issue_engine"
    / "fixed_dataset"
    / "bloc_history_results.csv"
)
ASSEMBLY = (
    ROOT
    / "data"
    / "raw"
    / "official_sources"
    / "nec_assembly_district_history.csv"
)
PARTY_TRANSITIONS = ROOT / "data" / "raw" / "party_lineage_transitions.csv"
TARGETS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    history = pd.read_csv(HISTORY, encoding="utf-8-sig")
    assembly = pd.read_csv(ASSEMBLY, encoding="utf-8-sig")
    events = build_exact_lineage_events(history, assembly)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(
        OUTPUT_DIR / "exact_lineage_events.csv", index=False, encoding="utf-8-sig"
    )

    reliability_rows: list[pd.DataFrame] = []
    profile_rows: list[pd.DataFrame] = []
    for target in TARGETS:
        cutoff = election_date(target)
        if cutoff is None:
            continue
        fit = fit_lineage_profiles(events, cutoff=pd.Timestamp(cutoff))
        reliability = fit.type_reliability.copy()
        reliability["target_election_id"] = target
        reliability_rows.append(reliability)
        profile = fit.profiles.copy()
        profile["target_election_id"] = target
        profile_rows.append(profile)

    reliability_audit = pd.concat(reliability_rows, ignore_index=True)
    profile_audit = pd.concat(profile_rows, ignore_index=True)
    reliability_audit.to_csv(
        OUTPUT_DIR / "type_reliability_by_target.csv",
        index=False,
        encoding="utf-8-sig",
    )
    profile_audit.to_csv(
        OUTPUT_DIR / "lineage_profiles_by_target.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lineage_summary = (
        events.groupby(["lineage_id", "broad_bloc"], as_index=False)
        .agg(
            rows=("election_id", "size"),
            elections=("election_id", "nunique"),
            regions=("region_id", "nunique"),
            first_event=("event_date", "min"),
            last_event=("event_date", "max"),
            maximum_regional_share=("regional_share", "max"),
        )
        .sort_values(["elections", "rows"], ascending=False)
    )
    lineage_summary.to_csv(
        OUTPUT_DIR / "lineage_summary.csv", index=False, encoding="utf-8-sig"
    )
    source_summary = (
        events.groupby(
            ["election_id", "election_type", "ballot_channel"], as_index=False
        )
        .agg(
            rows=("region_id", "size"),
            regions=("region_id", "nunique"),
            lineages=("lineage_id", "nunique"),
        )
        .sort_values(["election_id", "election_type"])
    )
    source_summary.to_csv(
        OUTPUT_DIR / "event_coverage.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema": "unified_exact_lineage_v21",
        "history_sha256": _sha256(HISTORY),
        "assembly_sha256": _sha256(ASSEMBLY),
        "party_transitions_sha256": _sha256(PARTY_TRANSITIONS),
        "history_rows": int(len(history)),
        "assembly_candidate_rows": int(len(assembly)),
        "event_rows": int(len(events)),
        "elections": int(events["election_id"].nunique()),
        "lineages": int(events["lineage_id"].nunique()),
        "regions": int(events["region_id"].nunique()),
        "assembly_district_source": "exact NEC party rows replace collapsed history",
        "other_election_source": "preserved pre-normalization history party label",
        "type_reliability": "prior same-date candidate versus direct-party agreement",
        "party_genealogy_routing": "dated predecessor-successor graph without vote fitting",
        "single_region_formula": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(
        reliability_audit[
            [
                "target_election_id",
                "election_type",
                "type_reliability",
                "paired_observations",
                "paired_correlation",
                "source",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
