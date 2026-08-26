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
from presidential_issue_engine import election_scope  # noqa: E402


def _target_cutoff(target: str):
    """Resolve a target's cutoff from the project's central registry.

    region_bloc_prior keeps its own presidential date map, and that copy had
    drifted: it stopped at 2022 while election_scope already carried
    pres_2025. The builder resolved cutoffs through the drifted copy and
    skipped any target it could not date, so pres_2025 profiles were never
    generated and nothing reported it.

    Reading election_scope here fixes the builder without changing what
    region_bloc_prior returns to everything else - frozen artifacts were
    produced against that map and must keep reproducing.
    """

    date = election_scope.ELECTION_DATES.get(target)
    if date is None:
        date = election_date(target)
    return pd.Timestamp(date) if date is not None else None
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
#: Every target a profile is fitted for, scored and forecast alike. The profile
#: needs only events strictly before the target's cutoff, so a forecast target
#: is no different in kind from a scored one - pres_2025 was simply never
#: listed, and its absence made every lineage_identity_* column zero in the
#: published 2025 forecast.
TARGETS = (
    "pres_2002",
    "pres_2007",
    "pres_2012",
    "pres_2017",
    "pres_2022",
    "pres_2025",
)


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
        cutoff = _target_cutoff(target)
        if cutoff is None:
            # Never skip. A configured target with no date used to fall through
            # here silently, and because the date map had drifted out of step
            # with election_scope, that is exactly what pres_2025 did: no
            # profile, no error, and five zeroed columns downstream.
            raise RuntimeError(
                f"no election date for configured lineage target {target}; the "
                "date registry and the target list disagree, and a skipped "
                "target produces no profile and no warning"
            )
        fit = fit_lineage_profiles(events, cutoff=cutoff)
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
