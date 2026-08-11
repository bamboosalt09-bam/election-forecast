"""Build candidate legacy political landscape from 15th Assembly issue matches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_15th_long_term_landscape import clean_name  # noqa: E402
from news_collector.sources.member_party import party_bloc  # noqa: E402
from presidential_issue_engine.region_bloc_prior import normalize_bloc  # noqa: E402
from presidential_issue_engine.speech_landscape_builder import AXES, load_axis_map  # noqa: E402


MATCHES = ROOT / "outputs" / "15th_assembly_conversion" / "issue_phrase_extraction" / "15th_assembly_issue_phrase_matches.csv"
RESULTS = ROOT / "presidential_issue_engine" / "fixed_dataset" / "presidential_results_standardized.csv"
AXIS_MAP = ROOT / "data" / "raw" / "political_landscape_issue_axis.csv"
OUTPUT = ROOT / "data" / "raw" / "assembly15_candidate_legacy_landscape.csv"


def build_candidate_legacy_landscape() -> pd.DataFrame:
    matches = pd.read_csv(MATCHES)
    results = pd.read_csv(RESULTS)
    axis_map = load_axis_map(AXIS_MAP)

    candidates = (
        results.loc[
            (results["slot"].astype(str) != "alpha")
            & results["is_active_slot"].astype(str).str.lower().isin(["1", "true", "yes", "y"]),
            ["election_id", "slot", "candidate_name", "party_name"],
        ]
        .drop_duplicates()
        .copy()
    )
    candidates["speaker_clean"] = candidates["candidate_name"].map(clean_name)
    candidates["candidate_bloc"] = candidates["party_name"].map(party_bloc).map(normalize_bloc)

    roster = pd.read_csv(ROOT / "data" / "raw" / "assembly15_member_roster.csv")
    roster = roster[["name", "bloc"]].drop_duplicates().rename(
        columns={"name": "speaker_clean", "bloc": "legacy_bloc"}
    )
    candidates = candidates.merge(roster, on="speaker_clean", how="left")
    candidates = candidates.loc[
        candidates["legacy_bloc"].notna()
        & (
            (candidates["candidate_bloc"] == candidates["legacy_bloc"])
            | (candidates["candidate_bloc"] == "무소속")
        )
    ].copy()

    frame = matches.copy()
    frame["speaker_clean"] = frame["speaker"].map(clean_name)
    frame["issue_weight"] = pd.to_numeric(frame["issue_weight"], errors="coerce").fillna(0.0)
    joined = frame.merge(candidates, on="speaker_clean", how="inner")
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "election_id",
                "slot",
                "candidate_name",
                *AXES,
                "matched_rows",
                "issue_count",
                "available_date",
                "confidence",
                "source",
                "notes",
            ]
        )

    issue = (
        joined.groupby(["election_id", "slot", "candidate_name", "issue_name"], as_index=False)
        .agg(issue_weight_sum=("issue_weight", "sum"), matched_rows=("issue_name", "size"))
    )
    total = issue.groupby(["election_id", "slot", "candidate_name"])["issue_weight_sum"].transform("sum").replace(0.0, np.nan)
    issue["emphasis_within"] = (issue["issue_weight_sum"] / total).fillna(0.0)
    weighted = issue.merge(axis_map[["issue_name", *AXES]], on="issue_name", how="inner")
    for axis in AXES:
        weighted[axis] = weighted[axis] * weighted["emphasis_within"]
    vectors = weighted.groupby(["election_id", "slot", "candidate_name"], as_index=False)[AXES].sum()
    evidence = issue.groupby(["election_id", "slot", "candidate_name"], as_index=False).agg(
        matched_rows=("matched_rows", "sum"),
        issue_count=("issue_name", "nunique"),
    )
    vectors = vectors.merge(evidence, on=["election_id", "slot", "candidate_name"], how="left")
    vectors["available_date"] = "2000-01-15"
    vectors["confidence"] = (
        np.sqrt(pd.to_numeric(vectors["matched_rows"], errors="coerce").fillna(0.0))
        / (
            np.sqrt(pd.to_numeric(vectors["matched_rows"], errors="coerce").fillna(0.0))
            + 18.0
        )
    ).clip(0.0, 0.65)
    vectors["source"] = "15th_assembly_candidate_issue_phrase_extraction"
    vectors["notes"] = "Candidate-name exact match after speaker-title cleanup; use as low-weight legacy landscape only"
    return vectors[
        [
            "election_id",
            "slot",
            "candidate_name",
            *AXES,
            "matched_rows",
            "issue_count",
            "available_date",
            "confidence",
            "source",
            "notes",
        ]
    ].sort_values(["election_id", "slot"], ignore_index=True)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame = build_candidate_legacy_landscape()
    frame.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved {len(frame)} rows: {OUTPUT}")
    if not frame.empty:
        print(frame[["election_id", "slot", "candidate_name", "matched_rows", "issue_count", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
