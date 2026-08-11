"""Build candidate generation profiles from Assembly-derived issue links.

This is a lightweight transform over existing processed speech outputs. It does
not rerun long Assembly parsing. Candidate issue emphasis is mapped through
issue-level generation sensitivity, then centered within each election so the
forecast engine receives relative candidate-generation fit rather than manual
candidate labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from presidential_issue_engine.point_in_time import cutoff_dates_as_strings, filter_available_by_election
except ModuleNotFoundError:  # supports direct script execution
    from point_in_time import cutoff_dates_as_strings, filter_available_by_election


DEFAULT_LINK = Path("data/candidate_issue_link.csv")
DEFAULT_SENSITIVITY = Path("presidential_issue_engine/fixed_dataset/generation_issue_sensitivity.csv")
DEFAULT_RESULTS = Path("presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv")
DEFAULT_OUTPUT = Path("data/raw/candidate_generation_profile.csv")

ELECTION_DATES = {
    "pres_2002": "2002-12-19",
    "pres_2007": "2007-12-19",
    "pres_2012": "2012-12-19",
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}


def _candidate_names(results_path: Path) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    required = {"election_id", "slot", "candidate_name"}
    if results.empty or not required.issubset(results.columns):
        return pd.DataFrame(columns=["election_id", "slot", "candidate_name"])
    return (
        results.loc[results["slot"].astype(str) != "alpha", ["election_id", "slot", "candidate_name"]]
        .dropna()
        .drop_duplicates(["election_id", "slot"])
    )


def build_generation_profile(
    link_path: Path = DEFAULT_LINK,
    sensitivity_path: Path = DEFAULT_SENSITIVITY,
    results_path: Path = DEFAULT_RESULTS,
    affinity_scale: float = 3.0,
) -> pd.DataFrame:
    """Return slot-level generation affinities derived from speech issue emphasis."""

    link = filter_available_by_election(
        pd.read_csv(link_path),
        ELECTION_DATES,
        source_name="candidate_issue_link",
    )
    sensitivity = pd.read_csv(sensitivity_path)
    required_link = {"election_id", "slot", "issue_name", "emphasis_within"}
    required_sensitivity = {"issue_name", "youth_lean"}
    if link.empty or not required_link.issubset(link.columns):
        return pd.DataFrame()
    if sensitivity.empty or not required_sensitivity.issubset(sensitivity.columns):
        return pd.DataFrame()

    frame = link.loc[link["slot"].astype(str) != "alpha"].copy()
    frame["emphasis_within"] = pd.to_numeric(frame["emphasis_within"], errors="coerce").fillna(0.0)
    sensitivity = sensitivity[["issue_name", "youth_lean"]].copy()
    sensitivity["youth_lean"] = pd.to_numeric(sensitivity["youth_lean"], errors="coerce").fillna(0.0)
    frame = frame.merge(sensitivity, on="issue_name", how="left")
    frame["youth_lean"] = frame["youth_lean"].fillna(0.0).clip(-1.0, 1.0)

    lean_abs = frame["youth_lean"].abs()
    weight = frame["emphasis_within"].clip(lower=0.0)
    frame["young_raw"] = weight * frame["youth_lean"].clip(lower=0.0)
    frame["senior_raw"] = weight * (-frame["youth_lean"]).clip(lower=0.0)
    frame["middle_raw"] = weight * (1.0 - lean_abs).clip(lower=0.0)
    frame["generation_salience"] = weight * lean_abs

    grouped = (
        frame.groupby(["election_id", "slot"], as_index=False)
        .agg(
            young_raw=("young_raw", "sum"),
            middle_raw=("middle_raw", "sum"),
            senior_raw=("senior_raw", "sum"),
            generation_salience=("generation_salience", "sum"),
        )
    )
    names = _candidate_names(results_path)
    grouped = grouped.merge(names, on=["election_id", "slot"], how="left")
    grouped["candidate_name"] = grouped["candidate_name"].fillna("")

    for source, target in [
        ("young_raw", "young_affinity"),
        ("middle_raw", "middle_affinity"),
        ("senior_raw", "senior_affinity"),
    ]:
        centered = grouped[source] - grouped.groupby("election_id")[source].transform("mean")
        grouped[target] = (0.50 + affinity_scale * centered).clip(0.05, 0.95)

    grouped["confidence"] = (0.35 + 1.20 * grouped["generation_salience"]).clip(0.0, 0.75)
    grouped["available_date"] = grouped["election_id"].map(cutoff_dates_as_strings(ELECTION_DATES)).fillna("")
    grouped["notes"] = (
        "assembly_issue_link derived; issue emphasis mapped through generation_issue_sensitivity"
    )
    return grouped[
        [
            "election_id",
            "slot",
            "candidate_name",
            "young_affinity",
            "middle_affinity",
            "senior_affinity",
            "available_date",
            "confidence",
            "notes",
        ]
    ].sort_values(["election_id", "slot"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link", type=Path, default=DEFAULT_LINK)
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--affinity-scale", type=float, default=3.0)
    args = parser.parse_args()

    profile = build_generation_profile(
        link_path=args.link,
        sensitivity_path=args.sensitivity,
        results_path=args.results,
        affinity_scale=args.affinity_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"wrote {args.output} rows={len(profile)}")


if __name__ == "__main__":
    main()
