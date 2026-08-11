"""Build candidate political landscape vectors from assembly speech issue links.

The builder consumes ``data/candidate_issue_link.csv``, which is already a
copyright-safe aggregate of National Assembly speech issue matches. It does not
read or store speech text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from presidential_issue_engine.point_in_time import cutoff_dates_as_strings, filter_available_by_election
except ModuleNotFoundError:  # supports direct script execution
    from point_in_time import cutoff_dates_as_strings, filter_available_by_election


AXES = [
    "conservative",
    "liberal",
    "progressive",
    "centrist",
    "anti_establishment",
    "reform",
    "regionalist",
]

DEFAULT_ISSUE_LINK = "data/candidate_issue_link.csv"
DEFAULT_RESULTS = "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
DEFAULT_AXIS_MAP = "data/raw/political_landscape_issue_axis.csv"
DEFAULT_MANUAL = "data/raw/candidate_political_landscape.csv"
DEFAULT_OUTPUT = "data/raw/candidate_political_landscape.csv"
ELECTION_DATES = {
    "pres_2002": "2002-12-19",
    "pres_2007": "2007-12-19",
    "pres_2012": "2012-12-19",
    "pres_2017": "2017-05-09",
    "pres_2022": "2022-03-09",
}


def load_axis_map(path: str | Path = DEFAULT_AXIS_MAP) -> pd.DataFrame:
    """Load issue-to-political-axis weights."""

    frame = pd.read_csv(path)
    required = {"issue_name", *AXES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"political landscape axis map is missing required columns: {missing}")
    out = frame[["issue_name", *AXES]].copy()
    for axis in AXES:
        out[axis] = pd.to_numeric(out[axis], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out


def candidate_metadata_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """Extract one candidate row per election-slot from standardized results."""

    required = {"election_id", "slot", "candidate_name", "is_active_slot"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"candidate metadata is missing required columns: {missing}")
    frame = results.loc[results["slot"].astype(str) != "alpha"].copy()
    frame["is_active_slot"] = frame["is_active_slot"].astype(str).str.lower().isin(
        ["1", "true", "yes", "y"]
    )
    frame = frame.loc[frame["is_active_slot"]].copy()
    return frame[["election_id", "slot", "candidate_name"]].drop_duplicates()


def build_landscape_from_issue_links(
    issue_link: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
    axis_map: pd.DataFrame,
    available_date_by_election: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build final-candidate landscape rows from assembly speech issue links."""

    required = {"election_id", "slot", "issue_name", "emphasis_within"}
    missing = sorted(required - set(issue_link.columns))
    if missing:
        raise ValueError(f"candidate issue link is missing required columns: {missing}")

    link = issue_link.loc[
        issue_link["slot"].astype(str) != "alpha",
        ["election_id", "slot", "issue_name", "emphasis_within"],
    ].copy()
    link["emphasis_within"] = pd.to_numeric(link["emphasis_within"], errors="coerce").fillna(0.0)
    joined = link.merge(axis_map, on="issue_name", how="inner")
    if joined.empty:
        return _empty_landscape()

    total_emphasis = (
        joined.groupby(["election_id", "slot"], as_index=False)["emphasis_within"]
        .sum()
        .rename(columns={"emphasis_within": "total_emphasis"})
    )
    for axis in AXES:
        joined[axis] = joined[axis] * joined["emphasis_within"]
    vectors = joined.groupby(["election_id", "slot"], as_index=False)[AXES].sum()
    vectors = vectors.merge(total_emphasis, on=["election_id", "slot"], how="left")
    denominator = vectors["total_emphasis"].replace(0.0, np.nan)
    for axis in AXES:
        vectors[axis] = (vectors[axis] / denominator).fillna(0.0).clip(0.0, 1.0)

    out = vectors.merge(candidate_metadata, on=["election_id", "slot"], how="left")
    out["candidate_name"] = out["candidate_name"].fillna("")
    out["candidate_role"] = "final"
    out["available_date"] = out["election_id"].map(available_date_by_election or {}).fillna("")
    out["confidence"] = (out["total_emphasis"] / (out["total_emphasis"] + 0.50)).clip(0.30, 0.85)
    out["notes"] = "assembly_issue_link derived from National Assembly speech issue aggregates"
    return out[
        [
            "election_id",
            "slot",
            "candidate_name",
            "candidate_role",
            *AXES,
            "available_date",
            "confidence",
            "notes",
        ]
    ].sort_values(["election_id", "slot"], ignore_index=True)


def merge_manual_rows(generated: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Keep generated final rows and append manual non-final rows.

    Final-candidate rows should be reproducible from assembly speech aggregates.
    Manual rows are retained only for candidates who are not represented in the
    final ballot speech-link table, such as withdrawn/unified candidates.
    """

    if manual.empty:
        return generated.copy()
    required = ["election_id", "slot", "candidate_name", "candidate_role", *AXES, "available_date", "confidence", "notes"]
    for column in required:
        if column not in manual.columns:
            manual[column] = "" if column in {"candidate_name", "candidate_role", "available_date", "notes"} else 0.0
    generated_keys = set(
        zip(
            generated["election_id"].astype(str),
            generated["slot"].astype(str),
            generated["candidate_role"].astype(str),
        )
    )
    manual = manual.loc[manual["candidate_role"].astype(str).str.lower() != "final"].copy()
    manual_keep = manual.loc[
        [
            (str(row.election_id), str(row.slot), str(row.candidate_role)) not in generated_keys
            for row in manual.itertuples(index=False)
        ],
        required,
    ].copy()
    out = pd.concat([generated[required], manual_keep], ignore_index=True)
    for axis in AXES + ["confidence"]:
        out[axis] = pd.to_numeric(out[axis], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out.sort_values(["election_id", "slot", "candidate_role"], ignore_index=True)


def build_from_files(
    issue_link_path: str | Path = DEFAULT_ISSUE_LINK,
    results_path: str | Path = DEFAULT_RESULTS,
    axis_map_path: str | Path = DEFAULT_AXIS_MAP,
    manual_path: str | Path | None = DEFAULT_MANUAL,
) -> pd.DataFrame:
    """Build a candidate political landscape frame from repository CSV inputs."""

    issue_link = filter_available_by_election(
        pd.read_csv(issue_link_path),
        ELECTION_DATES,
        source_name="candidate_issue_link",
    )
    results = pd.read_csv(results_path)
    axis_map = load_axis_map(axis_map_path)
    metadata = candidate_metadata_from_results(results)
    generated = build_landscape_from_issue_links(
        issue_link,
        metadata,
        axis_map,
        available_date_by_election=cutoff_dates_as_strings(ELECTION_DATES),
    )
    if manual_path is None or not Path(manual_path).exists():
        return generated
    manual = pd.read_csv(manual_path)
    return merge_manual_rows(generated, manual)


def _empty_landscape() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "election_id",
            "slot",
            "candidate_name",
            "candidate_role",
            *AXES,
            "available_date",
            "confidence",
            "notes",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-link", default=DEFAULT_ISSUE_LINK)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--axis-map", default=DEFAULT_AXIS_MAP)
    parser.add_argument("--manual", default=DEFAULT_MANUAL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--no-manual-merge", action="store_true")
    args = parser.parse_args()

    frame = build_from_files(
        args.issue_link,
        args.results,
        args.axis_map,
        None if args.no_manual_merge else args.manual,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8")
    print(f"saved {len(frame)} rows: {output}")


if __name__ == "__main__":
    main()
