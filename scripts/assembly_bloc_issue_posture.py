"""Build bloc-issue posture from speaker-level assembly issue matches.

This helper reuses already extracted phrase matches and speaker/member-history
weights.  It does not reprocess raw transcripts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from presidential_issue_engine.build_assembly_speaker_influence import clean_speaker_name  # noqa: E402
from presidential_issue_engine.issue_vote_engine import ELECTION_DATES  # noqa: E402
from presidential_issue_engine.point_in_time import filter_observed_by_election  # noqa: E402
from presidential_issue_engine.region_bloc_prior import normalize_bloc  # noqa: E402


MATCHES = ROOT / "outputs/assembly_speaker_issue_matches_15_22.csv"
SPEAKER_PROFILE = ROOT / "data/raw/assembly_speaker_influence.csv"
ELECTION_ORDER = [
    "pres_2002",
    "pres_2007",
    "pres_2012",
    "pres_2017",
    "pres_2022",
    "pres_2025",
]


def prior_election_rows(frame: pd.DataFrame, election_id: str) -> pd.DataFrame:
    """Return rows from elections strictly earlier than the target."""

    order = {value: index for index, value in enumerate(ELECTION_ORDER)}
    target_index = order.get(str(election_id))
    if frame.empty or target_index is None or "election_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    source_order = frame["election_id"].astype(str).map(order)
    return frame.loc[source_order.notna() & source_order.lt(target_index)].copy()


def _entropy_concentration(values: pd.Series) -> float:
    weights = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    probs = weights / total
    entropy = float(-(probs * np.log(probs.replace(0.0, np.nan))).sum(skipna=True))
    max_entropy = math.log(max(len(probs), 2))
    return float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))


def load_assembly_bloc_issue_posture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return election/bloc/issue posture and election/bloc diagnostics."""

    if not MATCHES.exists() or not SPEAKER_PROFILE.exists():
        issue_columns = [
            "election_id",
            "bloc",
            "issue_name",
            "bloc_issue_raw_weight",
            "bloc_issue_weight",
            "matched_rows",
            "unique_speakers",
        ]
        diag_columns = [
            "election_id",
            "bloc",
            "bloc_total_weight",
            "bloc_unique_speakers",
            "bloc_matched_rows",
            "frame_concentration",
            "speaker_coverage",
            "same_bloc_frame_convergence",
            "avg_mapping_confidence",
            "district_share",
            "proportional_share",
            "government_share",
        ]
        return pd.DataFrame(columns=issue_columns), pd.DataFrame(columns=diag_columns)

    matches = pd.read_csv(
        MATCHES,
        usecols=[
            "election_id",
            "assembly_daesu",
            "meeting_date",
            "speaker",
            "issue_name",
            "issue_weight",
            "matched_term_count",
        ],
    )
    matches = filter_observed_by_election(
        matches,
        ELECTION_DATES,
        source_name="assembly_bloc_issue_posture_matches",
        date_column="meeting_date",
    )
    profile = pd.read_csv(
        SPEAKER_PROFILE,
        usecols=[
            "election_id",
            "assembly_daesu",
            "speaker_clean",
            "speaker_bloc",
            "mandate_type",
            "seniority_weight",
            "role_weight",
            "meeting_weight",
            "mapping_confidence",
        ],
    )
    matches["election_id"] = matches["election_id"].astype(str)
    matches["assembly_daesu"] = matches["assembly_daesu"].astype(str)
    matches["speaker_clean"] = matches["speaker"].map(clean_speaker_name)
    matches = matches.loc[matches["speaker_clean"].astype(str).str.len() > 0].copy()
    profile["election_id"] = profile["election_id"].astype(str)
    profile["assembly_daesu"] = profile["assembly_daesu"].astype(str)
    profile["speaker_clean"] = profile["speaker_clean"].map(clean_speaker_name)
    profile = (
        profile.sort_values("mapping_confidence", ascending=False)
        .drop_duplicates(["election_id", "assembly_daesu", "speaker_clean"])
        .copy()
    )
    joined = matches.merge(
        profile,
        on=["election_id", "assembly_daesu", "speaker_clean"],
        how="left",
    )
    joined["bloc"] = joined["speaker_bloc"].map(normalize_bloc)
    joined = joined.loc[joined["bloc"].notna() & joined["issue_name"].notna()].copy()
    joined["bloc"] = joined["bloc"].astype(str)
    for column, default in [
        ("issue_weight", 0.0),
        ("matched_term_count", 1.0),
        ("seniority_weight", 1.0),
        ("role_weight", 0.85),
        ("meeting_weight", 1.0),
        ("mapping_confidence", 0.45),
    ]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(default)
    joined["mandate_type"] = joined["mandate_type"].fillna("unknown").astype(str)
    joined["bloc_issue_raw_weight"] = (
        joined["issue_weight"].clip(lower=0.0)
        * joined["matched_term_count"].clip(lower=1.0).pow(0.25)
        * joined["seniority_weight"].clip(lower=0.5, upper=1.5)
        * joined["role_weight"].clip(lower=0.5, upper=1.5)
        * joined["meeting_weight"].clip(lower=0.5, upper=1.5)
        * joined["mapping_confidence"].clip(lower=0.25, upper=1.0)
    )
    joined = joined.loc[joined["bloc_issue_raw_weight"] > 0.0].copy()

    issues = (
        joined.groupby(["election_id", "bloc", "issue_name"], as_index=False)
        .agg(
            bloc_issue_raw_weight=("bloc_issue_raw_weight", "sum"),
            matched_rows=("bloc_issue_raw_weight", "size"),
            unique_speakers=("speaker_clean", "nunique"),
        )
    )
    totals = issues.groupby(["election_id", "bloc"])["bloc_issue_raw_weight"].transform("sum")
    issues["bloc_issue_weight"] = np.where(
        totals.to_numpy(float) > 0.0,
        issues["bloc_issue_raw_weight"].to_numpy(float) / totals.to_numpy(float),
        0.0,
    )

    diagnostics = (
        joined.groupby(["election_id", "bloc"], as_index=False)
        .agg(
            bloc_total_weight=("bloc_issue_raw_weight", "sum"),
            bloc_unique_speakers=("speaker_clean", "nunique"),
            bloc_matched_rows=("bloc_issue_raw_weight", "size"),
            frame_concentration=("bloc_issue_raw_weight", _entropy_concentration),
            avg_mapping_confidence=("mapping_confidence", "mean"),
            district_rows=("mandate_type", lambda values: int((values == "district").sum())),
            proportional_rows=("mandate_type", lambda values: int((values == "proportional").sum())),
            government_rows=("mandate_type", lambda values: int((values == "government").sum())),
        )
    )
    max_speakers = diagnostics.groupby("election_id")["bloc_unique_speakers"].transform("max").clip(lower=1)
    diagnostics["speaker_coverage"] = (
        diagnostics["bloc_unique_speakers"].to_numpy(float) / max_speakers.to_numpy(float)
    ).clip(0.0, 1.0)
    diagnostics["same_bloc_frame_convergence"] = (
        0.55 * (1.0 - diagnostics["frame_concentration"])
        + 0.30 * diagnostics["speaker_coverage"]
        + 0.15 * diagnostics["avg_mapping_confidence"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    denom = diagnostics["bloc_matched_rows"].clip(lower=1)
    diagnostics["district_share"] = diagnostics["district_rows"] / denom
    diagnostics["proportional_share"] = diagnostics["proportional_rows"] / denom
    diagnostics["government_share"] = diagnostics["government_rows"] / denom
    diagnostics = diagnostics.drop(columns=["district_rows", "proportional_rows", "government_rows"])
    return issues, diagnostics


def vector_for(
    bloc_issues: pd.DataFrame,
    election_id: str,
    bloc: str,
    value_column: str = "bloc_issue_weight",
) -> pd.Series:
    """Return same-election bloc vector, falling back to historical same-bloc posture."""

    if bloc_issues.empty:
        return pd.Series(dtype=float)
    same = bloc_issues.loc[
        bloc_issues["election_id"].astype(str).eq(str(election_id))
        & bloc_issues["bloc"].astype(str).eq(str(bloc))
    ]
    if same.empty:
        prior = prior_election_rows(bloc_issues, election_id)
        same = prior.loc[prior["bloc"].astype(str).eq(str(bloc))]
    if same.empty:
        return pd.Series(dtype=float)
    return same.groupby("issue_name")[value_column].mean()


def diagnostics_for(bloc_diag: pd.DataFrame, election_id: str, bloc: str) -> pd.Series:
    """Return same-election diagnostics, falling back to historical same-bloc means."""

    if bloc_diag.empty:
        return pd.Series(dtype=float)
    same = bloc_diag.loc[
        bloc_diag["election_id"].astype(str).eq(str(election_id))
        & bloc_diag["bloc"].astype(str).eq(str(bloc))
    ]
    if same.empty:
        prior = prior_election_rows(bloc_diag, election_id)
        same = prior.loc[prior["bloc"].astype(str).eq(str(bloc))]
    if same.empty:
        return pd.Series(dtype=float)
    numeric = same.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.Series(dtype=float)
    return numeric.mean()
