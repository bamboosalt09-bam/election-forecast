"""Derive candidate issue profiles from point-in-time Assembly evidence.

The compiler intentionally separates unsigned issue association from signed
candidate treatment.  General issue discussion can increase association, but
only explicit person, party, or government attribution can create direction.
No election outcome column is accepted by these functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from presidential_issue_engine.point_in_time import filter_available_by_election


SCHEMA_VERSION = "speech_derived_issue_profile_v1"
KEYS = ["election_id", "slot", "issue_name"]
ALLOWED_TARGET_TYPES = {"person", "party", "government"}
POLITICAL_SHOCK_ISSUES = {
    "corruption_integrity",
    "external_shock",
    "regime_change",
    "security_nk",
    "unification_event",
    "withdrawal_event",
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _percentile(
    frame: pd.DataFrame,
    column: str,
    groups: Sequence[str],
) -> pd.Series:
    values = _numeric(frame, column).clip(lower=0.0)
    return values.groupby([frame[group] for group in groups]).rank(
        method="average",
        pct=True,
    )


def _geometric_mean(*values: pd.Series) -> pd.Series:
    if not values:
        raise ValueError("at least one component is required")
    matrix = np.column_stack(
        [pd.to_numeric(value, errors="coerce").fillna(0.0).clip(0.0, 1.0) for value in values]
    )
    return pd.Series(
        np.prod(np.maximum(matrix, 1e-12), axis=1) ** (1.0 / matrix.shape[1]),
        index=values[0].index,
    ).where(pd.Series((matrix > 0.0).all(axis=1), index=values[0].index), 0.0)


def _eligible(
    frame: pd.DataFrame,
    election_dates: Mapping[str, str],
    source_name: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return filter_available_by_election(
        frame.copy(),
        election_dates,
        source_name=source_name,
    )


def _candidate_registry(
    candidates: pd.DataFrame,
    elections: Sequence[str],
) -> pd.DataFrame:
    required = {"election_id", "slot", "candidate_name"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError("candidate registry missing: " + ", ".join(sorted(missing)))
    out = candidates.loc[candidates["election_id"].astype(str).isin(elections)].copy()
    out = out.loc[out["slot"].astype(str).ne("alpha")]
    if "is_active_slot" in out:
        active = out["is_active_slot"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
        out = out.loc[active]
    keep = ["election_id", "slot", "candidate_name"]
    if "party_name" in out:
        keep.append("party_name")
    return out[keep].drop_duplicates(["election_id", "slot"]).reset_index(drop=True)


def _salience_summary(
    salience: pd.DataFrame,
    election_dates: Mapping[str, str],
    elections: Sequence[str],
) -> pd.DataFrame:
    frame = _eligible(salience, election_dates, "speech_profile_salience")
    frame = frame.loc[frame["election_id"].astype(str).isin(elections)].copy()
    frame["salience_score"] = _numeric(frame, "salience_score").clip(lower=0.0)
    grouped = frame.groupby(["election_id", "issue_name"], as_index=False).agg(
        total_salience=("salience_score", "sum"),
        salience_available_date=("available_date", "max"),
        salience_periods=("period", "nunique"),
    )
    grouped["salience_percentile"] = _percentile(
        grouped, "total_salience", ["election_id"]
    )
    return grouped


def build_candidate_profile(
    links: pd.DataFrame,
    salience: pd.DataFrame,
    character: pd.DataFrame,
    candidates: pd.DataFrame,
    election_dates: Mapping[str, str],
    elections: Sequence[str],
) -> pd.DataFrame:
    """Compile candidate association, direction, and confidence.

    Association is the equal-weight geometric mean of candidate-within-election
    issue emphasis, election-wide salience, and issue evidence coverage.  An
    explicit target attribution can only strengthen that unsigned association.
    Direction is the signed balance of explicit target evidence.  Confidence
    is target-attribution confidence for signed rows and unsigned evidence
    quality otherwise.
    """

    elections = tuple(str(value) for value in elections)
    registry = _candidate_registry(candidates, elections)
    link = _eligible(links, election_dates, "speech_profile_candidate_links")
    link = link.loc[
        link["election_id"].astype(str).isin(elections)
        & link["slot"].astype(str).ne("alpha")
    ].copy()
    link = link.groupby(KEYS, as_index=False).agg(
        mentions=("mentions", "sum"),
        emphasis_volume=("emphasis_volume", "max"),
        emphasis_within=("emphasis_within", "sum"),
        link_available_date=("available_date", "max"),
    )
    salience_summary = _salience_summary(salience, election_dates, elections)

    char = _eligible(character, election_dates, "speech_profile_character")
    char = char.loc[char["election_id"].astype(str).isin(elections)].copy()
    required_target = {
        "target_directional_balance",
        "target_attribution_confidence",
        "target_absolute_evidence",
        "target_source_types",
    }
    missing_target = required_target - set(char.columns)
    if missing_target:
        raise ValueError(
            "character overlay missing target attribution: "
            + ", ".join(sorted(missing_target))
        )
    char_columns = [
        *KEYS,
        "available_date",
        "issue_confidence_quality",
        "issue_evidence_count",
        "issue_speaker_count",
        "issue_committee_count",
        "link_evidence_count",
        "link_reliability",
        "target_signed_evidence",
        "target_absolute_evidence",
        "target_directional_balance",
        "target_attribution_confidence",
        "target_source_types",
    ]
    char = char[[column for column in char_columns if column in char]].copy()
    char = char.rename(columns={"available_date": "character_available_date"})
    char = char.sort_values(KEYS).drop_duplicates(KEYS, keep="last")

    issues = salience_summary[["election_id", "issue_name"]].drop_duplicates()
    frame = registry.merge(issues, on="election_id", how="inner")
    frame = frame.merge(link, on=KEYS, how="left")
    frame = frame.merge(salience_summary, on=["election_id", "issue_name"], how="left")
    frame = frame.merge(char, on=KEYS, how="left")

    numeric_columns = [
        "mentions",
        "emphasis_volume",
        "emphasis_within",
        "total_salience",
        "salience_percentile",
        "salience_periods",
        "issue_confidence_quality",
        "issue_evidence_count",
        "issue_speaker_count",
        "issue_committee_count",
        "link_evidence_count",
        "link_reliability",
        "target_signed_evidence",
        "target_absolute_evidence",
        "target_directional_balance",
        "target_attribution_confidence",
    ]
    for column in numeric_columns:
        frame[column] = _numeric(frame, column)
    frame["target_source_types"] = frame.get("target_source_types", "").fillna("").astype(str)

    frame["emphasis_percentile"] = _percentile(
        frame, "emphasis_within", ["election_id", "slot"]
    )
    issue_evidence = (
        frame[["election_id", "issue_name", "issue_evidence_count"]]
        .drop_duplicates(["election_id", "issue_name"])
        .copy()
    )
    issue_evidence["evidence_coverage_percentile"] = _percentile(
        issue_evidence, "issue_evidence_count", ["election_id"]
    )
    frame = frame.merge(
        issue_evidence[["election_id", "issue_name", "evidence_coverage_percentile"]],
        on=["election_id", "issue_name"],
        how="left",
    )
    issue_quality = (
        frame[["election_id", "issue_name", "issue_confidence_quality"]]
        .drop_duplicates(["election_id", "issue_name"])
        .copy()
    )
    issue_quality["issue_quality_percentile"] = _percentile(
        issue_quality, "issue_confidence_quality", ["election_id"]
    )
    frame = frame.merge(
        issue_quality[["election_id", "issue_name", "issue_quality_percentile"]],
        on=["election_id", "issue_name"],
        how="left",
    )

    frame["unsigned_association"] = _geometric_mean(
        frame["emphasis_percentile"],
        frame["salience_percentile"],
        frame["evidence_coverage_percentile"],
    )
    explicit_type = frame["target_source_types"].str.split("|").map(
        lambda values: bool(ALLOWED_TARGET_TYPES.intersection(values))
    )
    explicit = (
        explicit_type
        & frame["target_absolute_evidence"].gt(0.0)
        & frame["target_attribution_confidence"].gt(0.0)
    )
    target_link = frame["target_attribution_confidence"].clip(0.0, 1.0).where(explicit, 0.0)
    frame["association_strength"] = (
        1.0 - (1.0 - frame["unsigned_association"]) * (1.0 - target_link)
    ).clip(0.0, 1.0)

    evidence_balance = np.divide(
        frame["target_signed_evidence"],
        frame["target_absolute_evidence"],
        out=np.zeros(len(frame), dtype=float),
        where=frame["target_absolute_evidence"].to_numpy(float) > 0.0,
    )
    fallback_balance = frame["target_directional_balance"].to_numpy(float)
    evidence_balance = np.where(
        frame["target_absolute_evidence"].to_numpy(float) > 0.0,
        evidence_balance,
        fallback_balance,
    )
    frame["direction"] = pd.Series(evidence_balance, index=frame.index).clip(-1.0, 1.0)
    frame.loc[~explicit, "direction"] = 0.0

    frame["unsigned_confidence"] = _geometric_mean(
        frame["issue_quality_percentile"],
        frame["evidence_coverage_percentile"],
    )
    frame["direction_confidence"] = (
        frame["target_attribution_confidence"].clip(0.0, 1.0)
        * frame["direction"].abs()
    ).where(explicit, 0.0)
    frame["confidence"] = frame["unsigned_confidence"]
    frame.loc[explicit, "confidence"] = frame.loc[explicit, "direction_confidence"]
    frame["confidence"] = frame["confidence"].clip(0.0, 1.0)

    date_columns = [
        pd.to_datetime(frame.get(column), errors="coerce")
        for column in [
            "link_available_date",
            "salience_available_date",
            "character_available_date",
        ]
    ]
    frame["available_date"] = pd.concat(date_columns, axis=1).max(axis=1).dt.strftime(
        "%Y-%m-%d"
    )
    frame["candidate_id"] = frame["election_id"].astype(str) + "_" + frame["slot"].astype(str)
    frame["source_type"] = "assembly_speech_derived"
    frame["provenance_class"] = "deterministic_source_derived"
    frame["derivation_version"] = SCHEMA_VERSION
    frame["notes"] = (
        "Association from equal-weight speech emphasis, salience, and evidence coverage; "
        "direction only from explicit target evidence"
    )
    columns = [
        "election_id",
        "candidate_id",
        "slot",
        "candidate_name",
        "issue_name",
        "association_strength",
        "direction",
        "available_date",
        "source_type",
        "confidence",
        "notes",
        "provenance_class",
        "derivation_version",
        "mentions",
        "emphasis_within",
        "emphasis_percentile",
        "salience_percentile",
        "evidence_coverage_percentile",
        "unsigned_association",
        "target_signed_evidence",
        "target_absolute_evidence",
        "target_directional_balance",
        "target_attribution_confidence",
        "target_source_types",
        "direction_confidence",
        "issue_evidence_count",
        "issue_speaker_count",
        "issue_committee_count",
        "link_evidence_count",
        "link_reliability",
    ]
    return frame[columns].sort_values(KEYS).reset_index(drop=True)


def build_mega_axis(
    salience: pd.DataFrame,
    character: pd.DataFrame,
    election_dates: Mapping[str, str],
    elections: Sequence[str],
) -> pd.DataFrame:
    """Select high-evidence issue axes without election-specific weights."""

    elections = tuple(str(value) for value in elections)
    grouped = _salience_summary(salience, election_dates, elections)
    char = _eligible(character, election_dates, "speech_profile_mega_character")
    char = char.loc[char["election_id"].astype(str).isin(elections)].copy()
    char = char.sort_values(["election_id", "issue_name", "slot"]).drop_duplicates(
        ["election_id", "issue_name"], keep="first"
    )
    keep = [
        "election_id",
        "issue_name",
        "issue_evidence_count",
        "issue_confidence_quality",
        "accountability_score",
        "polarized_score",
        "character_intensity",
        "available_date",
    ]
    char = char[[column for column in keep if column in char]].rename(
        columns={"available_date": "character_available_date"}
    )
    grouped = grouped.merge(char, on=["election_id", "issue_name"], how="left")
    for column in [
        "issue_evidence_count",
        "issue_confidence_quality",
        "accountability_score",
        "polarized_score",
        "character_intensity",
    ]:
        grouped[column] = _numeric(grouped, column)
    grouped["evidence_percentile"] = _percentile(
        grouped, "issue_evidence_count", ["election_id"]
    )
    grouped["quality_percentile"] = _percentile(
        grouped, "issue_confidence_quality", ["election_id"]
    )
    grouped["mega_score"] = _geometric_mean(
        grouped["salience_percentile"],
        grouped["evidence_percentile"],
        grouped["quality_percentile"],
    )
    grouped["is_political_shock"] = grouped["issue_name"].isin(POLITICAL_SHOCK_ISSUES)
    selected: list[pd.DataFrame] = []
    for _, election in grouped.groupby("election_id", sort=True):
        ordered = election.sort_values(
            ["mega_score", "total_salience", "issue_name"],
            ascending=[False, False, True],
        )
        base = ordered.head(2)
        shock = ordered.loc[
            ordered["is_political_shock"]
            & ordered["mega_score"].ge(float(ordered["mega_score"].median()))
            & ~ordered["issue_name"].isin(base["issue_name"])
        ].head(1)
        selected.append(pd.concat([base, shock], ignore_index=True))
    out = pd.concat(selected, ignore_index=True) if selected else grouped.iloc[0:0]
    out["mega_event"] = "assembly_speech_" + out["issue_name"].astype(str)
    out["primary_issue"] = out["issue_name"]
    out["secondary_issue"] = ""
    out["axis_weight"] = out["mega_score"].clip(0.0, 1.0)
    regime_character = out[["accountability_score", "polarized_score", "character_intensity"]].max(axis=1)
    out["regime_axis_weight"] = (
        out["mega_score"] * regime_character.clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["available_date"] = pd.concat(
        [
            pd.to_datetime(out["salience_available_date"], errors="coerce"),
            pd.to_datetime(out["character_available_date"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1).dt.strftime("%Y-%m-%d")
    out["activation_method"] = SCHEMA_VERSION
    out["notes"] = "Unsigned Assembly salience/evidence axis; no outcome fields"
    return out[
        [
            "election_id",
            "mega_event",
            "primary_issue",
            "secondary_issue",
            "axis_weight",
            "regime_axis_weight",
            "available_date",
            "activation_method",
            "notes",
        ]
    ].reset_index(drop=True)


def build_attribution(profile: pd.DataFrame, axis: pd.DataFrame) -> pd.DataFrame:
    """Compile mega attribution only where explicit signed evidence exists."""

    joined = axis[["election_id", "mega_event", "primary_issue"]].merge(
        profile,
        left_on=["election_id", "primary_issue"],
        right_on=["election_id", "issue_name"],
        how="left",
    )
    joined = joined.loc[
        joined["slot"].notna()
        & joined["direction"].abs().gt(0.0)
        & joined["direction_confidence"].gt(0.0)
    ].copy()
    joined["target_type"] = "candidate_slot"
    joined["target"] = joined["slot"].astype(str)
    joined["polarity"] = np.sign(joined["direction"])
    joined["weight"] = (
        joined["association_strength"] * joined["direction"].abs()
    ).clip(0.0, 1.0)
    joined["confidence"] = joined["direction_confidence"].clip(0.0, 1.0)
    joined["notes"] = "Explicit Assembly target evidence; deterministic source-derived"
    return joined[
        [
            "election_id",
            "mega_event",
            "issue_name",
            "target_type",
            "target",
            "polarity",
            "weight",
            "available_date",
            "confidence",
            "notes",
        ]
    ].reset_index(drop=True)


def build_outputs(
    links: pd.DataFrame,
    salience: pd.DataFrame,
    character: pd.DataFrame,
    candidates: pd.DataFrame,
    election_dates: Mapping[str, str],
    elections: Sequence[str],
) -> dict[str, pd.DataFrame]:
    profile = build_candidate_profile(
        links,
        salience,
        character,
        candidates,
        election_dates,
        elections,
    )
    axis = build_mega_axis(salience, character, election_dates, elections)
    attribution = build_attribution(profile, axis)
    return {
        "candidate_issue_profile.csv": profile,
        "mega_issue_axis.csv": axis,
        "mega_issue_attribution.csv": attribution,
    }
