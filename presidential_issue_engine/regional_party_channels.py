"""Prior-only regional-party identity with preference and organization channels."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from election_forecast.features.region_bloc_prior import (
    THIRD_BLOC,
    election_date,
    normalize_bloc,
)
from presidential_issue_engine.electorate_layers import REGIONALIST_PARTY_LABELS


DIRECT_PARTY_TYPES = frozenset(
    {"national_assembly_pr", "assembly_pr", "metro_council_pr", "local_council_pr"}
)
ORGANIZATION_TYPES = frozenset(
    {
        "national_assembly_district",
        "assembly_district",
        "metro_council_district",
        "local_council_district",
    }
)
CANDIDATE_TYPES = frozenset({"metro_governor", "local_governor", "presidential"})

# These are evidence weights, not fitted vote multipliers. Direct party ballots
# identify preference magnitude. Candidate ballots mainly identify persistence
# and organization, so they retain evidence without receiving equal weight.
CHANNEL_TYPE_WEIGHTS: Mapping[str, float] = {
    "national_assembly_pr": 1.00,
    "assembly_pr": 1.00,
    "metro_council_pr": 0.90,
    "local_council_pr": 0.70,
    "national_assembly_district": 0.75,
    "assembly_district": 0.75,
    "metro_council_district": 0.60,
    "local_council_district": 0.40,
    "metro_governor": 0.12,
    "local_governor": 0.08,
    "presidential": 0.25,
}

CHUNGCHEONG_REGIONALIST_ALIASES = frozenset(
    {
        "자유민주연합",
        "자민련",
        "국민중심당",
        "국민중심연합",
        "자유선진당",
        "선진통일당",
        "충청의미래당",
    }
)
LINEAGE_NAME = "chungcheong_regionalist_lineage"
GENERIC_FALLBACK_SCALE = 0.35


def _channel(election_type: object) -> str:
    value = str(election_type)
    if value in DIRECT_PARTY_TYPES:
        return "direct_party_preference"
    if value in ORGANIZATION_TYPES:
        return "district_organization"
    return "candidate_personal_proxy"


def _lineage_rows_from_bloc_history(
    history: pd.DataFrame,
    *,
    date_resolver: Callable[[object], pd.Timestamp | None],
) -> pd.DataFrame:
    work = history.loc[
        history["election_type"].astype(str).isin(CHANNEL_TYPE_WEIGHTS)
    ].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(date_resolver), errors="coerce"
    )
    work = work.loc[work["event_date"].notna()].copy()
    work["vote_share"] = pd.to_numeric(work["vote_share"], errors="coerce").fillna(0.0)
    work["quality"] = pd.to_numeric(
        work["data_quality_weight"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    raw_party = work["bloc"].fillna("").astype(str).str.strip()
    normalized = raw_party.map(normalize_bloc)
    named = raw_party.isin(CHUNGCHEONG_REGIONALIST_ALIASES)
    generic = normalized.eq(THIRD_BLOC) | raw_party.isin(REGIONALIST_PARTY_LABELS)
    work["lineage_specific"] = named
    work["lineage"] = LINEAGE_NAME
    work["fallback_scale"] = named.map({True: 1.0, False: GENERIC_FALLBACK_SCALE})
    work["identity_vote_share"] = work["vote_share"].where(generic, 0.0)
    work["evidence_channel"] = work["election_type"].map(_channel)
    work["source_detail"] = raw_party.where(named, "generic_third_fallback")
    return work[
        [
            "election_id",
            "election_type",
            "event_date",
            "region_id",
            "identity_vote_share",
            "quality",
            "lineage",
            "lineage_specific",
            "fallback_scale",
            "evidence_channel",
            "source_detail",
        ]
    ]


def _assembly_lineage_rows(assembly_history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "election_id",
        "election_date",
        "region_id",
        "district_name",
        "party_name",
        "candidate_votes",
        "district_valid_votes",
    }
    missing = required - set(assembly_history.columns)
    if missing:
        raise ValueError(f"assembly party history missing columns: {sorted(missing)}")
    raw = assembly_history.copy()
    raw["event_date"] = pd.to_datetime(raw["election_date"], errors="coerce")
    raw["candidate_votes"] = pd.to_numeric(
        raw["candidate_votes"], errors="coerce"
    ).fillna(0.0)
    raw["district_valid_votes"] = pd.to_numeric(
        raw["district_valid_votes"], errors="coerce"
    ).fillna(0.0)
    raw = raw.loc[raw["event_date"].notna() & raw["region_id"].notna()].copy()

    district_denominator = raw.drop_duplicates(
        ["election_id", "region_id", "district_name"]
    )
    denominators = (
        district_denominator.groupby(
            ["election_id", "event_date", "region_id"], as_index=False
        )["district_valid_votes"]
        .sum()
        .rename(columns={"district_valid_votes": "region_valid_votes"})
    )
    numerator = (
        raw.loc[
            raw["party_name"].fillna("").astype(str).str.strip().isin(
                CHUNGCHEONG_REGIONALIST_ALIASES
            )
        ]
        .groupby(["election_id", "event_date", "region_id"], as_index=False)[
            "candidate_votes"
        ]
        .sum()
        .rename(columns={"candidate_votes": "lineage_votes"})
    )
    out = denominators.merge(
        numerator, on=["election_id", "event_date", "region_id"], how="left"
    )
    out["lineage_votes"] = out["lineage_votes"].fillna(0.0)
    out["identity_vote_share"] = (
        out["lineage_votes"] / out["region_valid_votes"].replace(0.0, pd.NA)
    ).fillna(0.0).clip(0.0, 1.0)
    out["election_type"] = "assembly_district"
    out["quality"] = 1.0
    out["lineage"] = LINEAGE_NAME
    out["lineage_specific"] = True
    out["fallback_scale"] = 1.0
    out["evidence_channel"] = "district_organization"
    out["source_detail"] = "nec_constituency_party_lineage"
    return out[
        [
            "election_id",
            "election_type",
            "event_date",
            "region_id",
            "identity_vote_share",
            "quality",
            "lineage",
            "lineage_specific",
            "fallback_scale",
            "evidence_channel",
            "source_detail",
        ]
    ]


def build_two_channel_identity_events(
    history: pd.DataFrame,
    assembly_history: pd.DataFrame,
    *,
    date_resolver: Callable[[object], pd.Timestamp | None] = election_date,
) -> pd.DataFrame:
    """Build lineage-aware regional identity events without target outcomes.

    Assembly district rows are replaced by party-level NEC rows so historical
    Liberal Democrats support is not collapsed into a generic third bloc.
    """

    base = _lineage_rows_from_bloc_history(history, date_resolver=date_resolver)
    base = base.loc[~base["election_type"].eq("assembly_district")].copy()
    assembly = _assembly_lineage_rows(assembly_history)
    work = pd.concat([base, assembly], ignore_index=True)

    grouped = (
        work.groupby(
            [
                "election_id",
                "election_type",
                "event_date",
                "region_id",
                "lineage",
                "evidence_channel",
            ],
            as_index=False,
        )
        .agg(
            identity_share=("identity_vote_share", "sum"),
            quality=("quality", "mean"),
            lineage_specific=("lineage_specific", "max"),
            fallback_scale=("fallback_scale", "max"),
            source_detail=("source_detail", lambda values: "|".join(sorted(set(values)))),
        )
    )
    baseline = grouped.groupby(["election_id", "lineage"])["identity_share"].median()
    key = pd.MultiIndex.from_frame(grouped[["election_id", "lineage"]])
    grouped["national_identity_baseline"] = baseline.reindex(key).to_numpy()
    grouped["identity_excess_raw"] = (
        grouped["identity_share"] - grouped["national_identity_baseline"]
    ).clip(lower=0.0, upper=0.40)
    grouped["identity_excess"] = (
        grouped["identity_excess_raw"] * grouped["fallback_scale"]
    )
    grouped["type_weight"] = grouped["election_type"].map(
        CHANNEL_TYPE_WEIGHTS
    ).fillna(0.0)
    return grouped.sort_values(
        ["event_date", "election_id", "evidence_channel", "region_id"]
    ).reset_index(drop=True)


def build_lineage_corroborated_identity_events(
    history: pd.DataFrame,
    assembly_history: pd.DataFrame,
    *,
    corroboration_gain: float = 0.25,
    date_resolver: Callable[[object], pd.Timestamp | None] = election_date,
) -> pd.DataFrame:
    """Preserve the active reservoir and use party lineage as reliability only."""

    from presidential_issue_engine.automatic_regional_party_alignment import (
        build_full_history_identity_events,
    )

    base = build_full_history_identity_events(history, date_resolver=date_resolver)
    channels = build_two_channel_identity_events(
        history, assembly_history, date_resolver=date_resolver
    )
    named = (
        channels.loc[channels["lineage_specific"]]
        .groupby(["election_id", "election_type", "region_id"], as_index=False)
        .agg(lineage_named_share=("identity_share", "sum"))
    )
    out = base.merge(
        named,
        on=["election_id", "election_type", "region_id"],
        how="left",
        validate="one_to_one",
    )
    out["lineage_named_share"] = out["lineage_named_share"].fillna(0.0)
    out["lineage_purity"] = (
        out["lineage_named_share"]
        / out["identity_share"].replace(0.0, pd.NA)
    ).fillna(0.0).clip(0.0, 1.0)
    out["evidence_channel"] = out["election_type"].map(_channel)
    out["base_type_weight"] = out["type_weight"]
    gain = float(max(0.0, corroboration_gain))
    out["lineage_corroboration_multiplier"] = 1.0 + gain * out["lineage_purity"]
    out["type_weight"] = (
        out["base_type_weight"] * out["lineage_corroboration_multiplier"]
    )
    return out
