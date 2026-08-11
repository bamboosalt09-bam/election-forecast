"""Reconstruct candidate regional bases from constituency-level evidence."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import normalize_bloc


SCHEMA_VERSION = "district_reconstructed_candidate_base_v3"
MAJOR_BLOCS = {"국민의힘", "더불어민주당"}
INDEPENDENT_BLOCS = {"무소속", ""}


OUTPUT_COLUMNS = [
    "election_id",
    "slot",
    "candidate_name",
    "region_id",
    "regional_affinity",
    "organization_depth",
    "available_date",
    "confidence",
    "source_type",
    "notes",
    "personal_constituency_signal",
    "executive_office_signal",
    "party_district_organization_signal",
    "personal_match_count",
    "source_election_ids",
    "provenance_class",
    "derivation_version",
]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _district_key(value: Any) -> str:
    text = str(value or "")
    text = re.sub(
        r"서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|"
        r"대전광역시|울산광역시|세종특별자치시|경기도|강원(?:특별자치)?도|"
        r"충청북도|충청남도|전라북도|전북특별자치도|전라남도|"
        r"경상북도|경상남도|제주특별자치도|제주도",
        "",
        text,
    )
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text)


def _target_context(
    context: pd.DataFrame,
    target_dates: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    required = {
        "election_id",
        "slot",
        "candidate_name",
        "bloc",
        "organization_strength",
        "available_date",
        "confidence",
    }
    missing = required.difference(context.columns)
    if missing:
        raise ValueError(f"candidate context is missing columns: {sorted(missing)}")
    out = context[list(required)].copy()
    out["bloc"] = out["bloc"].map(normalize_bloc)
    out["organization_strength"] = pd.to_numeric(
        out["organization_strength"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["context_confidence"] = pd.to_numeric(
        out["confidence"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["target_date_context"] = out["election_id"].astype(str).map(target_dates)
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    out = out.loc[
        out["target_date_context"].notna()
        & out["available_date"].notna()
        & out["available_date"].le(out["target_date_context"])
    ].copy()
    out = out.sort_values("available_date").drop_duplicates(
        ["election_id", "slot", "candidate_name"], keep="last"
    )
    return out.drop(columns="confidence")


def _district_statistics(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = history.copy()
    frame["election_date"] = pd.to_datetime(frame["election_date"])
    frame["bloc"] = frame["bloc"].map(normalize_bloc)
    frame["candidate_votes"] = pd.to_numeric(
        frame["candidate_votes"], errors="coerce"
    ).fillna(0.0)
    frame["district_valid_votes"] = pd.to_numeric(
        frame["district_valid_votes"], errors="coerce"
    ).fillna(0.0)
    frame["candidate_vote_share"] = pd.to_numeric(
        frame["candidate_vote_share"], errors="coerce"
    ).fillna(0.0)
    frame["candidate_won"] = frame["candidate_won"].map(_bool)
    frame["district_key"] = frame["district_name"].map(_district_key)

    district_totals = frame.drop_duplicates(
        ["election_id", "region_id", "district_key"]
    )[
        ["election_id", "election_date", "region_id", "district_key", "district_valid_votes"]
    ]
    total_by_region = district_totals.groupby(
        ["election_id", "election_date", "region_id"], as_index=False
    ).agg(
        regional_valid_votes=("district_valid_votes", "sum"),
        regional_district_count=("district_key", "nunique"),
    )
    bloc_region = frame.groupby(
        ["election_id", "election_date", "region_id", "bloc"], as_index=False
    ).agg(
        bloc_votes=("candidate_votes", "sum"),
        fielded_districts=("district_key", "nunique"),
        won_districts=("candidate_won", "sum"),
    )
    bloc_region = bloc_region.merge(
        total_by_region,
        on=["election_id", "election_date", "region_id"],
        how="left",
    )
    bloc_region["region_bloc_share"] = (
        bloc_region["bloc_votes"] / bloc_region["regional_valid_votes"].replace(0.0, np.nan)
    ).fillna(0.0)
    bloc_region["coverage"] = (
        bloc_region["fielded_districts"]
        / bloc_region["regional_district_count"].replace(0.0, np.nan)
    ).fillna(0.0)
    bloc_region["seat_share"] = (
        bloc_region["won_districts"]
        / bloc_region["regional_district_count"].replace(0.0, np.nan)
    ).fillna(0.0)

    national = bloc_region.groupby(["election_id", "election_date", "bloc"], as_index=False).agg(
        national_bloc_votes=("bloc_votes", "sum"),
        national_valid_votes=("regional_valid_votes", "sum"),
        national_fielded=("fielded_districts", "sum"),
        national_won=("won_districts", "sum"),
        national_districts=("regional_district_count", "sum"),
    )
    national["national_bloc_share"] = (
        national["national_bloc_votes"] / national["national_valid_votes"].replace(0.0, np.nan)
    ).fillna(0.0)
    national["national_coverage"] = (
        national["national_fielded"] / national["national_districts"].replace(0.0, np.nan)
    ).fillna(0.0)
    national["national_seat_share"] = (
        national["national_won"] / national["national_districts"].replace(0.0, np.nan)
    ).fillna(0.0)
    bloc_region = bloc_region.merge(
        national[
            [
                "election_id",
                "election_date",
                "bloc",
                "national_bloc_share",
                "national_coverage",
                "national_seat_share",
            ]
        ],
        on=["election_id", "election_date", "bloc"],
        how="left",
    )
    return frame, bloc_region


def _bounded_union(values: pd.Series) -> float:
    """Combine repeated evidence without allowing linear double counting."""

    array = np.clip(pd.to_numeric(values, errors="coerce").fillna(0.0), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - array.to_numpy(float)))


def _municipal_scope_weight(row: pd.Series, district: pd.DataFrame) -> float:
    """Approximate a municipality's province footprint from prior constituencies."""

    source_date = pd.Timestamp(row["source_date"])
    region_id = str(row["source_region_id"])
    municipality_key = _district_key(row.get("source_district_name", ""))
    eligible = district.loc[
        district["region_id"].astype(str).eq(region_id)
        & district["election_date"].le(source_date)
    ].copy()
    if eligible.empty:
        return 0.0
    latest = eligible["election_date"].max()
    units = eligible.loc[eligible["election_date"].eq(latest)].drop_duplicates(
        ["region_id", "district_key"]
    )
    regional_votes = float(units["district_valid_votes"].sum())
    if regional_votes <= 0.0:
        return 0.0
    if municipality_key:
        matched = units.loc[
            units["district_key"].astype(str).str.startswith(municipality_key)
        ]
    else:
        matched = units.iloc[0:0]
    if matched.empty:
        district_count = int(units["district_key"].nunique())
        footprint = 1.0 / max(district_count, 1)
    else:
        footprint = float(matched["district_valid_votes"].sum()) / regional_votes
    return float(np.sqrt(np.clip(footprint, 0.0, 1.0)))


def build_district_reconstructed_candidate_base(
    candidate_history: pd.DataFrame,
    district_history: pd.DataFrame,
    candidate_context: pd.DataFrame,
    *,
    half_life_years: float = 12.0,
    footprint_controlled: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate regional bases before rolling them up to 17 provinces."""

    if candidate_history.empty or district_history.empty or candidate_context.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), pd.DataFrame()
    target_dates = (
        candidate_history[["target_election_id", "target_election_date"]]
        .drop_duplicates("target_election_id")
        .assign(
            target_election_date=lambda frame: pd.to_datetime(
                frame["target_election_date"], errors="coerce"
            )
        )
        .set_index("target_election_id")["target_election_date"]
        .to_dict()
    )
    context = _target_context(candidate_context, target_dates)
    district, bloc_region = _district_statistics(district_history)
    candidate = candidate_history.copy()
    candidate = candidate.loc[candidate["source_is_prior"].map(_bool)].copy()
    candidate["source_date"] = pd.to_datetime(candidate["source_election_date"])
    candidate["target_date"] = pd.to_datetime(candidate["target_election_date"])
    candidate["district_key"] = candidate["source_district_name"].map(_district_key)
    candidate["source_region_id"] = candidate["source_region_name"].map(
        dict(zip(district["region_name"], district["region_id"]))
    )
    candidate = candidate.merge(
        context,
        left_on=["target_election_id", "target_slot", "target_candidate_name"],
        right_on=["election_id", "slot", "candidate_name"],
        how="left",
        suffixes=("", "_context"),
    )

    personal_candidates = candidate.loc[
        candidate["source_sg_typecode"].astype(str).eq("2")
        & candidate["district_key"].ne("")
    ].copy()
    personal_matches = personal_candidates.merge(
        district,
        left_on=[
            "source_date",
            "source_region_id",
            "district_key",
            "target_candidate_name",
        ],
        right_on=[
            "election_date",
            "region_id",
            "district_key",
            "candidate_name",
        ],
        how="inner",
        suffixes=("_candidate", "_district"),
    )
    personal_matches = personal_matches.merge(
        bloc_region[
            [
                "election_date",
                "region_id",
                "bloc",
                "region_bloc_share",
                "regional_valid_votes",
            ]
        ],
        left_on=["source_date", "region_id", "bloc_district"],
        right_on=["election_date", "region_id", "bloc"],
        how="left",
    )
    if not personal_matches.empty:
        age_years = (
            personal_matches["target_date"] - personal_matches["source_date"]
        ).dt.days / 365.2425
        personal_matches["recency"] = 0.5 ** (
            age_years / max(half_life_years, 1e-6)
        )
        personal_matches["personal_excess"] = (
            personal_matches["candidate_vote_share"]
            - personal_matches["region_bloc_share"].fillna(0.0)
        ).clip(lower=0.0)
        personal_matches["personal_signal_raw"] = (
            0.15 * personal_matches["candidate_vote_share"]
            + 0.85 * personal_matches["personal_excess"]
        ) * personal_matches["recency"]
        personal_matches["scope_weight"] = np.sqrt(
            np.divide(
                personal_matches["district_valid_votes"],
                personal_matches["regional_valid_votes"].replace(0.0, np.nan),
            ).fillna(0.0).clip(0.0, 1.0)
        )
        personal_matches["personal_signal"] = personal_matches[
            "personal_signal_raw"
        ] * (
            personal_matches["scope_weight"] if footprint_controlled else 1.0
        )

    executives = candidate.loc[
        candidate["source_sg_typecode"].astype(str).isin({"3", "4"})
        & candidate["source_region_id"].notna()
    ].copy()
    if footprint_controlled:
        executives = executives.loc[
            executives["prior_election_won"].astype(str).str.upper().eq("Y")
        ].copy()
    if not executives.empty:
        age_years = (executives["target_date"] - executives["source_date"]).dt.days / 365.2425
        executives["recency"] = 0.5 ** (age_years / max(half_life_years, 1e-6))
        executives["executive_signal_raw"] = (
            executives["source_sg_typecode"].astype(str).map({"3": 0.80, "4": 0.55})
            * executives["recency"]
            * executives["prior_election_won"].astype(str).str.upper().map({"Y": 1.0, "N": 0.70}).fillna(0.70)
        )
        executives["scope_weight"] = 1.0
        if footprint_controlled:
            municipal = executives["source_sg_typecode"].astype(str).eq("4")
            executives.loc[municipal, "scope_weight"] = executives.loc[
                municipal
            ].apply(lambda row: _municipal_scope_weight(row, district), axis=1)
        executives["executive_signal"] = executives["executive_signal_raw"] * (
            executives["scope_weight"] if footprint_controlled else 1.0
        )

    component_rows: list[dict[str, Any]] = []
    if not personal_matches.empty:
        for row in personal_matches.itertuples(index=False):
            component_rows.append(
                {
                    "election_id": str(row.target_election_id),
                    "slot": str(row.target_slot),
                    "candidate_name": str(row.target_candidate_name),
                    "region_id": str(row.region_id),
                    "component": "personal_constituency",
                    "signal": float(row.personal_signal),
                    "raw_signal": float(row.personal_signal_raw),
                    "scope_weight": float(row.scope_weight),
                    "available_date": str(row.available_date_candidate),
                    "confidence": float(row.entity_match_confidence),
                    "source_election_id": str(row.source_sg_id),
                }
            )
    if not executives.empty:
        for row in executives.itertuples(index=False):
            component_rows.append(
                {
                    "election_id": str(row.target_election_id),
                    "slot": str(row.target_slot),
                    "candidate_name": str(row.target_candidate_name),
                    "region_id": str(row.source_region_id),
                    "component": "executive_office",
                    "signal": float(row.executive_signal),
                    "raw_signal": float(row.executive_signal_raw),
                    "scope_weight": float(row.scope_weight),
                    "available_date": str(row.available_date),
                    "confidence": float(row.entity_match_confidence),
                    "source_election_id": str(row.source_sg_id),
                }
            )

    for target in context.itertuples(index=False):
        target_date = pd.Timestamp(target.target_date_context)
        bloc = normalize_bloc(target.bloc)
        if bloc in MAJOR_BLOCS or bloc in INDEPENDENT_BLOCS:
            continue
        eligible = bloc_region.loc[
            bloc_region["bloc"].eq(bloc)
            & bloc_region["election_date"].lt(target_date)
        ].copy()
        if eligible.empty:
            continue
        latest = eligible["election_date"].max()
        eligible = eligible.loc[eligible["election_date"].eq(latest)].copy()
        eligible["share_excess"] = (
            eligible["region_bloc_share"] - eligible["national_bloc_share"]
        ).clip(lower=0.0)
        eligible["coverage_excess"] = (
            eligible["coverage"] - eligible["national_coverage"]
        ).clip(lower=0.0)
        eligible["seat_excess"] = (
            eligible["seat_share"] - eligible["national_seat_share"]
        ).clip(lower=0.0)
        eligible["organization_signal"] = (
            0.55 * (eligible["share_excess"] / 0.25).clip(0.0, 1.0)
            + 0.25 * eligible["coverage_excess"].clip(0.0, 1.0)
            + 0.20 * eligible["seat_excess"].clip(0.0, 1.0)
        ) * float(target.organization_strength)
        eligible = eligible.loc[eligible["organization_signal"].gt(0.0)]
        for row in eligible.itertuples(index=False):
            component_rows.append(
                {
                    "election_id": str(target.election_id),
                    "slot": str(target.slot),
                    "candidate_name": str(target.candidate_name),
                    "region_id": str(row.region_id),
                    "component": "party_district_organization",
                    "signal": float(row.organization_signal),
                    "raw_signal": float(row.organization_signal),
                    "scope_weight": 1.0,
                    "available_date": (latest + pd.Timedelta(days=1)).date().isoformat(),
                    "confidence": float(target.context_confidence),
                    "source_election_id": str(row.election_id),
                }
            )

    components = pd.DataFrame(component_rows)
    if components.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), components
    pivot = components.pivot_table(
        index=["election_id", "slot", "candidate_name", "region_id"],
        columns="component",
        values="signal",
        aggfunc=_bounded_union if footprint_controlled else "sum",
        fill_value=0.0,
    ).reset_index()
    for component in [
        "personal_constituency",
        "executive_office",
        "party_district_organization",
    ]:
        if component not in pivot.columns:
            pivot[component] = 0.0
    metadata = components.groupby(
        ["election_id", "slot", "candidate_name", "region_id"], as_index=False
    ).agg(
        available_date=("available_date", "max"),
        confidence=("confidence", "mean"),
        personal_match_count=("component", lambda values: sum(value == "personal_constituency" for value in values)),
        source_election_ids=("source_election_id", lambda values: "|".join(sorted(set(map(str, values))))),
    )
    output = pivot.merge(
        metadata,
        on=["election_id", "slot", "candidate_name", "region_id"],
        how="left",
    )
    combined_signal = 1.0 - (
        (1.0 - output["personal_constituency"].clip(0.0, 0.75))
        * (1.0 - output["executive_office"].clip(0.0, 0.85))
        * (1.0 - output["party_district_organization"].clip(0.0, 0.85))
    )
    output["regional_affinity"] = combined_signal.clip(0.0, 0.85)
    output["organization_depth"] = (
        0.25
        + 0.35 * output["personal_constituency"].clip(0.0, 1.0)
        + 0.55 * output["executive_office"].clip(0.0, 1.0)
        + 0.65 * output["party_district_organization"].clip(0.0, 1.0)
    ).clip(0.0, 0.85)
    output["confidence"] = pd.to_numeric(output["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    output = output.rename(
        columns={
            "personal_constituency": "personal_constituency_signal",
            "executive_office": "executive_office_signal",
            "party_district_organization": "party_district_organization_signal",
        }
    )
    output["source_type"] = "constituency_reconstructed_candidate_base"
    output["notes"] = (
        "Footprint-controlled candidate excess plus winning executive office and "
        "non-major party constituency organization; repeated evidence uses bounded union"
        if footprint_controlled
        else "Candidate excess over regional bloc baseline plus executive office and "
        "non-major party constituency organization; rolled up after district scoring"
    )
    output["provenance_class"] = "official_deterministic_source_derived"
    output["derivation_version"] = SCHEMA_VERSION
    return output[OUTPUT_COLUMNS].sort_values(
        ["election_id", "slot", "regional_affinity"],
        ascending=[True, True, False],
    ).reset_index(drop=True), components
