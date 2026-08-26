"""Unified point-in-time regional identity from exact party lineages.

The historical party name is preserved until lineage-level regional excess is
estimated.  Broad camps are attached only as metadata and never used to merge
lineages before the regional calculation.  All regions and election families
use the same estimator; ballot type changes evidence reliability, not the
meaning of the regional state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from presidential_issue_engine.region_bloc_prior import (
    attach_bloc_prior as attach_projected_bloc_prior,
    election_date,
    normalize_bloc,
)


DIRECT_PARTY_TYPES = frozenset(
    {"national_assembly_pr", "assembly_pr", "metro_council_pr", "local_council_pr"}
)
EXCLUDED_TYPES = frozenset({"education_superintendent", "education_council"})
ASSEMBLY_DISTRICT_TYPES = frozenset({"assembly_district", "national_assembly_district"})
INDEPENDENT_NAMES = frozenset({"", "무소속", "무소속후보자", "계"})
COLLAPSED_THIRD_NAMES = frozenset({"제3지대", "제3지대 및 기타"})

# This is a measurement prior, not a fitted vote multiplier.  Paired
# proportional/candidate ballots replace it as soon as prior observations are
# available.
CANDIDATE_BALLOT_RELIABILITY_PRIOR = 0.25
RELIABILITY_PRIOR_OBSERVATIONS = 20.0

LINEAGE_ALIASES: dict[str, frozenset[str]] = {
    "mainstream_conservative": frozenset(
        {
            "민주자유당",
            "신한국당",
            "한나라당",
            "새누리당",
            "자유한국당",
            "미래통합당",
            "미래한국당",
            "국민의힘",
            "국민의미래",
        }
    ),
    "mainstream_liberal": frozenset(
        {
            "민주당",
            "새정치국민회의",
            "새천년민주당",
            "열린우리당",
            "대통합민주신당",
            "통합민주당",
            "민주통합당",
            "새정치민주연합",
            "더불어민주당",
            "더불어민주연합",
            "더불어시민당",
        }
    ),
    "mainstream_progressive": frozenset(
        {
            "민주노동당",
            "진보신당",
            "통합진보당",
            "정의당",
            "녹색정의당",
            "진보당",
            "민중당",
        }
    ),
    "chungcheong_regionalist": frozenset(
        {
            "자유민주연합",
            "자민련",
            "국민중심당",
            "국민중심연합",
            "자유선진당",
            "선진통일당",
            "충청의미래당",
        }
    ),
    "honam_regionalist": frozenset({"민주평화당", "대안신당"}),
}

PARTY_TO_LINEAGE = {
    party: lineage
    for lineage, parties in LINEAGE_ALIASES.items()
    for party in parties
}


@dataclass(frozen=True)
class LineageFit:
    profiles: pd.DataFrame
    type_reliability: pd.DataFrame
    half_life_years: float


def party_lineage(value: object) -> str:
    """Return a stable lineage while retaining unknown parties exactly."""

    text = "" if pd.isna(value) else str(value).strip()
    if text in INDEPENDENT_NAMES:
        return "independent"
    if text in COLLAPSED_THIRD_NAMES:
        return "unresolved_third"
    return PARTY_TO_LINEAGE.get(text, f"party:{text}")


def _centered_profile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return numeric - float(numeric.median())


def _profile_similarity(left: pd.Series, right: pd.Series) -> float:
    common = sorted(set(left.index).intersection(right.index))
    if len(common) < 3:
        return 0.0
    a = _centered_profile(left.reindex(common)).to_numpy(float)
    b = _centered_profile(right.reindex(common)).to_numpy(float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _source_party_profiles(group: pd.DataFrame) -> dict[str, pd.Series]:
    values = (
        group.groupby(["region_id", "source_party_name"])["regional_votes_or_share"]
        .sum()
        .unstack(fill_value=0.0)
    )
    totals = values.sum(axis=1).replace(0.0, np.nan)
    shares = values.div(totals, axis=0).fillna(0.0)
    return {
        str(party): shares[party].copy()
        for party in shares.columns
    }


def _resolve_collapsed_lineages(work: pd.DataFrame) -> pd.DataFrame:
    """Resolve collapsed third-bloc rows from completed exact spatial profiles.

    Events are processed chronologically. Exact parties published in the same
    completed event may identify a collapsed companion row, but no later event
    can change an earlier resolution.
    """

    out = work.copy()
    # Keep measurement quality separate from lineage-resolution confidence.
    # A collapsed label can still be a reliable observed broad-party vote even
    # when its exact historical successor is uncertain.
    out["source_quality"] = pd.to_numeric(
        out["quality"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["lineage_id"] = out["source_party_name"].map(party_lineage)
    out["lineage_resolution"] = np.where(
        out["lineage_id"].eq("unresolved_third"),
        "unresolved_collapsed_label",
        "exact_party_alias",
    )
    out["lineage_resolution_confidence"] = np.where(
        out["lineage_id"].eq("unresolved_third"), 0.25, 1.0
    )
    # Major and progressive camp aggregates are semantically incompatible with
    # a collapsed third-party row. Every other named lineage competes under the
    # same spatial-similarity and margin thresholds.
    ineligible_lineages = {
        "mainstream_conservative",
        "mainstream_liberal",
        "mainstream_progressive",
        "independent",
        "unresolved_third",
    }
    prior_profiles: dict[str, list[tuple[pd.Timestamp, pd.Series]]] = {}
    event_keys = (
        out[["event_date", "election_id", "election_type"]]
        .drop_duplicates()
        .sort_values(["event_date", "election_id", "election_type"])
    )
    for event in event_keys.itertuples(index=False):
        current_date = pd.Timestamp(event.event_date)
        mask = (
            out["event_date"].eq(pd.Timestamp(event.event_date))
            & out["election_id"].astype(str).eq(str(event.election_id))
            & out["election_type"].astype(str).eq(str(event.election_type))
        )
        group = out.loc[mask].copy()
        profiles = _source_party_profiles(group)
        # Update the exact reference pool first. Same-date exact party ballots
        # are public together with a collapsed companion ballot and may resolve
        # that companion for future forecasts.
        observed_lineages = sorted(
            set(group["lineage_id"].astype(str)) - ineligible_lineages
        )
        for lineage in observed_lineages:
            parties = group.loc[
                group["lineage_id"].eq(lineage), "source_party_name"
            ].drop_duplicates()
            lineage_profiles = [profiles[str(party)] for party in parties if str(party) in profiles]
            if lineage_profiles:
                combined = pd.concat(lineage_profiles, axis=1).sum(axis=1)
                prior_profiles.setdefault(lineage, []).append(
                    (current_date, combined)
                )

        generic_parties = group.loc[
            group["lineage_id"].eq("unresolved_third"), "source_party_name"
        ].drop_duplicates()
        for party in generic_parties:
            profile = profiles.get(str(party))
            if profile is None:
                continue
            candidates: list[tuple[str, float, int, bool]] = []
            for lineage, history in prior_profiles.items():
                recent = [
                    (date, values)
                    for date, values in history
                    if 0.0 <= (current_date - date).days / 365.25 <= 8.0
                ]
                if not recent:
                    continue
                evidence_dates = {date for date, _ in recent}
                centroid = pd.concat(
                    [values for _, values in recent], axis=1
                ).mean(axis=1)
                candidates.append(
                    (
                        lineage,
                        _profile_similarity(profile, centroid),
                        len(evidence_dates),
                        any(date == current_date for date, _ in recent),
                    )
                )
            candidates.sort(key=lambda item: item[1], reverse=True)
            if not candidates:
                continue
            (
                best_lineage,
                best_similarity,
                evidence_events,
                same_date_evidence,
            ) = candidates[0]
            runner_up = candidates[1][1] if len(candidates) > 1 else 0.0
            if best_similarity < 0.60 or best_similarity - runner_up < 0.10:
                continue
            if not same_date_evidence and evidence_events < 2:
                continue
            evidence_reliability = (
                0.85
                if same_date_evidence
                else evidence_events / (evidence_events + 1.5)
            )
            confidence = float(
                np.clip(best_similarity * evidence_reliability, 0.25, 0.95)
            )
            party_mask = mask & out["source_party_name"].eq(str(party))
            out.loc[party_mask, "lineage_id"] = best_lineage
            out.loc[party_mask, "lineage_resolution"] = "prior_exact_spatial_profile"
            out.loc[party_mask, "lineage_resolution_confidence"] = confidence
            # Once resolved, the event becomes additional, lower-confidence
            # evidence for subsequent collapsed labels.
            prior_profiles[best_lineage].append(
                (current_date, profile * confidence)
            )
    out["quality"] = (
        out["source_quality"]
        * pd.to_numeric(
            out["lineage_resolution_confidence"], errors="coerce"
        ).fillna(0.0)
    )
    return out


def _base_history_rows(
    history: pd.DataFrame,
    *,
    date_resolver: Callable[[object], pd.Timestamp | None],
) -> pd.DataFrame:
    required = {
        "election_id",
        "election_type",
        "region_id",
        "bloc",
        "vote_share",
        "data_quality_weight",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"lineage history missing columns: {sorted(missing)}")
    out = history.loc[
        ~history["election_type"].astype(str).isin(EXCLUDED_TYPES | ASSEMBLY_DISTRICT_TYPES)
    ].copy()
    out["source_party_name"] = out["bloc"].fillna("").astype(str).str.strip()
    out["regional_votes_or_share"] = pd.to_numeric(
        out["vote_share"], errors="coerce"
    ).fillna(0.0)
    out["quality"] = pd.to_numeric(
        out["data_quality_weight"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["event_date"] = pd.to_datetime(
        out["election_id"].map(date_resolver), errors="coerce"
    )
    out = out.loc[
        out["event_date"].notna() & out["source_party_name"].ne("")
    ].copy()
    out["source_detail"] = "preserved_history_party_name"
    return out[
        [
            "election_id",
            "election_type",
            "event_date",
            "region_id",
            "source_party_name",
            "regional_votes_or_share",
            "quality",
            "source_detail",
        ]
    ]


def _assembly_history_rows(assembly_history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "election_id",
        "election_date",
        "region_id",
        "party_name",
        "candidate_votes",
    }
    missing = required - set(assembly_history.columns)
    if missing:
        raise ValueError(f"assembly lineage history missing columns: {sorted(missing)}")
    raw = assembly_history.copy()
    raw["event_date"] = pd.to_datetime(raw["election_date"], errors="coerce")
    raw["source_party_name"] = raw["party_name"].fillna("").astype(str).str.strip()
    raw["candidate_votes"] = pd.to_numeric(
        raw["candidate_votes"], errors="coerce"
    ).fillna(0.0)
    raw = raw.loc[
        raw["event_date"].notna()
        & raw["region_id"].notna()
        & raw["source_party_name"].ne("")
        & ~raw["source_party_name"].isin(INDEPENDENT_NAMES)
    ].copy()
    out = (
        raw.groupby(
            [
                "election_id",
                "event_date",
                "region_id",
                "source_party_name",
            ],
            as_index=False,
        )["candidate_votes"]
        .sum()
        .rename(columns={"candidate_votes": "regional_votes_or_share"})
    )
    out["election_type"] = "assembly_district"
    out["quality"] = 0.65
    out["source_detail"] = "nec_exact_constituency_party"
    return out[
        [
            "election_id",
            "election_type",
            "event_date",
            "region_id",
            "source_party_name",
            "regional_votes_or_share",
            "quality",
            "source_detail",
        ]
    ]


def _complete_event_distribution(group: pd.DataFrame) -> pd.DataFrame:
    election_id = str(group["election_id"].iloc[0])
    election_type = str(group["election_type"].iloc[0])
    event_date = pd.Timestamp(group["event_date"].iloc[0])
    regions = sorted(group["region_id"].astype(str).unique())
    lineages = sorted(group["lineage_id"].astype(str).unique())
    index = pd.MultiIndex.from_product(
        [regions, lineages], names=["region_id", "lineage_id"]
    )
    summed = (
        group.groupby(["region_id", "lineage_id"], as_index=True)
        .agg(
            regional_value=("regional_votes_or_share", "sum"),
            quality=("quality", "mean"),
            source_quality=("source_quality", "mean"),
            broad_bloc=("broad_bloc", "first"),
            source_party_names=(
                "source_party_name",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            source_detail=(
                "source_detail",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            lineage_resolution=(
                "lineage_resolution",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            lineage_resolution_confidence=(
                "lineage_resolution_confidence",
                "mean",
            ),
        )
        .reindex(index)
        .reset_index()
    )
    summed["regional_value"] = summed["regional_value"].fillna(0.0)
    event_quality = float(pd.to_numeric(group["quality"], errors="coerce").mean())
    event_source_quality = float(
        pd.to_numeric(group["source_quality"], errors="coerce").mean()
    )
    summed["quality"] = summed["quality"].fillna(event_quality)
    summed["source_quality"] = summed["source_quality"].fillna(
        event_source_quality
    )
    summed["broad_bloc"] = summed["broad_bloc"].fillna("")
    summed["source_party_names"] = summed["source_party_names"].fillna("")
    summed["source_detail"] = summed["source_detail"].fillna("")
    summed["lineage_resolution"] = summed["lineage_resolution"].fillna("")
    summed["lineage_resolution_confidence"] = pd.to_numeric(
        summed["lineage_resolution_confidence"], errors="coerce"
    ).fillna(0.0)
    totals = summed.groupby("region_id")["regional_value"].transform("sum")
    summed["regional_share"] = np.divide(
        summed["regional_value"],
        totals,
        out=np.zeros(len(summed), dtype=float),
        where=totals.to_numpy(float) > 0.0,
    )
    # This is a regional deviation feature, not a national party-strength
    # feature. Equal-region means preserve the legacy prior contract: every
    # lineage's regional gaps are centered at zero within each election.
    references = summed.groupby("lineage_id")["regional_share"].mean()
    summed["national_reference"] = summed["lineage_id"].map(references).fillna(0.0)
    summed["lineage_gap"] = (
        summed["regional_share"] - summed["national_reference"]
    ).clip(-0.50, 0.50)
    summed["distinctiveness"] = 0.5 * summed.groupby("region_id")[
        "lineage_gap"
    ].transform(lambda values: float(values.abs().sum()))
    summed["election_id"] = election_id
    summed["election_type"] = election_type
    summed["event_date"] = event_date
    summed["available_date"] = event_date + pd.Timedelta(days=1)
    summed["ballot_channel"] = (
        "direct_party"
        if election_type in DIRECT_PARTY_TYPES
        else "candidate_proxy"
    )
    return summed


def build_exact_lineage_events(
    history: pd.DataFrame,
    assembly_history: pd.DataFrame,
    *,
    date_resolver: Callable[[object], pd.Timestamp | None] = election_date,
) -> pd.DataFrame:
    """Build a complete region-by-lineage event ledger without early bloc collapse."""

    base = _base_history_rows(history, date_resolver=date_resolver)
    assembly = _assembly_history_rows(assembly_history)
    work = pd.concat([base, assembly], ignore_index=True)
    work = _resolve_collapsed_lineages(work)
    work["broad_bloc"] = work["source_party_name"].map(normalize_bloc)
    pieces = [
        _complete_event_distribution(group)
        for _, group in work.groupby(
            ["election_id", "election_type", "event_date"], sort=True
        )
    ]
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    columns = [
        "election_id",
        "election_type",
        "event_date",
        "available_date",
        "region_id",
        "lineage_id",
        "broad_bloc",
        "source_party_names",
        "regional_share",
        "national_reference",
        "lineage_gap",
        "distinctiveness",
        "quality",
        "source_quality",
        "ballot_channel",
        "source_detail",
        "lineage_resolution",
        "lineage_resolution_confidence",
    ]
    return out[columns].sort_values(
        ["event_date", "election_type", "region_id", "lineage_id"]
    ).reset_index(drop=True)


def _paired_type_observations(
    prior: pd.DataFrame, candidate_type: str
) -> pd.DataFrame:
    candidate = prior.loc[prior["election_type"].eq(candidate_type)].copy()
    direct = prior.loc[prior["election_type"].isin(DIRECT_PARTY_TYPES)].copy()
    if candidate.empty or direct.empty:
        return pd.DataFrame(columns=["candidate_gap", "direct_gap"])
    direct = (
        direct.groupby(["event_date", "region_id", "lineage_id"], as_index=False)[
            "lineage_gap"
        ]
        .mean()
        .rename(columns={"lineage_gap": "direct_gap"})
    )
    candidate = candidate.rename(columns={"lineage_gap": "candidate_gap"})
    paired = candidate.merge(
        direct,
        on=["event_date", "region_id", "lineage_id"],
        how="inner",
    )
    informative = paired["candidate_gap"].abs() + paired["direct_gap"].abs()
    return paired.loc[informative.gt(1e-8), ["candidate_gap", "direct_gap"]]


def estimate_type_reliability(
    events: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Estimate candidate-ballot reliability from prior same-date party ballots."""

    prior = events.loc[events["event_date"].lt(pd.Timestamp(cutoff))].copy()
    rows: list[dict[str, object]] = []
    for election_type in sorted(events["election_type"].astype(str).unique()):
        if election_type in DIRECT_PARTY_TYPES:
            rows.append(
                {
                    "election_type": election_type,
                    "type_reliability": 1.0,
                    "paired_observations": 0,
                    "paired_correlation": np.nan,
                    "source": "direct_party_ballot",
                }
            )
            continue
        paired = _paired_type_observations(prior, election_type)
        n = len(paired)
        correlation = np.nan
        if (
            n >= 3
            and float(paired["candidate_gap"].std(ddof=0)) > 1e-12
            and float(paired["direct_gap"].std(ddof=0)) > 1e-12
        ):
            correlation = float(
                paired[["candidate_gap", "direct_gap"]].corr().iloc[0, 1]
            )
        evidence = 0.0 if not np.isfinite(correlation) else float(np.clip(correlation, 0.0, 1.0))
        reliability = (
            CANDIDATE_BALLOT_RELIABILITY_PRIOR * RELIABILITY_PRIOR_OBSERVATIONS
            + evidence * n
        ) / (RELIABILITY_PRIOR_OBSERVATIONS + n)
        rows.append(
            {
                "election_type": election_type,
                "type_reliability": float(np.clip(reliability, 0.05, 1.0)),
                "paired_observations": int(n),
                "paired_correlation": correlation,
                "source": "prior_same_date_direct_party_pair"
                if n
                else "candidate_ballot_measurement_prior",
            }
        )
    return pd.DataFrame(rows)


def fit_lineage_profiles(
    events: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    half_life_years: float = 12.0,
    prior_strength: float = 1.5,
) -> LineageFit:
    """Fit exact-lineage regional profiles strictly before ``cutoff``."""

    prior = events.loc[events["event_date"].lt(pd.Timestamp(cutoff))].copy()
    type_reliability = estimate_type_reliability(events, cutoff=pd.Timestamp(cutoff))
    if prior.empty:
        return LineageFit(pd.DataFrame(), type_reliability, half_life_years)
    prior = prior.merge(type_reliability, on="election_type", how="left")
    age_years = (pd.Timestamp(cutoff) - prior["event_date"]).dt.days / 365.25
    prior["weight"] = (
        prior["quality"]
        * prior["type_reliability"].fillna(CANDIDATE_BALLOT_RELIABILITY_PRIOR)
        * np.exp(-np.log(2.0) * age_years / max(float(half_life_years), 0.1))
    )
    rows: list[dict[str, object]] = []
    for (region_id, lineage_id), group in prior.groupby(
        ["region_id", "lineage_id"], sort=True
    ):
        weights = group["weight"].to_numpy(float)
        values = group["lineage_gap"].to_numpy(float)
        total = float(weights.sum())
        if total <= 0.0:
            continue
        gap = float(np.average(values, weights=weights))
        mad = float(np.average(np.abs(values - gap), weights=weights))
        effective_n = total**2 / max(float(np.square(weights).sum()), 1e-12)
        reliability = effective_n / (effective_n + max(prior_strength, 1e-6))
        reliability *= float(
            np.clip(1.0 - mad / max(abs(gap) + 0.05, 0.05), 0.20, 1.0)
        )
        rows.append(
            {
                "region_id": str(region_id),
                "lineage_id": str(lineage_id),
                "lineage_gap": float(np.clip(gap, -0.40, 0.40)),
                "lineage_reliability": float(np.clip(reliability, 0.0, 1.0)),
                "effective_n": effective_n,
                "prior_events": int(group["election_id"].nunique()),
                "source_party_names": "|".join(
                    sorted(
                        {
                            party.strip()
                            for value in group["source_party_names"].fillna("")
                            for party in str(value).split("|")
                            if party.strip()
                        }
                    )
                ),
            }
        )
    profiles = pd.DataFrame(rows)
    if not profiles.empty:
        profiles["weighted_gap"] = (
            profiles["lineage_gap"] * profiles["lineage_reliability"]
        )
        profiles["regional_distinctiveness"] = 0.5 * profiles.groupby("region_id")[
            "weighted_gap"
        ].transform(lambda values: float(values.abs().sum()))
    return LineageFit(profiles, type_reliability, half_life_years)


def _candidate_names(frame: pd.DataFrame) -> pd.Series:
    for column in ("candidate_name", "candidate_name_x", "candidate_name_y"):
        if column in frame.columns:
            return frame[column].fillna("").astype(str)
    raise ValueError("lineage identity frame has no candidate-name column")


def _eligible_evidence(
    source: pd.DataFrame,
    election_id: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    if source.empty or "election_id" not in source.columns:
        return source.iloc[0:0].copy()
    out = source.loc[source["election_id"].astype(str).eq(election_id)].copy()
    out["available_date"] = pd.to_datetime(out.get("available_date"), errors="coerce")
    return out.loc[
        out["available_date"].notna() & out["available_date"].lt(pd.Timestamp(cutoff))
    ].copy()


def _candidate_party_lineages(
    group: pd.DataFrame,
    candidate_parties: pd.DataFrame,
) -> list[str]:
    names = _candidate_names(group)
    lookup = candidate_parties.copy()
    lookup = lookup.loc[
        lookup["election_id"].astype(str).eq(str(group["election_id"].iloc[0]))
    ].copy()
    lookup = lookup.drop_duplicates(["slot", "candidate_name"])
    by_key = {
        (str(row.slot), str(row.candidate_name)): party_lineage(row.party_name)
        for row in lookup.itertuples(index=False)
    }
    by_name = {
        str(candidate_name): party_lineage(party_rows.iloc[0]["party_name"])
        for candidate_name, party_rows in lookup.groupby(
            "candidate_name", sort=False
        )
        if len(party_rows) == 1
    }
    return [
        by_key.get(
            (str(slot), str(name)),
            by_name.get(str(name), party_lineage(bloc)),
        )
        for slot, name, bloc in zip(
            group["slot"], names, group.get("bloc", pd.Series("", index=group.index)), strict=False
        )
    ]


def _candidate_party_names(
    group: pd.DataFrame,
    candidate_parties: pd.DataFrame,
) -> list[str]:
    names = _candidate_names(group)
    lookup = candidate_parties.loc[
        candidate_parties["election_id"].astype(str).eq(
            str(group["election_id"].iloc[0])
        )
    ].drop_duplicates(["slot", "candidate_name"])
    by_key = {
        (str(row.slot), str(row.candidate_name)): str(row.party_name).strip()
        for row in lookup.itertuples(index=False)
    }
    by_name = {
        str(candidate_name): str(party_rows.iloc[0]["party_name"]).strip()
        for candidate_name, party_rows in lookup.groupby(
            "candidate_name", sort=False
        )
        if len(party_rows) == 1
    }
    return [
        by_key.get(
            (str(slot), str(name)),
            by_name.get(str(name), str(bloc).strip()),
        )
        for slot, name, bloc in zip(
            group["slot"],
            names,
            group.get("bloc", pd.Series("", index=group.index)),
            strict=False,
        )
    ]


def party_genealogy_affinity(
    transitions: pd.DataFrame,
    source_parties: list[str] | tuple[str, ...] | set[str],
    target_party: str,
    *,
    cutoff: pd.Timestamp,
) -> float:
    """Return dated predecessor-to-successor continuity without vote fitting."""

    target = str(target_party).strip()
    sources = {str(value).strip() for value in source_parties if str(value).strip()}
    if not target or not sources:
        return 0.0
    if target in sources:
        return 1.0
    if transitions.empty:
        return 0.0
    required = {
        "predecessor_party",
        "successor_party",
        "effective_date",
        "continuity",
        "confidence",
    }
    missing = required - set(transitions.columns)
    if missing:
        raise ValueError(f"party transitions missing columns: {sorted(missing)}")
    work = transitions.copy()
    work["effective_date"] = pd.to_datetime(work["effective_date"], errors="coerce")
    work = work.loc[
        work["effective_date"].notna()
        & work["effective_date"].lt(pd.Timestamp(cutoff))
    ].copy()
    if work.empty:
        return 0.0
    work["edge_affinity"] = (
        pd.to_numeric(work["continuity"], errors="coerce").fillna(0.0)
        * pd.to_numeric(work["confidence"], errors="coerce").fillna(0.0)
    ).clip(0.0, 1.0)
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for row in work.itertuples(index=False):
        predecessor = str(row.predecessor_party).strip()
        successor = str(row.successor_party).strip()
        if predecessor and successor and float(row.edge_affinity) > 0.0:
            adjacency.setdefault(predecessor, []).append(
                (successor, float(row.edge_affinity))
            )
    best = {source: 1.0 for source in sources}
    frontier = list(sources)
    while frontier:
        current = frontier.pop()
        current_affinity = best[current]
        for successor, edge_affinity in adjacency.get(current, []):
            candidate = current_affinity * edge_affinity
            if candidate > best.get(successor, 0.0) + 1e-12:
                best[successor] = candidate
                frontier.append(successor)
    return float(np.clip(best.get(target, 0.0), 0.0, 1.0))


def attach_exact_lineage_prior(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    candidate_parties: pd.DataFrame,
    *,
    half_life_years: float = 12.0,
    prior_strength: float = 1.5,
) -> pd.DataFrame:
    """Attach the Ridge regional prior from the same exact-lineage ledger.

    Candidate identity comes from the dated ballot registry. Historical votes
    are filtered inside ``fit_lineage_profiles`` to events strictly before the
    target election. The output column names intentionally match the legacy
    bloc-prior contract so the nested Ridge is refit without a parallel regional
    estimator.
    """

    out = frame.copy()
    output_columns = (
        "bloc_loyalty",
        "bloc_strength",
        "partisan_prior",
        "effective_election_count",
    )
    for column in output_columns:
        out[column] = 0.0
    out["exact_lineage_id"] = ""
    out["exact_lineage_prior_reliability"] = 0.0
    if out.empty or events.empty:
        return out

    for election_id, labels in out.groupby("election_id", sort=False).groups.items():
        cutoff = election_date(str(election_id))
        if cutoff is None:
            continue
        idx = pd.Index(labels)
        group = out.loc[idx].copy()
        fit = fit_lineage_profiles(
            events,
            cutoff=pd.Timestamp(cutoff),
            half_life_years=half_life_years,
            prior_strength=prior_strength,
        )
        if fit.profiles.empty:
            continue
        lineages = np.asarray(
            _candidate_party_lineages(group, candidate_parties), dtype=object
        )
        out.loc[idx, "exact_lineage_id"] = lineages
        represented = set(map(str, lineages))
        profiles = fit.profiles.loc[
            fit.profiles["lineage_id"].astype(str).isin(represented)
        ].copy()
        if profiles.empty:
            continue
        loyalty_lookup = profiles.set_index(["region_id", "lineage_id"])[
            "lineage_gap"
        ].to_dict()
        reliability_lookup = profiles.set_index(["region_id", "lineage_id"])[
            "lineage_reliability"
        ].to_dict()
        count_lookup = profiles.set_index(["region_id", "lineage_id"])[
            "prior_events"
        ].to_dict()
        strength_lookup = (
            profiles.assign(abs_gap=profiles["lineage_gap"].abs())
            .groupby("region_id")["abs_gap"]
            .max()
            .clip(0.0, 0.40)
            .to_dict()
        )
        regions = group["region_id"].astype(str).to_numpy()
        loyalty = np.asarray(
            [
                float(loyalty_lookup.get((region, str(lineage)), 0.0))
                for region, lineage in zip(regions, lineages, strict=False)
            ],
            dtype=float,
        )
        strength = np.asarray(
            [float(strength_lookup.get(region, 0.0)) for region in regions],
            dtype=float,
        )
        counts = np.asarray(
            [
                float(count_lookup.get((region, str(lineage)), 0.0))
                for region, lineage in zip(regions, lineages, strict=False)
            ],
            dtype=float,
        )
        reliability = np.asarray(
            [
                float(reliability_lookup.get((region, str(lineage)), 0.0))
                for region, lineage in zip(regions, lineages, strict=False)
            ],
            dtype=float,
        )
        out.loc[idx, "bloc_loyalty"] = loyalty
        out.loc[idx, "bloc_strength"] = strength
        out.loc[idx, "partisan_prior"] = loyalty * (1.0 + strength)
        out.loc[idx, "effective_election_count"] = counts
        out.loc[idx, "exact_lineage_prior_reliability"] = reliability
    return out


def _analytic_bloc_from_lineage(lineage_id: object) -> str:
    lineage = str(lineage_id)
    fixed = {
        "mainstream_conservative": "\uad6d\ubbfc\uc758\ud798",
        "mainstream_liberal": "\ub354\ubd88\uc5b4\ubbfc\uc8fc\ub2f9",
        "mainstream_progressive": "\uc9c4\ubcf4\uc815\ub2f9\uacc4",
        "chungcheong_regionalist": "\uc81c3\uc9c0\ub300",
        "honam_regionalist": "\uc81c3\uc9c0\ub300",
        "unresolved_third": "\uc81c3\uc9c0\ub300",
        "independent": "\ubb34\uc18c\uc18d",
    }
    if lineage in fixed:
        return fixed[lineage]
    if lineage.startswith("party:"):
        return normalize_bloc(lineage.removeprefix("party:"))
    return normalize_bloc(lineage)


def project_lineage_events_to_bloc_history(events: pd.DataFrame) -> pd.DataFrame:
    """Project the preserved lineage ledger only at the Ridge feature boundary."""

    if events.empty:
        return pd.DataFrame(
            columns=[
                "election_id",
                "election_type",
                "region_id",
                "bloc",
                "vote_share",
                "data_quality_weight",
            ]
        )
    work = events.loc[
        pd.to_numeric(events["regional_share"], errors="coerce").fillna(0.0).gt(0.0)
    ].copy()
    source_bloc = work.get("broad_bloc", pd.Series("", index=work.index))
    source_bloc = source_bloc.fillna("").astype(str).str.strip()
    fallback = work["lineage_id"].map(_analytic_bloc_from_lineage)
    work["bloc"] = source_bloc.where(source_bloc.ne(""), fallback)
    out = (
        work.groupby(
            ["election_id", "election_type", "region_id", "bloc"],
            as_index=False,
        )
        .agg(
            vote_share=("regional_share", "sum"),
            data_quality_weight=("source_quality", "max"),
        )
        .sort_values(["election_id", "region_id", "bloc"])
        .reset_index(drop=True)
    )
    return out


def attach_lineage_projected_prior(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    candidate_parties: pd.DataFrame,
    election_order: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Attach the legacy-scale prior from a final projection of one ledger."""

    history = project_lineage_events_to_bloc_history(events)
    out = attach_projected_bloc_prior(frame, history, list(election_order))
    out["exact_lineage_id"] = ""
    for _, labels in out.groupby("election_id", sort=False).groups.items():
        idx = pd.Index(labels)
        out.loc[idx, "exact_lineage_id"] = _candidate_party_lineages(
            out.loc[idx], candidate_parties
        )
    return out


def _candidate_base_scores(
    group: pd.DataFrame,
    *,
    election_id: str,
    region_id: str,
    cutoff: pd.Timestamp,
    candidate_regional_base: pd.DataFrame,
) -> np.ndarray:
    names = _candidate_names(group)
    score = np.zeros(len(group), dtype=float)
    base = _eligible_evidence(candidate_regional_base, election_id, cutoff)
    if base.empty:
        return score
    base = base.loc[base["region_id"].astype(str).eq(region_id)].copy()
    base["weighted_affinity"] = (
        pd.to_numeric(base.get("regional_affinity"), errors="coerce").fillna(0.0)
        * pd.to_numeric(base.get("organization_depth"), errors="coerce").fillna(0.0)
        * pd.to_numeric(base.get("confidence"), errors="coerce").fillna(0.0)
    ).clip(0.0, 1.0)
    for row in base.itertuples(index=False):
        score = np.maximum(
            score,
            np.where(names.eq(str(row.candidate_name)), float(row.weighted_affinity), 0.0),
        )
    return score


def _alignment_scores(
    group: pd.DataFrame,
    *,
    election_id: str,
    region_id: str,
    cutoff: pd.Timestamp,
    alignment: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    names = _candidate_names(group)
    score = np.zeros(len(group), dtype=float)
    evidence = ["none"] * len(group)
    routed = _eligible_evidence(alignment, election_id, cutoff)
    if routed.empty:
        return score, evidence
    scope = routed.get("region_scope", pd.Series("", index=routed.index)).astype(str)
    chungcheong = region_id in {"sido_30", "sido_36", "sido_43", "sido_44"}
    accepted = {region_id, "all", "national"}
    if chungcheong:
        accepted.add("chungcheong")
    routed = routed.loc[scope.isin(accepted)].copy()
    routed["weighted_affinity"] = (
        pd.to_numeric(routed.get("affinity"), errors="coerce").fillna(0.0)
        * pd.to_numeric(routed.get("confidence"), errors="coerce").fillna(0.0)
    ).clip(0.0, 1.0)
    for row in routed.itertuples(index=False):
        match = names.eq(str(row.candidate_name)).to_numpy()
        for pos in np.flatnonzero(match):
            value = float(row.weighted_affinity)
            if value > score[pos]:
                score[pos] = value
                evidence[pos] = str(row.evidence_type)
    return score, evidence


#: Off by default. When false the routing resolves dates exactly as V31 did -
#: through region_bloc_prior alone - and skips a target it cannot date, which is
#: what every frozen artifact through V31 was produced with. V32 switches it on
#: from its runner.
#:
#: This is a seam rather than an outright fix because the change is not
#: confined to the version making it: turning it on unconditionally moved V31's
#: frozen 2025 forecast by 0.00335, caught by prospective-reproduction. The
#: scored panel genuinely cannot move - all five scored elections resolve
#: through the original map - but the prospective target is precisely the case
#: that took the skip.
ROUTING_REQUIRES_DATABLE_TARGET = False


def _routing_cutoff(election_id: str):
    """Resolve a routed election's cutoff, central registry first.

    ``region_bloc_prior`` keeps its own presidential date map and it stopped at
    2022 while ``election_scope`` already carried pres_2025. The routing dated
    each election through that map and skipped what it could not date, so the
    2025 forecast carried five lineage_identity_* columns at zero - the value
    they are initialised to - and nothing said so.

    The fallback only fires where the old code returned ``None`` and skipped,
    so it cannot change a scored election: all five resolve through the
    original map, and both registries agree on every one of their dates.
    """

    date = election_date(election_id)
    if date is not None:
        return date
    if not ROUTING_REQUIRES_DATABLE_TARGET:
        return None
    from presidential_issue_engine import election_scope

    return election_scope.ELECTION_DATES.get(election_id)


def apply_unified_lineage_routing(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    candidate_regional_base: pd.DataFrame,
    alignment: pd.DataFrame,
    candidate_parties: pd.DataFrame,
    party_transitions: pd.DataFrame | None = None,
    *,
    prediction_column: str,
    gain: float,
    shift_cap: float = 0.08,
    half_life_years: float = 12.0,
    prior_strength: float = 1.5,
    include_direct_lineage_score: bool = True,
    direct_lineage_scope: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply one lineage-based regional adjustment to every region."""

    out = frame.copy().reset_index(drop=True)
    for column, default in (
        ("lineage_identity_distinctiveness", 0.0),
        ("lineage_identity_score", 0.0),
        ("lineage_identity_log_shift", 0.0),
        ("lineage_identity_transfer", 0.0),
        ("lineage_identity_gain", 0.0),
        ("lineage_identity_candidate_lineage", ""),
        ("lineage_identity_evidence", "none"),
    ):
        out[column] = default
    audits: list[dict[str, object]] = []
    reliability_audits: list[pd.DataFrame] = []
    effective_gain = float(np.clip(gain, 0.0, 1.0))
    effective_cap = float(np.clip(shift_cap, 0.0, 0.15))
    if direct_lineage_scope not in {"all", "non_major", "none"}:
        raise ValueError(f"unsupported direct lineage scope: {direct_lineage_scope}")
    if effective_gain <= 0.0 or events.empty:
        return out, pd.DataFrame(audits), pd.DataFrame()

    for election_id, election_idx in out.groupby("election_id", sort=False).indices.items():
        cutoff = _routing_cutoff(str(election_id))
        if cutoff is None:
            if not ROUTING_REQUIRES_DATABLE_TARGET:
                # V31 and earlier skip here, and their frozen prospective
                # artifacts were produced with the skip in place.
                continue
            raise ValueError(
                f"no election date for {election_id}; the lineage routing used to "
                "skip such a target silently, which left every "
                "lineage_identity_* column at its initialised zero and reported "
                "nothing. A target the caller asked to route must be datable."
            )
        fit = fit_lineage_profiles(
            events,
            cutoff=pd.Timestamp(cutoff),
            half_life_years=half_life_years,
            prior_strength=prior_strength,
        )
        type_audit = fit.type_reliability.copy()
        type_audit["target_election_id"] = str(election_id)
        reliability_audits.append(type_audit)
        profiles = fit.profiles
        if profiles.empty:
            continue
        election_positions = np.asarray(election_idx, dtype=int)
        election = out.loc[election_positions]
        for region_id, local_positions in election.groupby("region_id", sort=False).indices.items():
            region_id = str(region_id)
            profile = profiles.loc[profiles["region_id"].eq(region_id)].copy()
            if profile.empty:
                continue
            idx = election_positions[np.asarray(local_positions, dtype=int)]
            group = out.loc[idx].copy()
            base = pd.to_numeric(
                group[prediction_column], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            if float(base.sum()) <= 0.0:
                continue
            base = base / base.sum()
            lineages = _candidate_party_lineages(group, candidate_parties)
            party_names = _candidate_party_names(group, candidate_parties)
            gap_by_lineage = profile.set_index("lineage_id")["weighted_gap"].to_dict()
            # The broad camp prior has already priced regional deficits.  This
            # layer owns only the positive lineage reservoir; applying negative
            # lineage gaps here would count weak-terrain evidence twice.
            major_lineages = {
                "mainstream_conservative",
                "mainstream_liberal",
                "mainstream_progressive",
            }
            direct_score = np.asarray(
                [
                    max(float(gap_by_lineage.get(lineage, 0.0)), 0.0)
                    if include_direct_lineage_score
                    and direct_lineage_scope != "none"
                    and not (
                        direct_lineage_scope == "non_major"
                        and lineage in major_lineages
                    )
                    else 0.0
                    for lineage in lineages
                ],
                dtype=float,
            )
            distinctiveness = float(profile["regional_distinctiveness"].max())
            base_affinity = _candidate_base_scores(
                group,
                election_id=str(election_id),
                region_id=region_id,
                cutoff=pd.Timestamp(cutoff),
                candidate_regional_base=candidate_regional_base,
            )
            inherited, inherited_evidence = _alignment_scores(
                group,
                election_id=str(election_id),
                region_id=region_id,
                cutoff=pd.Timestamp(cutoff),
                alignment=alignment,
            )
            inherited_reservoir = max(
                float(gap_by_lineage.get("chungcheong_regionalist", 0.0)),
                0.0,
            )
            genealogy_score = np.zeros(len(group), dtype=float)
            genealogy_affinity = np.zeros(len(group), dtype=float)
            if party_transitions is not None and not party_transitions.empty:
                for profile_row in profile.itertuples(index=False):
                    reservoir = max(float(profile_row.weighted_gap), 0.0)
                    if reservoir <= 0.0:
                        continue
                    source_parties = {
                        party.strip()
                        for party in str(profile_row.source_party_names).split("|")
                        if party.strip()
                    }
                    for position, target_party in enumerate(party_names):
                        affinity = party_genealogy_affinity(
                            party_transitions,
                            source_parties,
                            target_party,
                            cutoff=pd.Timestamp(cutoff),
                        )
                        # Same-lineage support is already represented by the
                        # direct lineage score. The graph owns only dated
                        # cross-lineage continuity such as a completed merger.
                        if lineages[position] == str(profile_row.lineage_id):
                            affinity = 0.0
                        genealogy_affinity[position] = max(
                            genealogy_affinity[position], affinity
                        )
                        genealogy_score[position] += reservoir * affinity
            raw_score = (
                direct_score
                + distinctiveness * base_affinity
                + inherited_reservoir * inherited
                + genealogy_score
            )
            centered = raw_score - float(np.dot(base, raw_score))
            log_shift = effective_gain * centered
            adjusted = base * np.exp(np.clip(log_shift, -0.50, 0.50))
            adjusted /= adjusted.sum()
            delta = adjusted - base
            maximum_delta = float(np.max(np.abs(delta), initial=0.0))
            if maximum_delta > effective_cap > 0.0:
                adjusted = base + delta * (effective_cap / maximum_delta)
                adjusted = np.clip(adjusted, 0.0, None)
                adjusted /= adjusted.sum()
                delta = adjusted - base
            if not np.isclose(adjusted.sum(), 1.0, atol=1e-12):
                raise RuntimeError("unified lineage routing broke vote-mass conservation")
            out.loc[idx, prediction_column] = adjusted
            out.loc[idx, "lineage_identity_distinctiveness"] = distinctiveness
            out.loc[idx, "lineage_identity_score"] = raw_score
            out.loc[idx, "lineage_identity_log_shift"] = log_shift
            out.loc[idx, "lineage_identity_transfer"] = delta
            out.loc[idx, "lineage_identity_gain"] = effective_gain
            out.loc[idx, "lineage_identity_candidate_lineage"] = lineages
            out.loc[idx, "lineage_identity_evidence"] = inherited_evidence
            audits.append(
                {
                    "election_id": str(election_id),
                    "region_id": region_id,
                    "regional_distinctiveness": distinctiveness,
                    "maximum_absolute_lineage_gap": float(
                        profile["weighted_gap"].abs().max()
                    ),
                    "lineages_with_prior_evidence": int(
                        profile["prior_events"].gt(0).sum()
                    ),
                    "maximum_candidate_base_affinity": float(
                        base_affinity.max(initial=0.0)
                    ),
                    "maximum_inherited_affinity": float(
                        inherited.max(initial=0.0)
                    ),
                    "maximum_genealogy_affinity": float(
                        genealogy_affinity.max(initial=0.0)
                    ),
                    "maximum_genealogy_score": float(
                        genealogy_score.max(initial=0.0)
                    ),
                    "maximum_absolute_transfer": float(
                        np.max(np.abs(delta), initial=0.0)
                    ),
                    "direct_lineage_score_enabled": bool(
                        include_direct_lineage_score
                    ),
                    "direct_lineage_scope": direct_lineage_scope,
                    "target_outcome_used": False,
                }
            )
    reliability = (
        pd.concat(reliability_audits, ignore_index=True)
        if reliability_audits
        else pd.DataFrame()
    )
    return out, pd.DataFrame(audits), reliability
