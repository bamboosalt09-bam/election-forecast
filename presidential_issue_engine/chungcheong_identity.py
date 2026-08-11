"""Point-in-time Chungcheong regional-identity reservoir and routing layer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import election_date, normalize_bloc
from presidential_issue_engine.electorate_layers import (
    REGIONALIST_PARTY_LABELS,
    THIRD_BLOC,
)


CHUNGCHEONG = frozenset({"sido_30", "sido_36", "sido_43", "sido_44"})
DIRECT_PARTY_TYPES = frozenset(
    {"national_assembly_pr", "assembly_pr", "metro_council_pr", "local_council_pr"}
)
IDENTITY_EVENT_TYPES = frozenset({*DIRECT_PARTY_TYPES, "presidential"})
EVENT_TYPE_WEIGHTS = {
    "national_assembly_pr": 1.0,
    "assembly_pr": 1.0,
    "metro_council_pr": 1.0,
    "local_council_pr": 1.0,
    "presidential": 0.35,
}


def build_identity_events(
    history: pd.DataFrame,
    *,
    date_resolver: Callable[[object], pd.Timestamp | None] = election_date,
) -> pd.DataFrame:
    """Measure local third/regional-party excess without using target outcomes."""

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
        raise ValueError(f"identity history missing columns: {sorted(missing)}")
    work = history.loc[history["election_type"].isin(IDENTITY_EVENT_TYPES)].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(date_resolver), errors="coerce"
    )
    work = work.loc[work["event_date"].notna()].copy()
    work["vote_share"] = pd.to_numeric(work["vote_share"], errors="coerce").fillna(0.0)
    work["quality"] = pd.to_numeric(
        work["data_quality_weight"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    raw_bloc = work["bloc"].astype(str).str.strip()
    normalized = raw_bloc.map(normalize_bloc)
    work["identity_vote_share"] = work["vote_share"].where(
        normalized.eq(THIRD_BLOC) | raw_bloc.isin(REGIONALIST_PARTY_LABELS), 0.0
    )
    grouped = (
        work.groupby(
            ["election_id", "election_type", "event_date", "region_id"],
            as_index=False,
        )
        .agg(
            identity_share=("identity_vote_share", "sum"),
            quality=("quality", "mean"),
        )
    )
    # A nationally popular third party is not Chungcheong regional identity.
    # The cross-region median removes that common component before aggregation.
    baseline = grouped.groupby("election_id")["identity_share"].median()
    grouped["national_identity_baseline"] = grouped["election_id"].map(baseline)
    grouped["identity_excess"] = (
        grouped["identity_share"] - grouped["national_identity_baseline"]
    ).clip(lower=0.0, upper=0.40)
    grouped["type_weight"] = grouped["election_type"].map(EVENT_TYPE_WEIGHTS).fillna(0.0)
    return grouped.sort_values(["event_date", "election_id", "region_id"]).reset_index(
        drop=True
    )


def _profile(
    group: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    half_life_years: float,
    prior_strength: float,
) -> dict[str, float]:
    age = (pd.Timestamp(cutoff) - group["event_date"]).dt.days.clip(lower=0) / 365.25
    weight = (
        group["quality"].to_numpy(float)
        * group["type_weight"].to_numpy(float)
        * np.exp(-np.log(2.0) * age.to_numpy(float) / max(half_life_years, 0.1))
    )
    values = group["identity_excess"].to_numpy(float)
    total = float(weight.sum())
    if total <= 0.0:
        return {"reservoir": 0.0, "reliability": 0.0, "effective_n": 0.0, "events": 0}
    reservoir = float(np.average(values, weights=weight))
    mad = float(np.average(np.abs(values - reservoir), weights=weight))
    effective_n = total**2 / max(float(np.square(weight).sum()), 1e-12)
    reliability = effective_n / (effective_n + max(prior_strength, 1e-6))
    reliability *= float(np.clip(1.0 - mad / max(reservoir + 0.05, 0.05), 0.25, 1.0))
    return {
        "reservoir": float(np.clip(reservoir, 0.0, 0.35)),
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
        "effective_n": effective_n,
        "events": int(group["election_id"].nunique()),
    }


def fit_identity_profiles(
    events: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    half_life_years: float = 12.0,
    prior_strength: float = 1.5,
) -> pd.DataFrame:
    """Fit Chungcheong profiles from events strictly before the target date."""

    prior = events.loc[
        events["event_date"].lt(pd.Timestamp(cutoff))
        & events["region_id"].astype(str).isin(CHUNGCHEONG)
    ].copy()
    rows: list[dict[str, object]] = []
    for region_id, group in prior.groupby("region_id", sort=True):
        rows.append(
            {
                "region_id": str(region_id),
                "profile_source": "region",
                **_profile(
                    group,
                    cutoff=pd.Timestamp(cutoff),
                    half_life_years=half_life_years,
                    prior_strength=prior_strength,
                ),
            }
        )
    if not prior.empty:
        rows.append(
            {
                "region_id": "hierarchy:chungcheong",
                "profile_source": "hierarchy",
                **_profile(
                    prior,
                    cutoff=pd.Timestamp(cutoff),
                    half_life_years=half_life_years,
                    prior_strength=prior_strength,
                ),
            }
        )
    return pd.DataFrame(rows)


def _profile_for_region(profiles: pd.DataFrame, region_id: str) -> pd.Series | None:
    direct = profiles.loc[profiles["region_id"].eq(region_id)] if not profiles.empty else profiles
    if not direct.empty:
        return direct.iloc[0]
    fallback = profiles.loc[profiles["region_id"].eq("hierarchy:chungcheong")]
    return None if fallback.empty else fallback.iloc[0]


def _candidate_names(frame: pd.DataFrame) -> pd.Series:
    for column in ("candidate_name", "candidate_name_x", "candidate_name_y"):
        if column in frame.columns:
            return frame[column].fillna("").astype(str)
    raise ValueError("identity frame has no candidate-name column")


def _eligible_evidence(
    source: pd.DataFrame,
    election_id: str,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    if source.empty or "election_id" not in source.columns:
        return source.iloc[0:0].copy()
    work = source.loc[source["election_id"].astype(str).eq(election_id)].copy()
    work["available_date"] = pd.to_datetime(work.get("available_date"), errors="coerce")
    return work.loc[
        work["available_date"].notna() & work["available_date"].lt(pd.Timestamp(cutoff))
    ].copy()


def _candidate_affinity(
    group: pd.DataFrame,
    *,
    election_id: str,
    region_id: str,
    cutoff: pd.Timestamp,
    candidate_regional_base: pd.DataFrame,
    alignment: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    names = _candidate_names(group)
    score = np.zeros(len(group), dtype=float)
    sources = ["none"] * len(group)

    base = _eligible_evidence(candidate_regional_base, election_id, cutoff)
    if not base.empty:
        base = base.loc[base["region_id"].astype(str).eq(region_id)].copy()
        base["weighted_affinity"] = (
            pd.to_numeric(base.get("regional_affinity"), errors="coerce").fillna(0.0)
            * pd.to_numeric(base.get("organization_depth"), errors="coerce").fillna(0.0)
            * pd.to_numeric(base.get("confidence"), errors="coerce").fillna(0.0)
        ).clip(0.0, 1.0)
        for row in base.itertuples(index=False):
            match = names.eq(str(row.candidate_name)).to_numpy()
            value = float(row.weighted_affinity)
            for pos in np.flatnonzero(match):
                if value > score[pos]:
                    score[pos] = value
                    sources[pos] = "candidate_regional_base"

    routed = _eligible_evidence(alignment, election_id, cutoff)
    if not routed.empty:
        scope = routed.get("region_scope", pd.Series("", index=routed.index)).astype(str)
        routed = routed.loc[scope.isin({region_id, "chungcheong"})].copy()
        routed["weighted_affinity"] = (
            pd.to_numeric(routed.get("affinity"), errors="coerce").fillna(0.0)
            * pd.to_numeric(routed.get("confidence"), errors="coerce").fillna(0.0)
        ).clip(0.0, 1.0)
        for row in routed.itertuples(index=False):
            match = names.eq(str(row.candidate_name)).to_numpy()
            value = float(row.weighted_affinity)
            for pos in np.flatnonzero(match):
                if value > score[pos]:
                    score[pos] = value
                    sources[pos] = str(row.evidence_type)
    return score, sources


def apply_identity_routing(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    candidate_regional_base: pd.DataFrame,
    alignment: pd.DataFrame,
    *,
    prediction_column: str,
    gain: float,
    shift_cap: float = 0.08,
    half_life_years: float = 12.0,
    prior_strength: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route only evidenced regional-identity mass while conserving each region."""

    out = frame.copy().reset_index(drop=True)
    out["chung_identity_reservoir"] = 0.0
    out["chung_identity_reliability"] = 0.0
    out["chung_identity_affinity"] = 0.0
    out["chung_identity_evidence"] = "none"
    out["chung_identity_transfer"] = 0.0
    out["chung_identity_gain"] = 0.0
    audit: list[dict[str, object]] = []
    effective_gain = float(np.clip(gain, 0.0, 1.0))
    effective_cap = float(np.clip(shift_cap, 0.0, 0.15))
    if effective_gain <= 0.0:
        return out, pd.DataFrame(audit)

    for election_id, election_idx in out.groupby("election_id", sort=False).indices.items():
        cutoff = election_date(str(election_id))
        if cutoff is None:
            continue
        profiles = fit_identity_profiles(
            events,
            cutoff=pd.Timestamp(cutoff),
            half_life_years=half_life_years,
            prior_strength=prior_strength,
        )
        election_positions = np.asarray(election_idx, dtype=int)
        election = out.loc[election_positions]
        for region_id, local_positions in election.groupby("region_id", sort=False).indices.items():
            if str(region_id) not in CHUNGCHEONG:
                continue
            idx = election_positions[np.asarray(local_positions, dtype=int)]
            group = out.loc[idx].copy()
            profile = _profile_for_region(profiles, str(region_id))
            if profile is None:
                continue
            affinity, evidence = _candidate_affinity(
                group,
                election_id=str(election_id),
                region_id=str(region_id),
                cutoff=pd.Timestamp(cutoff),
                candidate_regional_base=candidate_regional_base,
                alignment=alignment,
            )
            recipient = affinity > 0.0
            base = pd.to_numeric(group[prediction_column], errors="coerce").fillna(0.0).to_numpy(float)
            reservoir = float(profile["reservoir"])
            reliability = float(profile["reliability"])
            out.loc[idx, "chung_identity_reservoir"] = reservoir
            out.loc[idx, "chung_identity_reliability"] = reliability
            out.loc[idx, "chung_identity_affinity"] = affinity
            out.loc[idx, "chung_identity_evidence"] = evidence
            if not recipient.any() or recipient.all():
                transfer = 0.0
            else:
                recipient_weights = np.square(affinity[recipient])
                recipient_weights /= recipient_weights.sum()
                donor_mass = float(base[~recipient].sum())
                transfer = min(
                    effective_cap,
                    effective_gain * reservoir * reliability * float(affinity.max()),
                    donor_mass,
                )
                adjusted = base.copy()
                adjusted[recipient] += transfer * recipient_weights
                adjusted[~recipient] -= transfer * base[~recipient] / donor_mass
                if (adjusted < -1e-12).any() or not np.isclose(
                    adjusted.sum(), base.sum(), atol=1e-12
                ):
                    raise RuntimeError("Chungcheong identity routing broke vote-mass conservation")
                out.loc[idx, prediction_column] = np.clip(adjusted, 0.0, 1.0)
                out.loc[idx, "chung_identity_transfer"] = adjusted - base
                out.loc[idx, "chung_identity_gain"] = effective_gain
            audit.append(
                {
                    "election_id": str(election_id),
                    "region_id": str(region_id),
                    "profile_source": str(profile["profile_source"]),
                    "reservoir": reservoir,
                    "reliability": reliability,
                    "effective_n": float(profile["effective_n"]),
                    "prior_events": int(profile["events"]),
                    "maximum_candidate_affinity": float(affinity.max(initial=0.0)),
                    "absolute_transfer": float(transfer),
                    "evidence": "|".join(sorted(set(evidence) - {"none"})) or "none",
                }
            )
    return out, pd.DataFrame(audit)
