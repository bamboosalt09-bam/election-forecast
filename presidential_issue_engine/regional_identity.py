"""Point-in-time regional distinctiveness reinforcement outside Chungcheong."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import election_date, normalize_bloc
from presidential_issue_engine.chungcheong_identity import CHUNGCHEONG


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


def build_distinctiveness_events(
    history: pd.DataFrame,
    *,
    date_resolver: Callable[[object], pd.Timestamp | None] = election_date,
) -> pd.DataFrame:
    """Measure how far each regional party distribution is from the event median."""

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
        raise ValueError(f"regional identity history missing columns: {sorted(missing)}")

    work = history.loc[history["election_type"].isin(IDENTITY_EVENT_TYPES)].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(date_resolver), errors="coerce"
    )
    work = work.loc[work["event_date"].notna()].copy()
    work["normalized_bloc"] = work["bloc"].astype(str).str.strip().map(normalize_bloc)
    work["vote_share"] = pd.to_numeric(work["vote_share"], errors="coerce").fillna(0.0)
    work["quality"] = pd.to_numeric(
        work["data_quality_weight"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)

    grouped = (
        work.groupby(
            [
                "election_id",
                "election_type",
                "event_date",
                "region_id",
                "normalized_bloc",
            ],
            as_index=False,
        )
        .agg(vote_share=("vote_share", "sum"), quality=("quality", "mean"))
    )
    totals = grouped.groupby(["election_id", "region_id"])["vote_share"].transform("sum")
    grouped["regional_share"] = np.divide(
        grouped["vote_share"],
        totals,
        out=np.zeros(len(grouped), dtype=float),
        where=totals.to_numpy(float) > 0.0,
    )
    baseline = (
        grouped.groupby(["election_id", "normalized_bloc"])["regional_share"]
        .median()
        .rename("median_share")
        .reset_index()
    )
    baseline_total = baseline.groupby("election_id")["median_share"].transform("sum")
    baseline["national_reference"] = np.divide(
        baseline["median_share"],
        baseline_total,
        out=np.zeros(len(baseline), dtype=float),
        where=baseline_total.to_numpy(float) > 0.0,
    )
    grouped = grouped.merge(
        baseline[["election_id", "normalized_bloc", "national_reference"]],
        on=["election_id", "normalized_bloc"],
        how="left",
    )
    grouped["absolute_gap"] = (
        grouped["regional_share"] - grouped["national_reference"].fillna(0.0)
    ).abs()
    events = (
        grouped.groupby(
            ["election_id", "election_type", "event_date", "region_id"],
            as_index=False,
        )
        .agg(
            distinctiveness=("absolute_gap", lambda values: 0.5 * float(values.sum())),
            quality=("quality", "mean"),
        )
    )
    events["distinctiveness"] = events["distinctiveness"].clip(0.0, 0.50)
    events["type_weight"] = events["election_type"].map(EVENT_TYPE_WEIGHTS).fillna(0.0)
    return events.sort_values(["event_date", "election_id", "region_id"]).reset_index(
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
    values = group["distinctiveness"].to_numpy(float)
    total = float(weight.sum())
    if total <= 0.0:
        return {
            "distinctiveness": 0.0,
            "reliability": 0.0,
            "effective_n": 0.0,
            "events": 0,
        }
    mean = float(np.average(values, weights=weight))
    mad = float(np.average(np.abs(values - mean), weights=weight))
    effective_n = total**2 / max(float(np.square(weight).sum()), 1e-12)
    reliability = effective_n / (effective_n + max(prior_strength, 1e-6))
    reliability *= float(np.clip(1.0 - mad / max(mean + 0.05, 0.05), 0.25, 1.0))
    return {
        "distinctiveness": float(np.clip(mean, 0.0, 0.50)),
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
        "effective_n": effective_n,
        "events": int(group["election_id"].nunique()),
    }


def fit_distinctiveness_profiles(
    events: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    half_life_years: float = 12.0,
    prior_strength: float = 1.5,
) -> pd.DataFrame:
    """Fit region-specific profiles using events strictly before the target date."""

    prior = events.loc[
        events["event_date"].lt(pd.Timestamp(cutoff))
        & ~events["region_id"].astype(str).isin(CHUNGCHEONG)
    ].copy()
    rows: list[dict[str, object]] = []
    for region_id, group in prior.groupby("region_id", sort=True):
        rows.append(
            {
                "region_id": str(region_id),
                **_profile(
                    group,
                    cutoff=pd.Timestamp(cutoff),
                    half_life_years=half_life_years,
                    prior_strength=prior_strength,
                ),
            }
        )
    return pd.DataFrame(rows)


def _candidate_names(frame: pd.DataFrame) -> pd.Series:
    for column in ("candidate_name", "candidate_name_x", "candidate_name_y"):
        if column in frame.columns:
            return frame[column].fillna("").astype(str)
    raise ValueError("regional identity frame has no candidate-name column")


def _candidate_affinity(
    group: pd.DataFrame,
    *,
    election_id: str,
    region_id: str,
    cutoff: pd.Timestamp,
    candidate_regional_base: pd.DataFrame,
) -> np.ndarray:
    if candidate_regional_base.empty:
        return np.zeros(len(group), dtype=float)
    evidence = candidate_regional_base.loc[
        candidate_regional_base["election_id"].astype(str).eq(election_id)
        & candidate_regional_base["region_id"].astype(str).eq(region_id)
    ].copy()
    evidence["available_date"] = pd.to_datetime(
        evidence.get("available_date"), errors="coerce"
    )
    evidence = evidence.loc[
        evidence["available_date"].notna()
        & evidence["available_date"].lt(pd.Timestamp(cutoff))
    ].copy()
    if evidence.empty:
        return np.zeros(len(group), dtype=float)

    evidence["weighted_affinity"] = (
        pd.to_numeric(evidence.get("regional_affinity"), errors="coerce").fillna(0.0)
        * pd.to_numeric(evidence.get("organization_depth"), errors="coerce").fillna(0.0)
        * pd.to_numeric(evidence.get("confidence"), errors="coerce").fillna(0.0)
    ).clip(0.0, 1.0)
    names = _candidate_names(group)
    score = np.zeros(len(group), dtype=float)
    for row in evidence.itertuples(index=False):
        score = np.maximum(
            score,
            np.where(names.eq(str(row.candidate_name)), float(row.weighted_affinity), 0.0),
        )
    return score


def _donor_weights(
    group: pd.DataFrame,
    base: np.ndarray,
    recipient: np.ndarray,
) -> np.ndarray:
    """Prefer donors least compatible with the prior regional camp profile."""

    camp_columns = {
        "camp_conservative": "regional_accent_conservative_share",
        "camp_liberal": "regional_accent_liberal_share",
        "camp_progressive": "regional_accent_progressive_share",
        "camp_centrist": "regional_accent_centrist_share",
        "camp_regionalist": "regional_accent_regionalist_share",
        "camp_reform": "regional_accent_reform_share",
    }
    if "candidate_camp" not in group.columns:
        return base[~recipient]
    compatibility = np.zeros(len(group), dtype=float)
    known = np.zeros(len(group), dtype=bool)
    camps = group["candidate_camp"].fillna("").astype(str).to_numpy()
    for position, camp in enumerate(camps):
        column = camp_columns.get(camp)
        if column is None or column not in group.columns:
            continue
        value = pd.to_numeric(group.iloc[position][column], errors="coerce")
        if pd.notna(value):
            compatibility[position] = float(np.clip(value, 0.0, 1.0))
            known[position] = True
    donor_known = known & ~recipient
    if not donor_known.any():
        return base[~recipient]
    reference = float(compatibility[~recipient].max(initial=0.0))
    if reference <= 0.0:
        return base[~recipient]
    mismatch = np.clip(1.0 - compatibility[~recipient] / reference, 0.0, 1.0) ** 2
    mismatch[~known[~recipient]] = 1.0
    weighted = base[~recipient] * mismatch
    return weighted if float(weighted.sum()) > 1e-12 else base[~recipient]


def apply_regional_identity_routing(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    candidate_regional_base: pd.DataFrame,
    *,
    prediction_column: str,
    gain: float,
    shift_cap: float = 0.04,
    half_life_years: float = 12.0,
    prior_strength: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reinforce dated candidate bases in distinctive non-Chungcheong regions."""

    out = frame.copy().reset_index(drop=True)
    out["regional_identity_distinctiveness"] = 0.0
    out["regional_identity_reliability"] = 0.0
    out["regional_identity_affinity"] = 0.0
    out["regional_identity_transfer"] = 0.0
    out["regional_identity_gain"] = 0.0
    audit: list[dict[str, object]] = []
    effective_gain = float(np.clip(gain, 0.0, 1.0))
    effective_cap = float(np.clip(shift_cap, 0.0, 0.08))
    if effective_gain <= 0.0:
        return out, pd.DataFrame(audit)

    for election_id, election_idx in out.groupby("election_id", sort=False).indices.items():
        cutoff = election_date(str(election_id))
        if cutoff is None:
            continue
        profiles = fit_distinctiveness_profiles(
            events,
            cutoff=pd.Timestamp(cutoff),
            half_life_years=half_life_years,
            prior_strength=prior_strength,
        )
        if profiles.empty:
            continue
        profiles = profiles.set_index("region_id")
        election_positions = np.asarray(election_idx, dtype=int)
        election = out.loc[election_positions]
        for region_id, local_positions in election.groupby("region_id", sort=False).indices.items():
            region_id = str(region_id)
            if region_id in CHUNGCHEONG or region_id not in profiles.index:
                continue
            idx = election_positions[np.asarray(local_positions, dtype=int)]
            group = out.loc[idx].copy()
            affinity = _candidate_affinity(
                group,
                election_id=str(election_id),
                region_id=region_id,
                cutoff=pd.Timestamp(cutoff),
                candidate_regional_base=candidate_regional_base,
            )
            recipient = affinity > 0.0
            profile = profiles.loc[region_id]
            distinctiveness = float(profile["distinctiveness"])
            reliability = float(profile["reliability"])
            out.loc[idx, "regional_identity_distinctiveness"] = distinctiveness
            out.loc[idx, "regional_identity_reliability"] = reliability
            out.loc[idx, "regional_identity_affinity"] = affinity
            transfer = 0.0
            if recipient.any() and not recipient.all():
                base = pd.to_numeric(
                    group[prediction_column], errors="coerce"
                ).fillna(0.0).to_numpy(float)
                donor_mass = float(base[~recipient].sum())
                donor_weights = _donor_weights(group, base, recipient)
                donor_weights /= donor_weights.sum()
                recipient_weights = np.square(affinity[recipient])
                recipient_weights /= recipient_weights.sum()
                transfer = min(
                    effective_cap,
                    effective_gain
                    * distinctiveness
                    * reliability
                    * float(affinity.max()),
                    donor_mass,
                )
                adjusted = base.copy()
                adjusted[recipient] += transfer * recipient_weights
                adjusted[~recipient] -= transfer * donor_weights
                if (adjusted < -1e-12).any() or not np.isclose(
                    adjusted.sum(), base.sum(), atol=1e-12
                ):
                    raise RuntimeError("regional identity routing broke vote-mass conservation")
                out.loc[idx, prediction_column] = np.clip(adjusted, 0.0, 1.0)
                out.loc[idx, "regional_identity_transfer"] = adjusted - base
                out.loc[idx, "regional_identity_gain"] = effective_gain
            audit.append(
                {
                    "election_id": str(election_id),
                    "region_id": region_id,
                    "distinctiveness": distinctiveness,
                    "reliability": reliability,
                    "effective_n": float(profile["effective_n"]),
                    "prior_events": int(profile["events"]),
                    "maximum_candidate_affinity": float(affinity.max(initial=0.0)),
                    "absolute_transfer": float(transfer),
                }
            )
    return out, pd.DataFrame(audit)
