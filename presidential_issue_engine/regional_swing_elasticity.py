"""Point-in-time regional response curves from direct-party ballots."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import (
    CONSERVATIVE_BLOC,
    LIBERAL_BLOC,
    election_date,
    normalize_bloc,
)


DIRECT_PARTY_TYPES = frozenset(
    {"national_assembly_pr", "assembly_pr", "metro_council_pr", "local_council_pr"}
)
CHUNGCHEONG = frozenset({"sido_30", "sido_36", "sido_43", "sido_44"})


def _logit(value: np.ndarray | pd.Series | float) -> np.ndarray:
    array = np.clip(np.asarray(value, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(array / (1.0 - array))


def _expit(value: np.ndarray | pd.Series | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-array))


def build_event_frame(
    history: pd.DataFrame,
    *,
    date_resolver: Callable[[object], pd.Timestamp | None] = election_date,
) -> pd.DataFrame:
    """Aggregate party rows into regional two-camp shares by election event."""

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
        raise ValueError(f"regional swing history missing columns: {sorted(missing)}")
    work = history.loc[history["election_type"].isin(DIRECT_PARTY_TYPES)].copy()
    work["event_date"] = work["election_id"].map(date_resolver)
    work = work.loc[work["event_date"].notna()].copy()
    work["bloc"] = work["bloc"].map(normalize_bloc)
    work = work.loc[work["bloc"].isin({CONSERVATIVE_BLOC, LIBERAL_BLOC})]
    grouped = (
        work.groupby(
            ["election_id", "election_type", "event_date", "region_id", "bloc"],
            as_index=False,
        )
        .agg(
            vote_share=("vote_share", "sum"),
            quality=("data_quality_weight", "mean"),
        )
    )
    shares = grouped.pivot_table(
        index=["election_id", "election_type", "event_date", "region_id"],
        columns="bloc",
        values="vote_share",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    quality = grouped.groupby(
        ["election_id", "election_type", "event_date", "region_id"], as_index=False
    )["quality"].mean()
    shares = shares.merge(
        quality,
        on=["election_id", "election_type", "event_date", "region_id"],
        how="left",
        validate="one_to_one",
    )
    for bloc in (CONSERVATIVE_BLOC, LIBERAL_BLOC):
        if bloc not in shares:
            shares[bloc] = 0.0
    denominator = shares[CONSERVATIVE_BLOC] + shares[LIBERAL_BLOC]
    shares = shares.loc[denominator.gt(0.0)].copy()
    shares["regional_conservative_share"] = (
        shares[CONSERVATIVE_BLOC]
        / (shares[CONSERVATIVE_BLOC] + shares[LIBERAL_BLOC])
    )
    national = shares.groupby("election_id", sort=False).apply(
        lambda group: float(
            np.average(group["regional_conservative_share"], weights=group["quality"])
        ),
        include_groups=False,
    )
    shares["national_conservative_share"] = shares["election_id"].map(national)
    shares["national_logit"] = _logit(shares["national_conservative_share"])
    shares["regional_logit"] = _logit(shares["regional_conservative_share"])
    return shares.sort_values(["event_date", "election_id", "region_id"]).reset_index(drop=True)


def _fit_curve(group: pd.DataFrame, prior_strength: float) -> dict[str, float]:
    weights = pd.to_numeric(group["quality"], errors="coerce").fillna(0.0).to_numpy(float)
    x = group["national_logit"].to_numpy(float)
    y = group["regional_logit"].to_numpy(float)
    design = np.column_stack([np.ones(len(group), dtype=float), x])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_y = y * np.sqrt(weights)
    intercept, slope = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
    effective_n = float(weights.sum())
    reliability = effective_n / (effective_n + max(float(prior_strength), 1e-6))
    return {
        "intercept": float(intercept * reliability),
        "slope": float(1.0 + (np.clip(slope, 0.25, 1.75) - 1.0) * reliability),
        "offset": float(np.average(y - x, weights=weights) * reliability),
        "effective_n": effective_n,
        "reliability": reliability,
        "events": int(group["election_id"].nunique()),
    }


def fit_profiles(
    events: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    prior_strength: float = 4.0,
) -> pd.DataFrame:
    """Fit region curves from events strictly before ``cutoff``."""

    prior = events.loc[events["event_date"].lt(pd.Timestamp(cutoff))].copy()
    rows: list[dict[str, object]] = []
    for region_id, group in prior.groupby("region_id", sort=True):
        if group["election_id"].nunique() < 2:
            continue
        rows.append({"region_id": region_id, "source": "region", **_fit_curve(group, prior_strength)})
    if not prior.empty:
        hierarchy_groups = {"chungcheong": CHUNGCHEONG}
        for name, region_ids in hierarchy_groups.items():
            group = prior.loc[prior["region_id"].isin(region_ids)]
            if group["election_id"].nunique() >= 2:
                rows.append(
                    {
                        "region_id": f"hierarchy:{name}",
                        "source": "hierarchy",
                        **_fit_curve(group, prior_strength),
                    }
                )
    return pd.DataFrame(rows)


def profile_for_region(profiles: pd.DataFrame, region_id: str) -> pd.Series | None:
    if profiles.empty or "region_id" not in profiles.columns:
        return None
    direct = profiles.loc[profiles["region_id"].eq(region_id)]
    if not direct.empty:
        return direct.iloc[0]
    if region_id in CHUNGCHEONG:
        hierarchy = profiles.loc[profiles["region_id"].eq("hierarchy:chungcheong")]
        if not hierarchy.empty:
            return hierarchy.iloc[0]
    return None


def predict_region_share(
    profile: pd.Series | None,
    national_share: float,
    *,
    method: str,
) -> float:
    if profile is None:
        return float(national_share)
    national_logit = float(_logit(national_share))
    if method == "offset":
        value = national_logit + float(profile["offset"])
    elif method == "elasticity":
        value = float(profile["intercept"]) + float(profile["slope"]) * national_logit
    else:
        raise ValueError(f"unknown regional swing method: {method}")
    return float(_expit(value))


def apply_regional_offset(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    prediction_column: str,
    gain_by_election: dict[str, float],
    prior_strength: float = 2.0,
) -> pd.DataFrame:
    """Apply a bounded offset fallback while preserving third-candidate mass."""

    required = {
        "election_id",
        "region_id",
        "bloc",
        "major_party_core_eligible",
        prediction_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"regional offset frame missing columns: {sorted(missing)}")
    out = frame.copy().reset_index(drop=True)
    out["regional_offset_gain"] = 0.0
    out["regional_offset_target"] = out[prediction_column].astype(float)
    out["regional_offset_shift"] = 0.0
    out["regional_offset_profile_source"] = "none"
    out["regional_offset_profile_reliability"] = 0.0

    for election_id, election_indices in out.groupby("election_id", sort=False).indices.items():
        base_gain = float(np.clip(gain_by_election.get(str(election_id), 0.0), 0.0, 0.35))
        cutoff = election_date(election_id)
        if base_gain <= 0.0 or cutoff is None:
            continue
        profiles = fit_profiles(events, cutoff=cutoff, prior_strength=prior_strength)
        election_idx = np.asarray(election_indices, dtype=int)
        election = out.loc[election_idx]
        normalized_bloc = election["bloc"].map(normalize_bloc)
        eligible = election["major_party_core_eligible"].fillna(False).astype(bool)
        regional_national_shares: list[float] = []
        for _, group in election.groupby("region_id", sort=False):
            group_bloc = group["bloc"].map(normalize_bloc)
            group_eligible = group["major_party_core_eligible"].fillna(False).astype(bool)
            conservative = float(
                group.loc[group_bloc.eq(CONSERVATIVE_BLOC) & group_eligible, prediction_column].sum()
            )
            liberal = float(
                group.loc[group_bloc.eq(LIBERAL_BLOC) & group_eligible, prediction_column].sum()
            )
            if conservative + liberal > 0.0:
                regional_national_shares.append(conservative / (conservative + liberal))
        if not regional_national_shares:
            continue
        national_share = float(np.mean(regional_national_shares))

        for region_id, region_indices in election.groupby("region_id", sort=False).indices.items():
            profile = profile_for_region(profiles, str(region_id))
            if profile is None:
                continue
            local_positions = np.asarray(region_indices, dtype=int)
            idx = election_idx[local_positions]
            group = out.loc[idx]
            group_bloc = group["bloc"].map(normalize_bloc)
            group_eligible = group["major_party_core_eligible"].fillna(False).astype(bool)
            conservative_mask = group_bloc.eq(CONSERVATIVE_BLOC) & group_eligible
            liberal_mask = group_bloc.eq(LIBERAL_BLOC) & group_eligible
            if not conservative_mask.any() or not liberal_mask.any():
                continue
            conservative_idx = idx[conservative_mask.to_numpy()]
            liberal_idx = idx[liberal_mask.to_numpy()]
            conservative = float(out.loc[conservative_idx, prediction_column].sum())
            liberal = float(out.loc[liberal_idx, prediction_column].sum())
            pool = conservative + liberal
            if pool <= 0.0:
                continue
            reliability = float(np.clip(profile["reliability"], 0.0, 1.0))
            gain = base_gain * reliability
            target_share = predict_region_share(
                profile, national_share, method="offset"
            )
            conservative_target = pool * target_share
            liberal_target = pool * (1.0 - target_share)
            before = out.loc[idx, prediction_column].to_numpy(float)
            if conservative > 0.0:
                out.loc[conservative_idx, prediction_column] *= (
                    ((1.0 - gain) * conservative + gain * conservative_target)
                    / conservative
                )
            if liberal > 0.0:
                out.loc[liberal_idx, prediction_column] *= (
                    ((1.0 - gain) * liberal + gain * liberal_target) / liberal
                )
            after = out.loc[idx, prediction_column].to_numpy(float)
            if not np.isclose(before.sum(), after.sum(), atol=1e-12):
                raise RuntimeError("regional offset failed vote-mass conservation")
            out.loc[idx, "regional_offset_gain"] = gain
            out.loc[conservative_idx, "regional_offset_target"] = conservative_target
            out.loc[liberal_idx, "regional_offset_target"] = liberal_target
            out.loc[idx, "regional_offset_shift"] = after - before
            out.loc[idx, "regional_offset_profile_source"] = str(profile["source"])
            out.loc[idx, "regional_offset_profile_reliability"] = reliability
    return out
