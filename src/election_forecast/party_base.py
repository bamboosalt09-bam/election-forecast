"""Regional party-base estimation from historical election results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from election_forecast.config import ForecastConfig
from election_forecast.features.region_bloc_prior import (
    CONSERVATIVE_BLOC,
    INDEPENDENT_BLOC,
    LIBERAL_BLOC,
    PROGRESSIVE_BLOC,
    THIRD_BLOC,
    compute_bloc_base,
)


BLOC_TO_CAMP = {
    CONSERVATIVE_BLOC: "conservative",
    LIBERAL_BLOC: "liberal",
    PROGRESSIVE_BLOC: "progressive",
    THIRD_BLOC: "centrist",
    INDEPENDENT_BLOC: "local_independent",
}


def _as_rate(series: pd.Series) -> pd.Series:
    """Convert percentage-like scores to 0-1 rates when needed."""

    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return values.where(values.abs().le(1.0), values / 100.0)


def compute_party_base(
    election_results: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
    config: ForecastConfig,
    bloc_history: pd.DataFrame | None = None,
    target_election_id: str | None = None,
) -> pd.DataFrame:
    """Compute recency-weighted camp vote base by region.

    When ``bloc_history`` and ``target_election_id`` are provided, use the
    repeated-election regional bloc prior as the primary source. This captures
    party-list, local council, and other party-terrain evidence. Otherwise,
    fall back to the older ``election_results`` camp average.
    """

    columns = ["region_id", *config.camp_columns]
    if bloc_history is not None and target_election_id:
        bloc_base = compute_bloc_base(
            bloc_history,
            target_election_id,
            election_type_weights=config.election_type_weights,
        )
        party_base = _bloc_base_to_camp_base(bloc_base, config)
        if not party_base.empty:
            return party_base[columns]

    if election_results.empty:
        return pd.DataFrame(columns=columns)

    frame = election_results.copy()
    cutoff = pd.Timestamp(forecast_date)
    frame["election_date"] = pd.to_datetime(frame["election_date"], errors="coerce")
    frame = frame.loc[frame["election_date"].le(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["vote_share"] = _as_rate(frame["vote_share"])
    age_days = (cutoff - frame["election_date"]).dt.days.clip(lower=0)
    half_life_days = config.recency_half_life_years * 365.25
    frame["recency_weight"] = np.exp(-age_days / half_life_days)
    frame["type_weight"] = frame["election_type"].map(config.election_type_weights).fillna(0.35)
    frame["weight"] = frame["recency_weight"] * frame["type_weight"]
    frame["weighted_share"] = frame["vote_share"] * frame["weight"]

    grouped = (
        frame.groupby(["region_id", "camp"], as_index=False)
        .agg(weighted_share=("weighted_share", "sum"), weight=("weight", "sum"))
    )
    grouped["party_base"] = grouped["weighted_share"] / grouped["weight"].replace(0, np.nan)
    grouped["party_base"] = grouped["party_base"].fillna(0.0)

    pivot = grouped.pivot_table(
        index="region_id", columns="camp", values="party_base", aggfunc="mean", fill_value=0.0
    )
    for camp in config.camp_columns:
        if camp not in pivot.columns:
            pivot[camp] = 0.0
    pivot = pivot[config.camp_columns].reset_index()
    pivot[config.camp_columns] = pivot[config.camp_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    camp_sum = pivot[config.camp_columns].sum(axis=1)
    nonzero = camp_sum.gt(0)
    normalized = pivot.loc[nonzero, config.camp_columns].div(camp_sum.loc[nonzero], axis=0).astype(float)
    pivot.loc[nonzero, config.camp_columns] = normalized
    return pivot[columns]


def _bloc_base_to_camp_base(bloc_base: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Map normalized analytic blocs to the forecast camp-vector columns."""

    columns = ["region_id", *config.camp_columns]
    if bloc_base.empty:
        return pd.DataFrame(columns=columns)
    frame = bloc_base.copy()
    frame["camp"] = frame["bloc"].map(BLOC_TO_CAMP).fillna("anti_party")
    frame = frame.loc[
        frame["bloc"].isin(BLOC_TO_CAMP) & frame["camp"].isin(config.camp_columns)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    pivot = frame.pivot_table(
        index="region_id",
        columns="camp",
        values="bloc_base",
        aggfunc="sum",
        fill_value=0.0,
    )
    for camp in config.camp_columns:
        if camp not in pivot.columns:
            pivot[camp] = 0.0
    pivot = pivot[config.camp_columns].reset_index()
    pivot[config.camp_columns] = pivot[config.camp_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    camp_sum = pivot[config.camp_columns].sum(axis=1)
    nonzero = camp_sum.gt(0)
    normalized = pivot.loc[nonzero, config.camp_columns].div(camp_sum.loc[nonzero], axis=0).astype(float)
    pivot.loc[nonzero, config.camp_columns] = normalized
    return pivot[columns]


def compute_party_base_effect(
    party_base: pd.DataFrame,
    candidate_party_vectors: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Compute ``PartyBaseEffect(c, r) = RegionPartyBase(r) · z_c``."""

    records: list[dict[str, object]] = []
    if party_base.empty or candidate_party_vectors.empty:
        return pd.DataFrame(columns=["candidate_id", "region_id", "party_base_effect"])

    vector_frame = candidate_party_vectors.copy()
    for col in config.camp_columns:
        if col not in vector_frame.columns:
            vector_frame[col] = 0.0

    for _, region in party_base.iterrows():
        region_vec = region[config.camp_columns].astype(float).to_numpy()
        for _, candidate in vector_frame.iterrows():
            candidate_vec = candidate[config.camp_columns].astype(float).to_numpy()
            records.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "region_id": region["region_id"],
                    "party_base_effect": float(np.dot(region_vec, candidate_vec)),
                }
            )
    return pd.DataFrame.from_records(records)
