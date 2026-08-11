"""Prior-only regional-party reservoir and candidate-routing evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import (
    CONSERVATIVE_BLOC,
    LIBERAL_BLOC,
    PROGRESSIVE_BLOC,
    THIRD_BLOC,
    election_date,
    normalize_bloc,
)
from presidential_issue_engine.chungcheong_identity import CHUNGCHEONG
from presidential_issue_engine.electorate_layers import REGIONALIST_PARTY_LABELS
from presidential_issue_engine.electorate_layers import LAYER_ELECTION_TYPE_WEIGHTS


FULL_HISTORY_TYPE_WEIGHTS = {
    "national_assembly_pr": 1.0,
    "assembly_pr": 1.0,
    "metro_council_pr": 1.0,
    "local_council_pr": 1.0,
    "assembly_district": 1.0,
    "national_assembly_district": 1.0,
    "metro_council_district": 0.80,
    "local_council_district": 0.50,
    "metro_governor": 0.15,
    "local_governor": 0.10,
    "presidential": 0.35,
}
SEMANTIC_HISTORY_TYPE_WEIGHTS = {
    election_type: float(weight)
    for election_type, weight in LAYER_ELECTION_TYPE_WEIGHTS.items()
    if float(weight) > 0.0
}
MAJOR_OR_PROGRESSIVE = {CONSERVATIVE_BLOC, LIBERAL_BLOC, PROGRESSIVE_BLOC}


def build_full_history_identity_events(
    history: pd.DataFrame,
    *,
    date_resolver: Callable[[object], pd.Timestamp | None] = election_date,
    type_weights: Mapping[str, float] = FULL_HISTORY_TYPE_WEIGHTS,
) -> pd.DataFrame:
    """Measure regional-party excess across all usable election families."""

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
        raise ValueError(f"regional-party history missing columns: {sorted(missing)}")
    work = history.loc[
        history["election_type"].astype(str).isin(type_weights)
    ].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(date_resolver), errors="coerce"
    )
    work = work.loc[work["event_date"].notna()].copy()
    raw_bloc = work["bloc"].fillna("").astype(str).str.strip()
    normalized = raw_bloc.map(normalize_bloc)
    work["identity_vote_share"] = pd.to_numeric(
        work["vote_share"], errors="coerce"
    ).fillna(0.0).where(
        normalized.eq(THIRD_BLOC) | raw_bloc.isin(REGIONALIST_PARTY_LABELS),
        0.0,
    )
    work["quality"] = pd.to_numeric(
        work["data_quality_weight"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
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
    baseline = grouped.groupby("election_id")["identity_share"].median()
    grouped["national_identity_baseline"] = grouped["election_id"].map(baseline)
    grouped["identity_excess"] = (
        grouped["identity_share"] - grouped["national_identity_baseline"]
    ).clip(lower=0.0, upper=0.40)
    grouped["type_weight"] = grouped["election_type"].map(
        type_weights
    ).fillna(0.0)
    return grouped.sort_values(
        ["event_date", "election_id", "region_id"]
    ).reset_index(drop=True)


def _relative_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    low = float(numeric.min())
    high = float(numeric.max())
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-12:
        return pd.Series(0.5, index=values.index, dtype=float)
    return ((numeric - low) / (high - low)).clip(0.0, 1.0)


def build_automatic_nonmajor_alignment(
    history: pd.DataFrame,
    candidate_context: pd.DataFrame,
    candidate_landscape: pd.DataFrame,
    bloc_landscape: pd.DataFrame,
    *,
    type_weights: Mapping[str, float] = FULL_HISTORY_TYPE_WEIGHTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route a prior regional-party reservoir to evidenced non-major candidates."""

    events = build_full_history_identity_events(history, type_weights=type_weights)
    context = candidate_context.copy()
    landscape = candidate_landscape.copy()
    context["available_date"] = pd.to_datetime(
        context["available_date"], errors="coerce"
    )
    landscape["available_date"] = pd.to_datetime(
        landscape["available_date"], errors="coerce"
    )
    context["normalized_bloc"] = context["bloc"].map(normalize_bloc)
    candidate = context.merge(
        landscape,
        on=["election_id", "slot", "candidate_name"],
        how="inner",
        suffixes=("_context", "_landscape"),
    )
    if "candidate_role" in candidate.columns:
        candidate = candidate.loc[
            ~candidate["candidate_role"].fillna("").astype(str).eq("withdrawn")
        ].copy()

    bloc_vectors = bloc_landscape.copy()
    bloc_vectors["normalized_bloc"] = bloc_vectors["bloc"].map(normalize_bloc)
    third = bloc_vectors.loc[bloc_vectors["normalized_bloc"].eq(THIRD_BLOC)].copy()
    orientation = 1.0
    if not third.empty:
        orientation_value = float(
            pd.to_numeric(third["conservative"], errors="coerce").mean()
            - pd.to_numeric(third["liberal"], errors="coerce").mean()
        )
        orientation = 1.0 if orientation_value >= 0.0 else -1.0

    audit_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    for election_id, group in candidate.groupby("election_id", sort=True):
        cutoff = election_date(str(election_id))
        if cutoff is None:
            continue
        prior = events.loc[
            events["event_date"].lt(pd.Timestamp(cutoff))
            & events["region_id"].astype(str).isin(CHUNGCHEONG)
        ].copy()
        reliable_events = int(prior["election_id"].nunique())
        reservoir_evidence = float(
            (prior["identity_excess"] * prior["quality"] * prior["type_weight"]).sum()
        )
        work = group.copy()
        work["ideology_margin"] = orientation * (
            pd.to_numeric(work["conservative"], errors="coerce").fillna(0.0)
            - pd.to_numeric(work["liberal"], errors="coerce").fillna(0.0)
        )
        work["ideology_rank"] = _relative_rank(work["ideology_margin"])
        outsider = pd.to_numeric(
            work.get("outsider_status", 0.0), errors="coerce"
        ).fillna(0.0)
        work["nonmajor_candidate"] = (
            ~work["normalized_bloc"].isin(MAJOR_OR_PROGRESSIVE)
            & (work["normalized_bloc"].eq(THIRD_BLOC) | outsider.ge(0.50))
        )
        eligible = work.loc[
            work["nonmajor_candidate"] & work["ideology_rank"].ge(0.50)
        ].copy()
        chosen_index = eligible["ideology_rank"].idxmax() if not eligible.empty else None
        for index, row in work.iterrows():
            selected = chosen_index is not None and index == chosen_index
            audit_rows.append(
                {
                    "election_id": str(election_id),
                    "slot": str(row["slot"]),
                    "candidate_name": str(row["candidate_name"]),
                    "normalized_bloc": str(row["normalized_bloc"]),
                    "ideology_margin": float(row["ideology_margin"]),
                    "ideology_rank": float(row["ideology_rank"]),
                    "nonmajor_candidate": bool(row["nonmajor_candidate"]),
                    "selected": bool(selected),
                    "prior_identity_events": reliable_events,
                    "prior_reservoir_evidence": reservoir_evidence,
                    "target_outcome_used": False,
                }
            )
        if chosen_index is None or reliable_events < 2 or reservoir_evidence <= 0.0:
            continue
        chosen = work.loc[chosen_index]
        confidence = float(
            np.sqrt(
                np.clip(float(chosen.get("confidence_context", 0.0)), 0.0, 1.0)
                * np.clip(float(chosen.get("confidence_landscape", 0.0)), 0.0, 1.0)
            )
        )
        available_date = max(
            pd.Timestamp(chosen["available_date_context"]),
            pd.Timestamp(chosen["available_date_landscape"]),
        )
        if available_date >= pd.Timestamp(cutoff):
            available_date = pd.Timestamp(cutoff) - pd.Timedelta(days=1)
        alignment_rows.append(
            {
                "election_id": str(election_id),
                "region_scope": "chungcheong",
                "candidate_name": str(chosen["candidate_name"]),
                "affinity": float(chosen["ideology_rank"]),
                "available_date": available_date.date().isoformat(),
                "confidence": confidence,
                "evidence_type": "automatic_regional_party_candidate_fit",
                "source_url": "official_nec_history_and_assembly_vectors",
                "notes": (
                    "Prior-only Chungcheong regional-party reservoir routed to the "
                    "strongest compatible active non-major candidate"
                ),
            }
        )
    columns = [
        "election_id",
        "region_scope",
        "candidate_name",
        "affinity",
        "available_date",
        "confidence",
        "evidence_type",
        "source_url",
        "notes",
    ]
    return pd.DataFrame(alignment_rows, columns=columns), pd.DataFrame(audit_rows)
