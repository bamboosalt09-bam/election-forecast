"""Derive non-major candidate regional organization from prior party ballots."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import (
    INDEPENDENT_BLOC,
    election_date,
    normalize_bloc,
)
from presidential_issue_engine.electorate_layers import (
    DIRECT_PARTY_ELECTION_TYPES,
    LAYER_ELECTION_TYPE_WEIGHTS,
    MAJOR_PARTY_CORE_BLOCS,
)
from presidential_issue_engine.point_in_time import filter_available_by_election


SCHEMA_VERSION = "speech_derived_candidate_regional_base_v1"
KEYS = ["election_id", "slot", "candidate_name"]


def build_automatic_candidate_regional_base(
    speech_context: pd.DataFrame,
    bloc_history: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> pd.DataFrame:
    """Build regional organization only where prior party ballots identify it.

    Major-party candidates are excluded because their regional party terrain is
    already represented elsewhere. Independent candidates are excluded because
    an aggregate independent vote is not candidate-specific evidence.
    """

    columns = [
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
        "candidate_bloc",
        "source_election_ids",
        "source_election_types",
        "source_vote_share",
        "regional_excess",
        "regional_excess_rank",
        "source_data_quality",
        "provenance_class",
        "derivation_version",
    ]
    speech_required = {
        *KEYS,
        "bloc",
        "organization_strength",
        "available_date",
        "confidence",
    }
    history_required = {
        "election_id",
        "election_type",
        "region_id",
        "bloc",
        "vote_share",
        "data_quality_weight",
    }
    if (
        speech_context.empty
        or bloc_history.empty
        or not speech_required.issubset(speech_context.columns)
        or not history_required.issubset(bloc_history.columns)
    ):
        return pd.DataFrame(columns=columns)

    candidates = filter_available_by_election(
        speech_context.copy(),
        election_dates,
        source_name="automatic_candidate_regional_base_speech",
    )
    candidates = (
        candidates.sort_values("available_date")
        .drop_duplicates(KEYS, keep="last")
        .reset_index(drop=True)
    )
    candidates["bloc"] = candidates["bloc"].map(normalize_bloc)
    eligible = ~candidates["bloc"].isin(
        {*MAJOR_PARTY_CORE_BLOCS, INDEPENDENT_BLOC}
    )
    candidates = candidates.loc[eligible].copy()

    history = bloc_history.copy()
    history["bloc"] = history["bloc"].map(normalize_bloc)
    history = history.loc[
        history["election_type"].astype(str).isin(DIRECT_PARTY_ELECTION_TYPES)
    ].copy()
    history["source_date"] = pd.to_datetime(
        history["election_id"].map(election_date), errors="coerce"
    )
    history["vote_share"] = pd.to_numeric(
        history["vote_share"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    history["data_quality_weight"] = pd.to_numeric(
        history["data_quality_weight"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    history["type_weight"] = history["election_type"].map(
        LAYER_ELECTION_TYPE_WEIGHTS
    ).fillna(0.0)
    history["evidence_weight"] = (
        history["type_weight"] * history["data_quality_weight"]
    )

    rows: list[dict[str, object]] = []
    for candidate in candidates.itertuples(index=False):
        target_date = pd.Timestamp(election_dates.get(str(candidate.election_id)))
        evidence = history.loc[
            history["bloc"].eq(str(candidate.bloc))
            & history["source_date"].notna()
            & history["source_date"].lt(target_date)
            & history["evidence_weight"].gt(0.0)
        ].copy()
        if evidence.empty:
            continue
        latest_date = evidence["source_date"].max()
        evidence = evidence.loc[evidence["source_date"].eq(latest_date)].copy()
        evidence["weighted_share"] = (
            evidence["vote_share"] * evidence["evidence_weight"]
        )
        regional = evidence.groupby("region_id", as_index=False).agg(
            weighted_share=("weighted_share", "sum"),
            evidence_weight=("evidence_weight", "sum"),
            source_data_quality=("data_quality_weight", "mean"),
            source_election_ids=(
                "election_id",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            source_election_types=(
                "election_type",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
        )
        regional["source_vote_share"] = regional["weighted_share"] / regional[
            "evidence_weight"
        ].replace(0.0, np.nan)
        regional["source_vote_share"] = regional["source_vote_share"].fillna(0.0)
        center = float(regional["source_vote_share"].mean())
        regional["regional_excess"] = (
            regional["source_vote_share"] - center
        ).clip(lower=0.0)
        regional = regional.loc[regional["regional_excess"].gt(0.0)].copy()
        if regional.empty:
            continue
        max_excess = float(regional["regional_excess"].max())
        max_share = float(regional["source_vote_share"].max())
        regional["regional_excess_rank"] = regional["regional_excess"].rank(
            method="average", pct=True
        )
        regional["regional_affinity"] = np.sqrt(
            (
                regional["regional_excess"] / max(max_excess, 1e-9)
                * regional["regional_excess_rank"]
            ).clip(0.0, 1.0)
        )
        organization = float(
            np.clip(
                pd.to_numeric(candidate.organization_strength, errors="coerce"),
                0.0,
                1.0,
            )
        )
        regional["organization_depth"] = np.sqrt(
            (
                regional["source_vote_share"] / max(max_share, 1e-9)
                * organization
            ).clip(0.0, 1.0)
        )
        candidate_confidence = float(
            np.clip(
                pd.to_numeric(candidate.confidence, errors="coerce"), 0.0, 1.0
            )
        )
        regional["confidence"] = (
            candidate_confidence * regional["source_data_quality"]
        ).clip(0.0, 1.0)
        available_date = max(
            pd.Timestamp(candidate.available_date), latest_date + pd.Timedelta(days=1)
        ).date().isoformat()
        for regional_row in regional.itertuples(index=False):
            rows.append(
                {
                    "election_id": str(candidate.election_id),
                    "slot": str(candidate.slot),
                    "candidate_name": str(candidate.candidate_name),
                    "region_id": str(regional_row.region_id),
                    "regional_affinity": float(regional_row.regional_affinity),
                    "organization_depth": float(regional_row.organization_depth),
                    "available_date": available_date,
                    "confidence": float(regional_row.confidence),
                    "source_type": "latest_prior_direct_party_ballot",
                    "notes": (
                        "Non-major regional organization from positive excess "
                        "in the latest prior direct-party ballot"
                    ),
                    "candidate_bloc": str(candidate.bloc),
                    "source_election_ids": str(regional_row.source_election_ids),
                    "source_election_types": str(
                        regional_row.source_election_types
                    ),
                    "source_vote_share": float(regional_row.source_vote_share),
                    "regional_excess": float(regional_row.regional_excess),
                    "regional_excess_rank": float(
                        regional_row.regional_excess_rank
                    ),
                    "source_data_quality": float(
                        regional_row.source_data_quality
                    ),
                    "provenance_class": "deterministic_source_derived",
                    "derivation_version": SCHEMA_VERSION,
                }
            )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["election_id", "slot", "regional_affinity"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
