"""Compile dated regional policy commitments into routing evidence."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from presidential_issue_engine.point_in_time import filter_available_by_election


SCHEMA_VERSION = "regional_policy_commitment_v1"
SOURCE_QUALITY = {
    "official_manifesto": 1.00,
    "official_campaign_record": 0.90,
    "academic_pre_election_record": 0.85,
    "contemporaneous_news_record": 0.75,
}


def compile_policy_alignment(
    registry: pd.DataFrame,
    candidate_issue_profile: pd.DataFrame,
    issue_importance: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the existing alignment schema without hand-entered strength."""

    required = {
        "event_id",
        "election_id",
        "candidate_name",
        "issue_name",
        "region_scope",
        "event_date",
        "available_date",
        "source_type",
        "source_url",
    }
    if registry.empty or not required.issubset(registry.columns):
        return pd.DataFrame(), pd.DataFrame()
    facts = filter_available_by_election(
        registry.copy(),
        dict(election_dates),
        source_name="regional_policy_commitment_registry",
    )
    profile = filter_available_by_election(
        candidate_issue_profile.copy(),
        dict(election_dates),
        source_name="regional_policy_commitment_candidate_profile",
    )
    importance = filter_available_by_election(
        issue_importance.copy(),
        dict(election_dates),
        source_name="regional_policy_commitment_issue_importance",
    )
    profile = (
        profile.sort_values("available_date")
        .drop_duplicates(["election_id", "candidate_name", "issue_name"], keep="last")
    )
    importance = (
        importance.sort_values("available_date")
        .drop_duplicates(["election_id", "issue_name"], keep="last")
    )
    importance["importance_multiplier"] = pd.to_numeric(
        importance["importance_multiplier"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    importance["election_importance_max"] = importance.groupby("election_id")[
        "importance_multiplier"
    ].transform("max")
    joined = facts.merge(
        profile[
            [
                "election_id",
                "candidate_name",
                "issue_name",
                "association_strength",
                "available_date",
            ]
        ].rename(columns={"available_date": "profile_available_date"}),
        on=["election_id", "candidate_name", "issue_name"],
        how="left",
    ).merge(
        importance[
            [
                "election_id",
                "issue_name",
                "importance_multiplier",
                "election_importance_max",
                "confidence",
                "available_date",
            ]
        ].rename(
            columns={
                "confidence": "importance_confidence",
                "available_date": "importance_available_date",
            }
        ),
        on=["election_id", "issue_name"],
        how="left",
    )
    joined["association_strength"] = pd.to_numeric(
        joined["association_strength"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    joined["importance_multiplier"] = pd.to_numeric(
        joined["importance_multiplier"], errors="coerce"
    ).fillna(1.0).clip(lower=0.0)
    joined["importance_normalized"] = (
        joined["importance_multiplier"]
        / joined["election_importance_max"].replace(0.0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    joined["affinity"] = np.sqrt(
        joined["association_strength"] * joined["importance_normalized"]
    )
    joined["source_quality"] = joined["source_type"].map(SOURCE_QUALITY).fillna(0.50)
    joined["importance_confidence"] = pd.to_numeric(
        joined["importance_confidence"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    joined["confidence"] = np.sqrt(
        joined["source_quality"] * joined["importance_confidence"]
    )
    date_columns = ["available_date", "profile_available_date", "importance_available_date"]
    joined["available_date"] = (
        joined[date_columns]
        .apply(pd.to_datetime, errors="coerce")
        .max(axis=1)
        .dt.date.astype(str)
    )
    output = pd.DataFrame(
        {
            "election_id": joined["election_id"].astype(str),
            "region_scope": joined["region_scope"].astype(str),
            "candidate_name": joined["candidate_name"].astype(str),
            "affinity": joined["affinity"],
            "available_date": joined["available_date"],
            "confidence": joined["confidence"],
            "evidence_type": "automatic_regional_policy_commitment",
            "source_url": joined["source_url"].astype(str),
            "notes": (
                "Factual commitment routed with automatically derived candidate-issue "
                "association and issue importance"
            ),
        }
    )
    joined["weighted_affinity"] = joined["affinity"] * joined["confidence"]
    joined["target_outcome_used"] = False
    joined["derivation_version"] = SCHEMA_VERSION
    return output, joined
