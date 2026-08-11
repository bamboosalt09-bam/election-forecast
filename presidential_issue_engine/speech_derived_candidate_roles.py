"""Derive third-candidate stature from dated Assembly-context evidence.

The compiler contains no election-specific candidate constants. It combines
absolute signal levels with within-election ranks so a highly discussed
candidate is not automatically treated as a highly convertible candidate.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from presidential_issue_engine.electorate_layers import MAJOR_PARTY_CORE_BLOCS
from presidential_issue_engine.point_in_time import filter_available_by_election


SCHEMA_VERSION = "speech_derived_candidate_role_v1"
KEYS = ["election_id", "slot", "candidate_name"]


def _bounded(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(column, pd.Series(0.0, index=frame.index)), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)


def _within_election_rank(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(frame["election_id"]).rank(
        method="average", pct=True
    ).fillna(0.0).clip(0.0, 1.0)


def _level_rank_bridge(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    rank = _within_election_rank(frame, values)
    return np.sqrt((values * rank).clip(0.0, 1.0))


def _geometric_mean(columns: list[pd.Series]) -> pd.Series:
    if not columns:
        return pd.Series(dtype=float)
    matrix = np.column_stack([column.clip(0.0, 1.0) for column in columns])
    return pd.Series(
        np.prod(np.clip(matrix, 1e-9, 1.0), axis=1) ** (1.0 / matrix.shape[1]),
        index=columns[0].index,
    )


def build_automatic_third_candidate_profile(
    speech_context: pd.DataFrame,
    public_treatment: pd.DataFrame,
    political_landscape: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> pd.DataFrame:
    """Compile a continuous non-major-candidate profile without outcomes."""

    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "viability",
        "centrist_appeal",
        "anti_major_party_appeal",
        "regional_base_overlap",
        "available_date",
        "confidence",
        "notes",
        "provenance_class",
        "derivation_version",
        "major_party_core_eligible",
        "serious_component",
        "legitimacy_component",
        "organization_component",
        "party_support_component",
        "coalition_stability_component",
    ]
    speech_required = {
        *KEYS,
        "bloc",
        "party_elite_support_score",
        "party_elite_fragmentation_score",
        "organization_strength",
        "available_date",
        "confidence",
    }
    treatment_required = {
        *KEYS,
        "serious_contender_score",
        "legitimacy_score",
        "alternative_score",
        "protest_vote_score",
        "available_date",
        "confidence",
    }
    if (
        speech_context.empty
        or public_treatment.empty
        or not speech_required.issubset(speech_context.columns)
        or not treatment_required.issubset(public_treatment.columns)
    ):
        return pd.DataFrame(columns=columns)

    speech = filter_available_by_election(
        speech_context.copy(),
        election_dates,
        source_name="automatic_third_candidate_speech_context",
    )
    treatment = filter_available_by_election(
        public_treatment.copy(),
        election_dates,
        source_name="automatic_third_candidate_public_treatment",
    )
    speech = speech.sort_values("available_date").drop_duplicates(KEYS, keep="last")
    treatment = treatment.sort_values("available_date").drop_duplicates(
        KEYS, keep="last"
    )
    speech = speech.rename(
        columns={
            "available_date": "speech_available_date",
            "confidence": "speech_confidence",
        }
    )
    treatment = treatment.rename(
        columns={
            "available_date": "treatment_available_date",
            "confidence": "treatment_confidence",
        }
    )
    frame = speech.merge(treatment, on=KEYS, how="inner", validate="one_to_one")

    landscape = political_landscape.copy()
    landscape_required = {*KEYS, "centrist", "anti_establishment"}
    if not landscape.empty and landscape_required.issubset(landscape.columns):
        if "available_date" in landscape.columns:
            landscape = filter_available_by_election(
                landscape,
                election_dates,
                source_name="automatic_third_candidate_political_landscape",
            )
        keep = [
            *KEYS,
            "centrist",
            "anti_establishment",
            *[
                column
                for column in ["available_date", "confidence"]
                if column in landscape.columns
            ],
        ]
        landscape = landscape[keep].sort_values(
            "available_date" if "available_date" in keep else "election_id"
        ).drop_duplicates(KEYS, keep="last")
        landscape = landscape.rename(
            columns={
                "available_date": "landscape_available_date",
                "confidence": "landscape_confidence",
            }
        )
        frame = frame.merge(landscape, on=KEYS, how="left", validate="one_to_one")

    major = frame["bloc"].astype(str).str.strip().isin(MAJOR_PARTY_CORE_BLOCS)
    frame["major_party_core_eligible"] = major

    serious = _bounded(frame, "serious_contender_score")
    legitimacy = _bounded(frame, "legitimacy_score")
    organization = _bounded(frame, "organization_strength")
    party_support = _bounded(frame, "party_elite_support_score")
    stability = 1.0 - _bounded(frame, "party_elite_fragmentation_score")
    components = {
        "serious_component": _level_rank_bridge(frame, serious),
        "legitimacy_component": _level_rank_bridge(frame, legitimacy),
        "organization_component": _level_rank_bridge(frame, organization),
        "party_support_component": _level_rank_bridge(frame, party_support),
        "coalition_stability_component": _level_rank_bridge(frame, stability),
    }
    for name, values in components.items():
        frame[name] = values
    frame["viability"] = pd.concat(components, axis=1).mean(axis=1).clip(0.0, 1.0)

    centrist = _bounded(frame, "centrist")
    anti_establishment = _bounded(frame, "anti_establishment")
    frame["centrist_appeal"] = (
        0.5 * centrist + 0.5 * _within_election_rank(frame, centrist)
    ).clip(0.0, 1.0)
    frame["anti_major_party_appeal"] = pd.concat(
        [
            _bounded(frame, "alternative_score"),
            _bounded(frame, "protest_vote_score"),
            _bounded(frame, "outsider_status"),
            anti_establishment,
        ],
        axis=1,
    ).mean(axis=1).clip(0.0, 1.0)
    frame["regional_base_overlap"] = np.sqrt(
        (organization * party_support).clip(0.0, 1.0)
    )

    speech_confidence = _bounded(frame, "speech_confidence")
    treatment_confidence = _bounded(frame, "treatment_confidence")
    landscape_confidence = _bounded(frame, "landscape_confidence")
    landscape_confidence = landscape_confidence.where(
        landscape_confidence.gt(0.0),
        _geometric_mean([speech_confidence, treatment_confidence]),
    )
    frame["confidence"] = _geometric_mean(
        [speech_confidence, treatment_confidence, landscape_confidence]
    ).clip(0.0, 1.0)

    date_columns = [
        column
        for column in [
            "speech_available_date",
            "treatment_available_date",
            "landscape_available_date",
        ]
        if column in frame.columns
    ]
    frame["available_date"] = (
        frame[date_columns]
        .apply(pd.to_datetime, errors="coerce")
        .max(axis=1)
        .dt.date.astype(str)
    )
    frame["notes"] = (
        "Equal-weight speech stature with Assembly-derived landscape; no "
        "candidate-specific viability constant or vote outcome"
    )
    frame["provenance_class"] = "deterministic_source_derived"
    frame["derivation_version"] = SCHEMA_VERSION
    return (
        frame.loc[~frame["major_party_core_eligible"], columns]
        .sort_values(["election_id", "slot", "candidate_name"])
        .reset_index(drop=True)
    )
