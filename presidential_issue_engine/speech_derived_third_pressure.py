"""Derive third-candidate source-lane pressure without election constants."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import (
    CONSERVATIVE_BLOC,
    LIBERAL_BLOC,
    PROGRESSIVE_BLOC,
)
from presidential_issue_engine.electorate_layers import MAJOR_PARTY_CORE_BLOCS
from presidential_issue_engine.point_in_time import filter_available_by_election


SCHEMA_VERSION = "speech_derived_third_pressure_v1"
PROFILE_KEYS = ["election_id", "slot", "candidate_name"]
AXES = ["conservative", "liberal", "progressive"]


def _bounded(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).clip(0.0, 1.0)


def _fallback_axis(bloc: pd.Series, axis: str) -> pd.Series:
    normalized = bloc.astype(str).str.strip()
    if axis == "conservative":
        return normalized.eq(CONSERVATIVE_BLOC).astype(float)
    if axis == "liberal":
        return normalized.eq(LIBERAL_BLOC).astype(float)
    if axis == "progressive":
        return normalized.eq(PROGRESSIVE_BLOC).astype(float)
    return pd.Series(0.0, index=bloc.index)


def _latest_available(
    frame: pd.DataFrame,
    election_dates: Mapping[str, object],
    source_name: str,
    keys: list[str],
) -> pd.DataFrame:
    filtered = filter_available_by_election(
        frame.copy(), election_dates, source_name=source_name
    )
    if filtered.empty:
        return filtered
    return (
        filtered.sort_values("available_date")
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )


def build_automatic_third_candidate_pressure(
    third_profile: pd.DataFrame,
    speech_context: pd.DataFrame,
    political_landscape: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> pd.DataFrame:
    """Allocate a third candidate's draw capacity across major-party lanes.

    Total draw capacity is the geometric mean of centrist and anti-major-party
    appeal. Source-lane shares are normalized affinities, where affinity is the
    equal mean of ideological proximity, centrist appeal, and anti-major-party
    appeal. Viability is deliberately not included because the downstream
    feature compiler already multiplies pressure by third-candidate viability.
    """

    columns = [
        "election_id",
        "slot",
        "source_slot",
        "transfer_pressure",
        "available_date",
        "confidence",
        "notes",
        "candidate_name",
        "source_candidate_name",
        "source_bloc",
        "ideological_affinity",
        "lane_affinity",
        "lane_share",
        "draw_propensity",
        "provenance_class",
        "derivation_version",
    ]
    profile_required = {
        *PROFILE_KEYS,
        "centrist_appeal",
        "anti_major_party_appeal",
        "available_date",
        "confidence",
    }
    speech_required = {
        *PROFILE_KEYS,
        "bloc",
        "available_date",
        "confidence",
    }
    if (
        third_profile.empty
        or speech_context.empty
        or not profile_required.issubset(third_profile.columns)
        or not speech_required.issubset(speech_context.columns)
    ):
        return pd.DataFrame(columns=columns)

    profile = _latest_available(
        third_profile,
        election_dates,
        "automatic_third_pressure_profile",
        PROFILE_KEYS,
    ).rename(
        columns={
            "available_date": "profile_available_date",
            "confidence": "profile_confidence",
        }
    )
    speech = _latest_available(
        speech_context,
        election_dates,
        "automatic_third_pressure_speech",
        PROFILE_KEYS,
    )
    third_speech = speech[
        [*PROFILE_KEYS, "bloc", "available_date", "confidence"]
    ].rename(
        columns={
            "bloc": "third_bloc",
            "available_date": "third_speech_available_date",
            "confidence": "third_speech_confidence",
        }
    )
    source = speech.loc[
        speech["bloc"].astype(str).str.strip().isin(MAJOR_PARTY_CORE_BLOCS),
        [*PROFILE_KEYS, "bloc", "available_date", "confidence"],
    ].rename(
        columns={
            "slot": "source_slot",
            "candidate_name": "source_candidate_name",
            "bloc": "source_bloc",
            "available_date": "source_available_date",
            "confidence": "source_confidence",
        }
    )
    source = source.drop(columns=["slot"], errors="ignore")

    third = profile.merge(third_speech, on=PROFILE_KEYS, how="inner", validate="one_to_one")
    landscape_required = {*PROFILE_KEYS, *AXES, "available_date", "confidence"}
    if not political_landscape.empty and landscape_required.issubset(
        political_landscape.columns
    ):
        landscape = _latest_available(
            political_landscape,
            election_dates,
            "automatic_third_pressure_landscape",
            PROFILE_KEYS,
        )[[*PROFILE_KEYS, *AXES, "available_date", "confidence"]].rename(
            columns={
                "available_date": "landscape_available_date",
                "confidence": "landscape_confidence",
            }
        )
        third = third.merge(landscape, on=PROFILE_KEYS, how="left", validate="one_to_one")

    for axis in AXES:
        observed = _bounded(third, axis)
        third[axis] = observed.where(observed.notna(), _fallback_axis(third["third_bloc"], axis))
    third["landscape_confidence"] = _bounded(third, "landscape_confidence").fillna(
        _bounded(third, "profile_confidence")
    )

    expanded = third.merge(source, on="election_id", how="inner", validate="one_to_many")
    expanded = expanded.loc[expanded["slot"].ne(expanded["source_slot"])].copy()
    conservative_source = expanded["source_bloc"].astype(str).str.strip().eq(
        CONSERVATIVE_BLOC
    )
    liberal_affinity = expanded[["liberal", "progressive"]].max(axis=1)
    expanded["ideological_affinity"] = np.where(
        conservative_source,
        expanded["conservative"],
        liberal_affinity,
    )
    centrist = _bounded(expanded, "centrist_appeal").fillna(0.0)
    anti_major = _bounded(expanded, "anti_major_party_appeal").fillna(0.0)
    expanded["lane_affinity"] = pd.concat(
        [expanded["ideological_affinity"], centrist, anti_major], axis=1
    ).mean(axis=1)
    lane_total = expanded.groupby(["election_id", "slot"])["lane_affinity"].transform(
        "sum"
    )
    lane_count = expanded.groupby(["election_id", "slot"])["source_slot"].transform(
        "count"
    )
    expanded["lane_share"] = np.where(
        lane_total.gt(0.0),
        expanded["lane_affinity"] / lane_total,
        1.0 / lane_count.clip(lower=1),
    )
    expanded["draw_propensity"] = np.sqrt((centrist * anti_major).clip(0.0, 1.0))
    expanded["transfer_pressure"] = (
        expanded["draw_propensity"] * expanded["lane_share"]
    ).clip(0.0, 1.0)

    confidence_columns = [
        _bounded(expanded, "profile_confidence").fillna(0.0),
        _bounded(expanded, "third_speech_confidence").fillna(0.0),
        _bounded(expanded, "source_confidence").fillna(0.0),
        _bounded(expanded, "landscape_confidence").fillna(0.0),
    ]
    confidence_matrix = np.column_stack(confidence_columns)
    expanded["confidence"] = np.prod(
        np.clip(confidence_matrix, 1e-9, 1.0), axis=1
    ) ** (1.0 / confidence_matrix.shape[1])

    date_columns = [
        column
        for column in [
            "profile_available_date",
            "third_speech_available_date",
            "source_available_date",
            "landscape_available_date",
        ]
        if column in expanded.columns
    ]
    expanded["available_date"] = (
        expanded[date_columns]
        .apply(pd.to_datetime, errors="coerce")
        .max(axis=1)
        .dt.date.astype(str)
    )
    expanded["notes"] = (
        "Automatic source-lane pressure from political-axis affinity and "
        "centrist/anti-major draw propensity; no election-specific constant"
    )
    expanded["provenance_class"] = "deterministic_source_derived"
    expanded["derivation_version"] = SCHEMA_VERSION
    return (
        expanded[columns]
        .sort_values(["election_id", "slot", "source_slot"])
        .reset_index(drop=True)
    )
