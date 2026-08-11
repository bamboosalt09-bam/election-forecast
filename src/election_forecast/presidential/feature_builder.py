"""Build presidential political variables from rule-based feature CSVs."""

from __future__ import annotations

import pandas as pd

from election_forecast.presidential.schemas import SLOTS, VALID_VARIABLE_NAMES


def build_political_variables(
    manual_variables: pd.DataFrame,
    party_controversy: pd.DataFrame,
    candidate_tone: pd.DataFrame,
    regions: pd.DataFrame,
) -> pd.DataFrame:
    """Build ``political_variables.csv`` rows from manual and generated features.

    The generated inputs are already aggregated feature CSVs. This function does
    not read article text, call an AI service, or infer ideology/camp labels.
    """

    generated = pd.concat(
        [
            party_controversy_to_variables(party_controversy, regions),
            candidate_tone_to_variables(candidate_tone, regions),
        ],
        ignore_index=True,
    )
    manual = _normalize_manual_variables(manual_variables)
    combined = pd.concat([generated, manual], ignore_index=True)
    combined["_priority"] = combined["source_note"].str.startswith("manual").astype(int)
    combined = combined.sort_values(
        ["election_id", "region_id", "slot", "variable_name", "_priority", "available_date"]
    )
    combined = combined.drop_duplicates(
        ["election_id", "region_id", "slot", "variable_name"],
        keep="last",
    )
    combined = combined.drop(columns=["_priority"])
    return combined[
        ["election_id", "region_id", "slot", "variable_name", "variable_value", "available_date", "source_note"]
    ].sort_values(["election_id", "region_id", "slot", "variable_name"]).reset_index(drop=True)


def party_controversy_to_variables(party_controversy: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Convert party controversy scores into national_mood rows for all regions."""

    frame = party_controversy.copy()
    if frame.empty:
        return _empty_variables()
    frame = _validate_slots(frame, "slot")
    frame["variable_name"] = "national_mood"
    frame["variable_value"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    frame["source_note"] = "party_controversy_scores: " + frame["source_note"].fillna("")
    return _expand_national_rows(frame, regions)


def candidate_tone_to_variables(candidate_tone: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Convert candidate tone scores into candidate_strength rows for all regions."""

    frame = candidate_tone.copy()
    if frame.empty:
        return _empty_variables()
    frame = _validate_slots(frame, "slot")
    frame["variable_name"] = "candidate_strength"
    frame["variable_value"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    frame["source_note"] = "candidate_tone_scores: " + frame["source_note"].fillna("")
    return _expand_national_rows(frame, regions)


def _expand_national_rows(frame: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    region_ids = regions[["region_id"]].drop_duplicates()
    expanded = frame[
        ["election_id", "slot", "variable_name", "variable_value", "available_date", "source_note"]
    ].merge(region_ids, how="cross")
    return expanded[
        ["election_id", "region_id", "slot", "variable_name", "variable_value", "available_date", "source_note"]
    ]


def _normalize_manual_variables(manual_variables: pd.DataFrame) -> pd.DataFrame:
    if manual_variables.empty:
        return _empty_variables()
    frame = manual_variables.copy()
    frame = _validate_slots(frame, "slot")
    invalid_variables = sorted(set(frame["variable_name"]) - set(VALID_VARIABLE_NAMES))
    if invalid_variables:
        raise ValueError(f"Unsupported variable_name values: {invalid_variables}")
    frame["variable_value"] = pd.to_numeric(frame["variable_value"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    frame["source_note"] = "manual_political_variables: " + frame["source_note"].fillna("")
    return frame[
        ["election_id", "region_id", "slot", "variable_name", "variable_value", "available_date", "source_note"]
    ]


def _validate_slots(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.copy()
    result[column] = result[column].astype(str)
    invalid_slots = sorted(set(result[column]) - set(SLOTS))
    if invalid_slots:
        raise ValueError(f"Unsupported slot values: {invalid_slots}")
    return result


def _empty_variables() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "election_id",
            "region_id",
            "slot",
            "variable_name",
            "variable_value",
            "available_date",
            "source_note",
        ]
    )
