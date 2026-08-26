"""Political variable preparation for presidential utility models."""

from __future__ import annotations

import pandas as pd

from election_forecast.presidential.schemas import SLOTS


def prepare_variables(
    variables: pd.DataFrame,
    election_id: str,
    available_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter variables to one election and optional information cutoff."""

    frame = variables.loc[variables["election_id"] == election_id].copy()
    if frame.empty:
        return frame
    if "available_date" not in frame.columns:
        raise ValueError("political_variables.csv is missing available_date")
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    if frame["available_date"].isna().any():
        raise ValueError("political_variables.csv contains missing or invalid available_date")
    if available_date is not None:
        cutoff = pd.to_datetime(available_date)
        frame = frame.loc[frame["available_date"] <= cutoff].copy()
    frame["slot"] = frame["slot"].astype(str)
    invalid_slots = sorted(set(frame["slot"]) - set(SLOTS))
    if invalid_slots:
        raise ValueError(f"political_variables.csv contains unsupported slots: {invalid_slots}")
    # A value that will not parse is missing, not zero. Filling it with 0.0 made
    # a parse failure indistinguishable from a genuine neutral reading - the
    # same defect V32 removed from the prospective feature assembly, in a
    # different room. This path does not feed the frozen forecast, but the rule
    # is the rule.
    numeric = pd.to_numeric(frame["variable_value"], errors="coerce")
    unparsed = numeric.isna()
    if bool(unparsed.any()):
        offenders = (
            frame.loc[unparsed, ["slot", "variable_name"]]
            if "variable_name" in frame.columns
            else frame.loc[unparsed, ["slot"]]
        )
        raise ValueError(
            "political_variables.csv has "
            f"{int(unparsed.sum())} variable_value entries that are not numeric; "
            "a value that does not parse is missing, not zero. First few: "
            f"{offenders.head(5).to_dict('records')}"
        )
    # Clipping to the declared range stays: saturation at a stated bound is a
    # visible policy, and tests/test_variable_model_softmax.py exercises it on
    # purpose. Only the silent substitution above was the defect.
    frame["variable_value"] = numeric.clip(-1.0, 1.0)
    return frame
