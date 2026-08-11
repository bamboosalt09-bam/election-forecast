"""Transfer-event adjustments for presidential Utility values."""

from __future__ import annotations

import pandas as pd

from election_forecast.presidential.schemas import SLOTS


def load_transfer_events(
    events: pd.DataFrame,
    election_id: str,
    available_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize and filter transfer events for one election."""

    if events.empty or "election_id" not in events.columns:
        return _empty_events()
    frame = events.loc[events["election_id"] == election_id].copy()
    if frame.empty:
        return _empty_events()
    if "available_date" not in frame.columns:
        raise ValueError("transfer_events.csv is missing available_date")
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    if frame["available_date"].isna().any():
        raise ValueError("transfer_events.csv contains missing or invalid available_date")
    if available_date is not None:
        cutoff = pd.to_datetime(available_date)
        frame = frame.loc[frame["available_date"] <= cutoff].copy()
    for column in ["source_slot", "target_slot"]:
        frame[column] = frame[column].astype(str)
        invalid = sorted(set(frame[column]) - set(SLOTS))
        if invalid:
            raise ValueError(f"transfer_events.csv has unsupported {column} values: {invalid}")
    for column in ["transfer_strength", "transfer_rate", "abstention_rate"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    frame["region_id"] = frame["region_id"].fillna("ALL").astype(str)
    return frame


def compute_transfer_adjustments(utilities: pd.DataFrame, transfer_events: pd.DataFrame) -> pd.DataFrame:
    """Compute utility adjustment rows for each applicable event."""

    if utilities.empty or transfer_events.empty:
        return _empty_contributions()

    rows: list[dict[str, object]] = []
    for _, event in transfer_events.iterrows():
        event_regions = _event_regions(utilities, event["election_id"], event["region_id"])
        target_delta = float(event["transfer_strength"] * event["transfer_rate"])
        source_delta = -float(event["transfer_strength"] * (event["transfer_rate"] + event["abstention_rate"]))
        alpha_delta = float(event["transfer_strength"] * event["abstention_rate"])

        for region_id in event_regions:
            model_names = utilities.loc[
                (utilities["election_id"] == event["election_id"]) & (utilities["region_id"] == region_id),
                "model_name",
            ].drop_duplicates()
            for model_name in model_names:
                rows.extend(
                    [
                        _contribution_row(event, region_id, model_name, event["target_slot"], target_delta),
                        _contribution_row(event, region_id, model_name, event["source_slot"], source_delta),
                    ]
                )
                if alpha_delta:
                    rows.append(_contribution_row(event, region_id, model_name, "alpha", alpha_delta))

    return pd.DataFrame(rows, columns=_empty_contributions().columns)


def apply_transfer_adjustments(
    utilities: pd.DataFrame,
    transfer_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply transfer-event Utility adjustments and return contribution rows."""

    adjusted = utilities.copy()
    contributions = compute_transfer_adjustments(adjusted, transfer_events)
    if contributions.empty:
        return adjusted, contributions

    for _, row in contributions.iterrows():
        mask = (
            (adjusted["election_id"] == row["election_id"])
            & (adjusted["region_id"] == row["region_id"])
            & (adjusted["model_name"] == row["model_name"])
            & (adjusted["slot"] == row["target_slot"])
        )
        adjusted.loc[mask, "utility"] = adjusted.loc[mask, "utility"] + row["utility_adjustment"]
    return adjusted, contributions


def _event_regions(utilities: pd.DataFrame, election_id: str, region_id: str) -> list[str]:
    election_regions = utilities.loc[utilities["election_id"] == election_id, "region_id"].drop_duplicates()
    if region_id == "ALL":
        return list(election_regions)
    return [region_id] if region_id in set(election_regions) else []


def _contribution_row(
    event: pd.Series,
    region_id: str,
    model_name: str,
    target_slot: str,
    utility_adjustment: float,
) -> dict[str, object]:
    return {
        "election_id": event["election_id"],
        "region_id": region_id,
        "source_slot": event["source_slot"],
        "target_slot": target_slot,
        "model_name": model_name,
        "transfer_strength": float(event["transfer_strength"]),
        "transfer_rate": float(event["transfer_rate"]),
        "abstention_rate": float(event["abstention_rate"]),
        "utility_adjustment": utility_adjustment,
    }


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "election_id",
            "event_date",
            "available_date",
            "source_slot",
            "target_slot",
            "region_id",
            "transfer_strength",
            "transfer_rate",
            "abstention_rate",
            "notes",
        ]
    )


def _empty_contributions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "election_id",
            "region_id",
            "source_slot",
            "target_slot",
            "model_name",
            "transfer_strength",
            "transfer_rate",
            "abstention_rate",
            "utility_adjustment",
        ]
    )
