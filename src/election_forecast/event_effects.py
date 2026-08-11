"""Event layer for coalition, split, withdrawal, endorsement, and conflict effects."""

from __future__ import annotations

import pandas as pd


SUPPORTED_EVENT_TYPES = {
    "coalition",
    "split",
    "withdrawal_transfer",
    "endorsement",
    "party_conflict",
}


def compute_event_effects(
    event_effects: pd.DataFrame,
    candidates: pd.DataFrame,
    regions: pd.DataFrame,
    forecast_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Compute aggregate event effects by candidate and region.

    Transfer mechanics are intentionally explicit: an event's contribution is
    ``transfer_rate x voter_compliance x effect_strength``. CSV inputs decide
    the sign and magnitude; the code does not hardcode political interpretation.
    """

    output_columns = ["candidate_id", "region_id", "event_effect"]
    if event_effects.empty:
        return pd.DataFrame(columns=output_columns)

    cutoff = pd.Timestamp(forecast_date)
    frame = event_effects.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame = frame.loc[frame["event_date"].le(cutoff)].copy()
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    region_ids = set(regions["region_id"].astype(str))
    all_regions = list(region_ids)
    candidate_ids = set(candidates["candidate_id"].astype(str))
    records: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        transfer_rate = float(pd.to_numeric(pd.Series([row["transfer_rate"]]), errors="coerce").fillna(0.0).iloc[0])
        compliance = float(pd.to_numeric(pd.Series([row["voter_compliance"]]), errors="coerce").fillna(0.0).iloc[0])
        strength = float(pd.to_numeric(pd.Series([row["effect_strength"]]), errors="coerce").fillna(0.0).iloc[0])
        magnitude = transfer_rate * compliance * strength

        raw_region = "" if pd.isna(row.get("region_id")) else str(row.get("region_id"))
        affected_regions = all_regions if raw_region in {"", "ALL", "all", "national"} else [raw_region]
        affected_regions = [region_id for region_id in affected_regions if region_id in region_ids]

        source = "" if pd.isna(row.get("source_candidate_id")) else str(row.get("source_candidate_id"))
        target = "" if pd.isna(row.get("target_candidate_id")) else str(row.get("target_candidate_id"))

        for region_id in affected_regions:
            if source and source in candidate_ids:
                records.append({"candidate_id": source, "region_id": region_id, "event_effect": -magnitude})
            if target and target in candidate_ids:
                records.append({"candidate_id": target, "region_id": region_id, "event_effect": magnitude})

    if not records:
        return pd.DataFrame(columns=output_columns)

    return pd.DataFrame.from_records(records).groupby(["candidate_id", "region_id"], as_index=False)[
        "event_effect"
    ].sum()
