"""Standardize raw presidential results to A/B/C/alpha slots."""

from __future__ import annotations

import pandas as pd

from election_forecast.presidential.schemas import SLOTS, normalize_slots
from election_forecast.presidential.validate_presidential import (
    validate_slot_coverage,
    validate_vote_share_sums,
)


def standardize_presidential_results(
    raw_results: pd.DataFrame,
    candidate_slots: pd.DataFrame,
    regions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map raw candidate results into A/B/C/alpha rows.

    A/B/C are matched from ``candidate_slots``. Any other candidate in the raw
    data is aggregated into alpha. Inactive slots, such as 2012 C, are emitted
    as explicit zero-vote rows.
    """

    slots = normalize_slots(candidate_slots)
    raw = raw_results.copy()
    raw["votes"] = pd.to_numeric(raw["votes"], errors="coerce").fillna(0.0)

    region_lookup = regions[["region_id", "region_name", "province"]].drop_duplicates()
    rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []

    for election_id, election_raw in raw.groupby("election_id"):
        election_slots = slots.loc[slots["election_id"] == election_id].copy()
        if set(election_slots["slot"]) != set(SLOTS):
            raise ValueError(f"candidate_slots.csv must define A/B/C/alpha for {election_id}")

        named_slots = election_slots.loc[election_slots["slot"] != "alpha"]
        for region_id, region_raw in election_raw.groupby("region_id"):
            total_votes = float(region_raw["votes"].sum())
            matched_mask = pd.Series(False, index=region_raw.index)

            for _, slot_row in named_slots.iterrows():
                candidate_mask = (
                    (region_raw["candidate_name"] == slot_row["candidate_name"])
                    & (region_raw["party_name"] == slot_row["party_name"])
                )
                matched_votes = float(region_raw.loc[candidate_mask, "votes"].sum())
                matched_mask = matched_mask | candidate_mask
                active = bool(slot_row["is_active_slot"])
                votes = matched_votes if active else 0.0
                rows.append(
                    _result_row(
                        election_id,
                        region_id,
                        slot_row,
                        votes,
                        total_votes,
                    )
                )

            alpha_slot = election_slots.loc[election_slots["slot"] == "alpha"].iloc[0]
            alpha_votes = float(region_raw.loc[~matched_mask, "votes"].sum())
            rows.append(_result_row(election_id, region_id, alpha_slot, alpha_votes, total_votes))
            report_rows.append(
                {
                    "election_id": election_id,
                    "region_id": region_id,
                    "raw_candidates": int(len(region_raw)),
                    "matched_non_alpha_candidates": int(matched_mask.sum()),
                    "alpha_votes": alpha_votes,
                    "total_votes": total_votes,
                }
            )

    standardized = pd.DataFrame(rows).merge(region_lookup, on="region_id", how="left")
    standardized = standardized[
        [
            "election_id",
            "region_id",
            "region_name",
            "province",
            "slot",
            "candidate_name",
            "party_name",
            "is_active_slot",
            "votes",
            "vote_share",
        ]
    ].sort_values(["election_id", "region_id", "slot"])
    standardized["vote_share"] = standardized["vote_share"].fillna(0.0)
    validate_slot_coverage(standardized)
    validate_vote_share_sums(standardized)
    return standardized.reset_index(drop=True), pd.DataFrame(report_rows)


def _result_row(
    election_id: str,
    region_id: str,
    slot_row: pd.Series,
    votes: float,
    total_votes: float,
) -> dict[str, object]:
    return {
        "election_id": election_id,
        "region_id": region_id,
        "slot": slot_row["slot"],
        "candidate_name": slot_row["candidate_name"],
        "party_name": slot_row["party_name"],
        "is_active_slot": bool(slot_row["is_active_slot"]),
        "votes": votes,
        "vote_share": votes / total_votes if total_votes else 0.0,
    }

