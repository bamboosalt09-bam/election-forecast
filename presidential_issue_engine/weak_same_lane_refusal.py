"""Hypothesis layer for wasted-vote refusal around a weak same-lane candidate.

The layer is intentionally narrow and outcome-blind.  It applies only to a
slot-C candidate whose pre-election lineage lacks a major-party split and moves
a declared fraction of its non-protected forecast mass to aligned major
candidates. The protected floor is an explicit hypothesis mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine import strategic_lane_transfer
from presidential_issue_engine import (
    third_candidate_lineage_constraint as lineage_constraint,
)


DEFAULT_GAIN = 0.50
DEFAULT_AFFINITY_POWER = 2.0
DEFAULT_FLOOR_MODE = "theoretical"
DEFAULT_THEORETICAL_FLOOR = 0.01
FLOOR_MODES = {"candidate_ballot", "theoretical", "none"}
DEFAULT_RECIPIENT_WEIGHT_MODE = "prediction_tilted"
RECIPIENT_WEIGHT_MODES = {"affinity_only", "prediction_tilted"}


def lineage_scaled_gain(
    election_id: str,
    lineage: pd.DataFrame,
    *,
    base_gain: float = DEFAULT_GAIN,
    exemption_scale: float | None = None,
) -> float:
    """Reduce the refusal rate as major-party carry-in rises.

    A vehicle that absorbed sitting members of a governing or main opposition
    party carries a weaker wasted-vote argument than one founded without any
    such transfer, so the desertion rate should not be identical for both.

    The rate falls linearly from ``base_gain`` at zero carry-in to zero at the
    exemption threshold used by ``third_candidate_lineage_constraint``, which
    makes the rate continuous where the weak classification itself ends. No new
    constant is introduced: both anchors already exist in the pipeline.

    Rows without a derived carry-in share - independents and the two warmup-only
    vehicles whose founding rosters are unsourced - keep ``base_gain``.
    """

    if exemption_scale is None:
        exemption_scale = lineage_constraint.DEFAULT_DEFECTION_FLOOR
    if not (exemption_scale > 0.0):
        return float(base_gain)
    rows = lineage.loc[lineage["election_id"].astype(str).eq(str(election_id))]
    if rows.empty:
        return float(base_gain)
    row = rows.iloc[0]
    seats = pd.to_numeric(pd.Series([row.get("defection_seats")]), errors="coerce").iloc[0]
    size = pd.to_numeric(pd.Series([row.get("assembly_size")]), errors="coerce").iloc[0]
    has_party = str(row.get("has_party", True)).strip().lower() in {"1", "true", "yes", "y"}
    if not has_party or pd.isna(seats) or pd.isna(size) or size <= 0:
        return float(base_gain)
    share = float(seats) / float(size)
    return float(np.clip(base_gain * (1.0 - share / exemption_scale), 0.0, base_gain))


def apply_weak_same_lane_refusal(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "layer_pred",
    slot_column: str = "source_slot",
    gain: float = DEFAULT_GAIN,
    affinity_power: float = DEFAULT_AFFINITY_POWER,
    floor_mode: str = DEFAULT_FLOOR_MODE,
    theoretical_floor: float = DEFAULT_THEORETICAL_FLOOR,
    recipient_weight_mode: str = DEFAULT_RECIPIENT_WEIGHT_MODE,
    lineage: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Move a bounded weak-candidate excess to same-lane major candidates."""

    required = {
        "election_id",
        "region_id",
        slot_column,
        prediction_column,
        "candidate_ballot_recent_base",
        *{
            f"landscape_axis_{axis}"
            for axis in strategic_lane_transfer.engine.LANDSCAPE_VECTOR_COLUMNS
        },
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"weak same-lane refusal frame missing columns: {sorted(missing)}")

    declared_lineage = (
        lineage_constraint.load_lineage() if lineage is None else lineage.copy()
    )
    if "origin_lane" not in declared_lineage.columns:
        declared_lineage["origin_lane"] = ""
    declared_lineage["origin_lane"] = (
        declared_lineage["origin_lane"].fillna("").astype(str).str.strip()
    )
    origin_lane_by_election = (
        declared_lineage.sort_values("available_date")
        .drop_duplicates("election_id", keep="last")
        .set_index("election_id")["origin_lane"]
        .to_dict()
    )
    weak_elections = lineage_constraint.self_founded_elections(declared_lineage)
    out = frame.copy()
    for column in (
        "weak_lane_refusal_transfer_in",
        "weak_lane_refusal_transfer_out",
        "weak_lane_refusal_transfer_net",
        "weak_lane_refusal_reservoir",
        "weak_lane_refusal_rate",
        "weak_lane_refusal_floor",
    ):
        out[column] = 0.0

    gain = float(np.clip(gain, 0.0, 1.0))
    affinity_power = max(float(affinity_power), 1.0)
    floor_mode = str(floor_mode).strip().lower()
    if floor_mode not in FLOOR_MODES:
        raise ValueError(f"unknown weak same-lane floor mode: {floor_mode}")
    theoretical_floor = float(np.clip(theoretical_floor, 0.0, 1.0))
    recipient_weight_mode = str(recipient_weight_mode).strip().lower()
    if recipient_weight_mode not in RECIPIENT_WEIGHT_MODES:
        raise ValueError(
            f"unknown weak same-lane recipient mode: {recipient_weight_mode}"
        )
    audit_rows: list[dict[str, object]] = []
    if gain <= 0.0:
        return out, pd.DataFrame(audit_rows)

    for election_id, election in out.groupby("election_id", sort=False):
        election_gain = lineage_scaled_gain(
            str(election_id), declared_lineage, base_gain=gain
        )
        if election_gain <= 0.0:
            continue
        if str(election_id) not in weak_elections:
            continue
        for region_id, region in election.groupby("region_id", sort=False):
            donor_rows = region.loc[region[slot_column].astype(str).eq("C")]
            major_rows = region.loc[region[slot_column].astype(str).isin(["A", "B"])]
            if len(donor_rows) != 1 or major_rows.empty:
                continue
            donor_index = donor_rows.index[0]
            donor = donor_rows.iloc[0]
            donor_for_affinity = donor.copy()
            declared_origin_lane = str(
                origin_lane_by_election.get(str(election_id), "")
            )
            if declared_origin_lane:
                donor_for_affinity["declared_origin_lane"] = declared_origin_lane
            prediction = float(out.at[donor_index, prediction_column])
            ballot_base = float(
                np.clip(
                    pd.to_numeric(
                        pd.Series([donor["candidate_ballot_recent_base"]]),
                        errors="coerce",
                    ).fillna(0.0).iloc[0],
                    0.0,
                    1.0,
                )
            )
            if floor_mode == "candidate_ballot":
                protected_floor = ballot_base
            elif floor_mode == "theoretical":
                protected_floor = min(prediction, theoretical_floor)
            else:
                protected_floor = 0.0
            reservoir = max(prediction - protected_floor, 0.0)
            if reservoir <= 0.0:
                continue

            affinities = major_rows.apply(
                lambda recipient: strategic_lane_transfer._same_lane_affinity(
                    donor_for_affinity, recipient
                ),
                axis=1,
            ).clip(0.0, 1.0)
            if recipient_weight_mode == "affinity_only":
                weights = affinities.pow(affinity_power)
            else:
                major_prior = pd.to_numeric(
                    major_rows[prediction_column], errors="coerce"
                ).fillna(0.0).clip(lower=0.0)
                weights = major_prior * (1.0 + affinities).pow(affinity_power)
            if float(weights.sum()) <= 0.0:
                continue
            weights /= float(weights.sum())
            transfer = min(election_gain * reservoir, prediction)
            if transfer <= 0.0:
                continue

            out.at[donor_index, prediction_column] -= transfer
            out.at[donor_index, "weak_lane_refusal_transfer_out"] = transfer
            out.at[donor_index, "weak_lane_refusal_reservoir"] = reservoir
            out.at[donor_index, "weak_lane_refusal_rate"] = election_gain
            out.at[donor_index, "weak_lane_refusal_floor"] = protected_floor
            recipient_slots: list[str] = []
            for recipient_index, weight in weights.items():
                recipient_transfer = transfer * float(weight)
                out.at[recipient_index, prediction_column] += recipient_transfer
                out.at[
                    recipient_index, "weak_lane_refusal_transfer_in"
                ] += recipient_transfer
                if recipient_transfer > 0.0:
                    recipient_slots.append(str(out.at[recipient_index, slot_column]))
            audit_rows.append(
                {
                    "election_id": str(election_id),
                    "region_id": str(region_id),
                    "donor_slot": "C",
                    "recipient_slots": "|".join(recipient_slots),
                    "candidate_ballot_recent_base": ballot_base,
                    "floor_mode": floor_mode,
                    "recipient_weight_mode": recipient_weight_mode,
                    "declared_origin_lane": declared_origin_lane,
                    "resolved_origin_lane": strategic_lane_transfer.engine._orientation_label(
                        donor_for_affinity
                    ),
                    "protected_floor": protected_floor,
                    "before": prediction,
                    "reservoir": reservoir,
                    "gain": election_gain,
                    "transfer": transfer,
                    "after": prediction - transfer,
                }
            )

    out["weak_lane_refusal_transfer_net"] = (
        out["weak_lane_refusal_transfer_in"]
        - out["weak_lane_refusal_transfer_out"]
    )
    return out, pd.DataFrame(audit_rows)
