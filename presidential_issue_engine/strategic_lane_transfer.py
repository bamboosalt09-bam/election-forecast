"""Outcome-blind tactical transfer within broad ideological lanes.

Minor-party support is not candidate concrete support. Its stable attachment
remains a lane reservoir that can move to an aligned major-party candidate
when the minor candidate is weak and wasted-vote pressure is high.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from presidential_issue_engine import issue_vote_engine as engine
from presidential_issue_engine.point_in_time import filter_available_by_election


DEFAULT_AFFINITY_POWER = 2.0


def _same_lane_affinity(left: pd.Series, right: pd.Series) -> float:
    """Return affinity only when candidates are in compatible broad camps."""

    left_label = engine._orientation_label(left)
    right_label = engine._orientation_label(right)
    conservative = {"conservative", "conservative_centrist"}
    liberal = {"liberal", "liberal_centrist"}
    if (left_label in conservative and right_label in liberal) or (
        left_label in liberal and right_label in conservative
    ):
        return 0.0
    return engine._orientation_affinity(left, right)


def attach_conversion_context(
    frame: pd.DataFrame,
    context: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> pd.DataFrame:
    """Attach PIT-safe conversion context by candidate, independent of slot labels."""

    out = frame.copy()
    required = {
        "election_id",
        "candidate_name",
        "wasted_vote_resistance",
        "major_party_gravity",
        "available_date",
        "confidence",
    }
    if context.empty or not required.issubset(context.columns):
        for column in (
            "wasted_vote_resistance",
            "major_party_gravity",
            "strategic_transfer_confidence",
        ):
            out[column] = 0.0
        return out

    eligible = filter_available_by_election(
        context.copy(),
        election_dates,
        source_name="strategic_lane_transfer_context",
    )
    eligible = eligible.sort_values("available_date").drop_duplicates(
        ["election_id", "candidate_name"], keep="last"
    )
    eligible = eligible[
        [
            "election_id",
            "candidate_name",
            "wasted_vote_resistance",
            "major_party_gravity",
            "confidence",
        ]
    ].rename(columns={"confidence": "strategic_transfer_confidence"})
    name_column = (
        "candidate_name_x" if "candidate_name_x" in out.columns else "candidate_name"
    )
    out["_strategic_candidate_name"] = out[name_column].astype(str)
    out = out.merge(
        eligible.rename(columns={"candidate_name": "_strategic_candidate_name"}),
        on=["election_id", "_strategic_candidate_name"],
        how="left",
        validate="many_to_one",
    ).drop(columns="_strategic_candidate_name")
    for column in (
        "wasted_vote_resistance",
        "major_party_gravity",
        "strategic_transfer_confidence",
    ):
        out[column] = (
            pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
    return out


def apply_strategic_lane_transfer(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "layer_pred",
    affinity_power: float = DEFAULT_AFFINITY_POWER,
) -> pd.DataFrame:
    """Move bounded minor-candidate mass to aligned major-party candidates.

    The transferable reservoir is the donor's effective critical-support mass,
    never its concrete mass. Pressure rises continuously as the donor's
    preliminary strength falls relative to the strongest major-party candidate.
    """

    out = frame.copy()
    for column in (
        "strategic_lane_reservoir",
        "strategic_lane_pressure",
        "strategic_lane_transfer_out",
        "strategic_lane_transfer_in",
        "strategic_lane_transfer_net",
    ):
        out[column] = 0.0
    required = {
        "election_id",
        "region_id",
        "major_party_core_eligible",
        "critical_voting_mass_effective",
        "preliminary_mean_share",
        "wasted_vote_resistance",
        "major_party_gravity",
        "strategic_transfer_confidence",
        prediction_column,
        *{f"landscape_axis_{axis}" for axis in engine.LANDSCAPE_VECTOR_COLUMNS},
    }
    if out.empty or not required.issubset(out.columns):
        return out

    prediction = pd.to_numeric(out[prediction_column], errors="coerce").fillna(0.0)
    adjusted = prediction.to_numpy(float).copy()
    position = pd.Series(np.arange(len(out)), index=out.index)
    eligible_major = out["major_party_core_eligible"].fillna(False).astype(bool)
    affinity_power = max(float(affinity_power), 1.0)

    for _, region in out.groupby(["election_id", "region_id"], sort=False):
        major = region.loc[eligible_major.loc[region.index]]
        donors = region.loc[~eligible_major.loc[region.index]]
        if major.empty or donors.empty:
            continue
        major_preliminary = pd.to_numeric(
            major["preliminary_mean_share"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        strongest_major = float(major_preliminary.max())
        if strongest_major <= 0.0:
            continue

        for donor_index, donor in donors.iterrows():
            reservoir = float(
                np.clip(
                    pd.to_numeric(
                        pd.Series([donor["critical_voting_mass_effective"]]),
                        errors="coerce",
                    ).fillna(0.0).iloc[0],
                    0.0,
                    1.0,
                )
            )
            donor_preliminary = float(
                np.clip(
                    pd.to_numeric(
                        pd.Series([donor["preliminary_mean_share"]]), errors="coerce"
                    ).fillna(0.0).iloc[0],
                    0.0,
                    1.0,
                )
            )
            relative_viability = float(
                np.clip(donor_preliminary / strongest_major, 0.0, 1.0)
            )
            resistance = float(np.clip(donor["wasted_vote_resistance"], 0.0, 1.0))
            gravity = float(np.clip(donor["major_party_gravity"], 0.0, 1.0))
            confidence = float(
                np.clip(donor["strategic_transfer_confidence"], 0.0, 1.0)
            )
            affinities = major.apply(
                lambda recipient: _same_lane_affinity(donor, recipient), axis=1
            ).clip(lower=0.0, upper=1.0)
            if float(affinities.max()) <= 0.0:
                continue
            lane_clarity = float(affinities.max())
            pressure = (
                gravity
                * (1.0 - resistance)
                * (1.0 - relative_viability)
                * confidence
                * lane_clarity
            )
            donor_position = int(position.loc[donor_index])
            transfer = min(reservoir * pressure, float(max(adjusted[donor_position], 0.0)))
            if transfer <= 0.0:
                continue

            weights = affinities.pow(affinity_power)
            weights = weights / float(weights.sum())
            adjusted[donor_position] -= transfer
            out.at[donor_index, "strategic_lane_reservoir"] = reservoir
            out.at[donor_index, "strategic_lane_pressure"] = pressure
            out.at[donor_index, "strategic_lane_transfer_out"] += transfer
            for recipient_index, weight in weights.items():
                recipient_transfer = transfer * float(weight)
                adjusted[int(position.loc[recipient_index])] += recipient_transfer
                out.at[recipient_index, "strategic_lane_transfer_in"] += recipient_transfer

    out["strategic_lane_transfer_net"] = (
        out["strategic_lane_transfer_in"] - out["strategic_lane_transfer_out"]
    )
    out[prediction_column] = adjusted
    return out
