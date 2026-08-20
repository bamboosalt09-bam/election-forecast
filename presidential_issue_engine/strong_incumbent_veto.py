"""Outcome-blind tail response for an already decisive incumbent-party defeat.

The ordinary contest-regime layer handles continuous asymmetry.  This layer is
deliberately narrower: it activates only when the model itself already projects
the government-burdened major candidate to trail the structurally dominant
challenger by at least ten percentage points.  It then moves a small part of
the burdened candidate's non-core vote to the challenger.

No election result, realised margin, polling value, or post-election field is
required.  The projected margin is the equal-region mean of the model forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PROJECTED_MARGIN_THRESHOLD = 0.10
# Declared hypothesis constants. Historical evaluation records sensitivity but
# does not select these values.
DEFAULT_GAIN = 1.00
DEFAULT_THEORETICAL_FLOOR = 0.01
DEFAULT_RUPTURE_FLOOR_EROSION_ENABLED = True
RUPTURE_SCORE_REFERENCE = 0.25
MAJOR_SLOTS = {"A", "B"}


def apply_strong_incumbent_veto(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "layer_pred",
    slot_column: str = "source_slot",
    projected_margin_threshold: float = DEFAULT_PROJECTED_MARGIN_THRESHOLD,
    gain: float = DEFAULT_GAIN,
    rupture_floor_erosion_enabled: bool = DEFAULT_RUPTURE_FLOOR_EROSION_ENABLED,
    theoretical_floor: float = DEFAULT_THEORETICAL_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a bounded incumbent-veto tail and return a per-region audit."""

    required = {
        "election_id",
        "region_id",
        slot_column,
        prediction_column,
        "government_direction_score",
        "government_rejection_strength",
        "dominant_slot",
        "runner_up_slot",
        "dominance_activation",
        "regime_certainty",
        "regime_core_floor",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"strong incumbent veto frame missing columns: {sorted(missing)}")

    out = frame.copy()
    generated = (
        "strong_veto_transfer_in",
        "strong_veto_transfer_out",
        "strong_veto_transfer_net",
        "strong_veto_rate",
        "strong_veto_projected_margin",
        "strong_veto_rupture_floor_activation",
        "strong_veto_effective_floor",
    )
    for column in generated:
        out[column] = 0.0

    threshold = max(float(projected_margin_threshold), 0.0)
    gain = max(float(gain), 0.0)
    theoretical_floor = float(np.clip(theoretical_floor, 0.0, 1.0))
    audit_rows: list[dict[str, object]] = []

    for election_id, election in out.groupby("election_id", sort=False):
        majors = election.loc[election[slot_column].astype(str).isin(MAJOR_SLOTS)]
        if majors.empty:
            continue

        dominant_slot = str(election["dominant_slot"].iloc[0])
        runner_slot = str(election["runner_up_slot"].iloc[0])
        if dominant_slot not in MAJOR_SLOTS or runner_slot not in MAJOR_SLOTS:
            continue

        mean_prediction = (
            majors.groupby(slot_column)[prediction_column]
            .mean()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        projected_margin = float(
            mean_prediction.get(dominant_slot, 0.0)
            - mean_prediction.get(runner_slot, 0.0)
        )
        direction = (
            majors.groupby(slot_column)["government_direction_score"]
            .mean()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        runner_is_burdened = float(direction.get(runner_slot, 0.0)) < 0.0
        activation = float(
            np.clip(
                pd.to_numeric(
                    pd.Series([election["dominance_activation"].iloc[0]]),
                    errors="coerce",
                ).fillna(0.0).iloc[0],
                0.0,
                1.0,
            )
        )
        certainty = float(
            np.clip(
                pd.to_numeric(
                    pd.Series([election["regime_certainty"].iloc[0]]),
                    errors="coerce",
                ).fillna(0.0).iloc[0],
                0.0,
                1.0,
            )
        )
        runner_rows = majors.loc[majors[slot_column].astype(str).eq(runner_slot)]
        rejection = float(
            pd.to_numeric(
                runner_rows["government_rejection_strength"], errors="coerce"
            ).fillna(0.0).mean()
        )
        intensity = float(
            pd.to_numeric(
                runner_rows.get(
                    "mega_issue_intensity_response",
                    pd.Series(1.0, index=runner_rows.index),
                ),
                errors="coerce",
            ).fillna(1.0).mean()
        )
        direct_mega = float(
            pd.to_numeric(
                runner_rows.get(
                    "direct_mega_score",
                    pd.Series(0.0, index=runner_rows.index),
                ),
                errors="coerce",
            ).fillna(0.0).mean()
        )
        negative_share = float(
            np.clip(
                pd.to_numeric(
                    runner_rows.get(
                        "government_negative_share",
                        pd.Series(0.0, index=runner_rows.index),
                    ),
                    errors="coerce",
                ).fillna(0.0).mean(),
                0.0,
                1.0,
            )
        )
        rejection_breadth = float(
            np.clip(
                pd.to_numeric(
                    runner_rows.get(
                        "government_rejection_breadth",
                        pd.Series(0.0, index=runner_rows.index),
                    ),
                    errors="coerce",
                ).fillna(0.0).mean(),
                0.0,
                1.0,
            )
        )
        rupture_floor_activation = 0.0
        if rupture_floor_erosion_enabled:
            rupture_floor_activation = float(
                np.clip(intensity - 1.0, 0.0, 1.0)
                * np.clip(-direct_mega / RUPTURE_SCORE_REFERENCE, 0.0, 1.0)
                * negative_share
                * np.sqrt(rejection_breadth)
            )
        gate = (
            runner_is_burdened
            and projected_margin >= threshold
            and activation > 0.0
            and certainty > 0.0
            and rejection > 0.0
        )
        rate = float(np.clip(gain * rejection * activation * certainty, 0.0, 1.0)) if gate else 0.0
        out.loc[election.index, "strong_veto_rate"] = rate
        out.loc[election.index, "strong_veto_projected_margin"] = projected_margin
        out.loc[
            election.index, "strong_veto_rupture_floor_activation"
        ] = rupture_floor_activation
        if rate <= 0.0:
            continue

        for region_id, region in election.groupby("region_id", sort=False):
            dominant = region.index[region[slot_column].astype(str).eq(dominant_slot)]
            runner = region.index[region[slot_column].astype(str).eq(runner_slot)]
            if len(dominant) != 1 or len(runner) != 1:
                continue
            dominant_index = int(dominant[0])
            runner_index = int(runner[0])
            runner_prediction = float(out.at[runner_index, prediction_column])
            base_runner_floor = float(
                np.clip(out.at[runner_index, "regime_core_floor"], 0.0, 1.0)
            )
            eroded_floor = base_runner_floor * (1.0 - rupture_floor_activation)
            effective_floor = min(
                base_runner_floor,
                max(theoretical_floor, eroded_floor),
            )
            out.at[runner_index, "strong_veto_effective_floor"] = effective_floor
            flexible = max(runner_prediction - effective_floor, 0.0)
            transfer = min(rate * flexible, runner_prediction)
            if transfer <= 0.0:
                continue
            out.at[runner_index, prediction_column] -= transfer
            out.at[dominant_index, prediction_column] += transfer
            out.at[runner_index, "strong_veto_transfer_out"] = transfer
            out.at[dominant_index, "strong_veto_transfer_in"] = transfer
            audit_rows.append(
                {
                    "election_id": str(election_id),
                    "region_id": str(region_id),
                    "beneficiary_slot": dominant_slot,
                    "burdened_slot": runner_slot,
                    "projected_margin": projected_margin,
                    "government_rejection_strength": rejection,
                    "dominance_activation": activation,
                    "regime_certainty": certainty,
                    "veto_rate": rate,
                    "base_runner_core_floor": base_runner_floor,
                    "rupture_floor_activation": rupture_floor_activation,
                    "theoretical_floor": theoretical_floor,
                    "effective_runner_floor": effective_floor,
                    "runner_flexible_mass": flexible,
                    "transfer": transfer,
                }
            )

    out["strong_veto_transfer_net"] = (
        out["strong_veto_transfer_in"] - out["strong_veto_transfer_out"]
    )
    return out, pd.DataFrame(audit_rows)
