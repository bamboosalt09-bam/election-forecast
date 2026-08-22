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
# How a constitutional rupture erodes the burdened candidate's core floor.
#
# ``proportional`` multiplies each region's floor by the same election-level
# rate, so the absolute mass released scales with core depth: in 2017 the rate
# is 0.866 nationwide, 경북 releases 0.418 and 강원 0.289. The region with the
# deepest core gives up the most vote, which inverts what a core is for.
#
# ``absolute`` lets the shock set one nationwide erosion in vote terms and lets
# each region resist it out of its own depth. Shock magnitude then sizes the
# push and regional core depth decides how much actually moves.
# ``layered`` draws the transfer from the electorate layers in order rather
# than from one undifferentiated pool. contest_regime already separates them -
# critical support responds at 0.75 and swing at 1.25 - while the veto lumps
# concrete, critical and swing together above an eroded floor and applies one
# rate to all of it. In 대구 2017 that let it take 0.0527 from a candidate whose
# non-core mass was 0.0083, pulling the prediction to 0.4274 beneath his own
# concrete core of 0.4584.
FLOOR_EROSION_MODES = {"proportional", "absolute", "layered"}
CRITICAL_ELASTICITY = 0.75
SWING_ELASTICITY = 1.25
# How much of the rupture-eroded core the veto may actually draw on, in
# ``layered`` mode. 1.0 reproduces the undifferentiated behaviour; 0.0
# protects concrete support outright. Both endpoints are measured.
DEFAULT_CORE_EROSION_RESISTANCE = 1.0
DEFAULT_FLOOR_EROSION_MODE = "proportional"
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
    floor_erosion_mode: str = DEFAULT_FLOOR_EROSION_MODE,
    core_erosion_resistance: float = DEFAULT_CORE_EROSION_RESISTANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a bounded incumbent-veto tail and return a per-region audit."""

    if floor_erosion_mode not in FLOOR_EROSION_MODES:
        raise ValueError(f"unknown floor erosion mode: {floor_erosion_mode}")
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

        # One erosion in vote terms for the whole election, taken from the
        # typical regional core rather than from each region's own depth.
        reference_floor = float(
            np.clip(
                pd.to_numeric(
                    election.get("regime_core_floor", pd.Series(0.0, index=election.index)),
                    errors="coerce",
                ).fillna(0.0).mean(),
                0.0,
                1.0,
            )
        )
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
            if floor_erosion_mode == "absolute":
                eroded_floor = base_runner_floor - rupture_floor_activation * reference_floor
            else:
                eroded_floor = base_runner_floor * (1.0 - rupture_floor_activation)
            effective_floor = min(
                base_runner_floor,
                max(theoretical_floor, eroded_floor),
            )
            out.at[runner_index, "strong_veto_effective_floor"] = effective_floor
            if floor_erosion_mode == "layered":
                critical_mass = float(
                    np.clip(
                        pd.to_numeric(
                            out.at[runner_index, "critical_support_raw"], errors="coerce"
                        ),
                        0.0,
                        None,
                    )
                    if "critical_support_raw" in out.columns
                    else 0.0
                )
                above_core = max(runner_prediction - base_runner_floor, 0.0)
                critical_available = min(critical_mass, above_core)
                swing_available = max(above_core - critical_available, 0.0)
                # The core yields only what the rupture eroded, and the two
                # mobile layers respond at the elasticities contest_regime
                # already uses. Protecting the core outright was measured and
                # is worse: the veto's total transfer is load-bearing for the
                # national levels, so removing most of it breaks them.
                core_available = max(base_runner_floor - effective_floor, 0.0) * float(
                    np.clip(core_erosion_resistance, 0.0, 1.0)
                )
                flexible = (
                    swing_available * SWING_ELASTICITY
                    + critical_available * CRITICAL_ELASTICITY
                    + core_available
                )
            else:
                flexible = max(runner_prediction - effective_floor, 0.0)
            transfer = min(rate * flexible, max(runner_prediction - theoretical_floor, 0.0))
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
