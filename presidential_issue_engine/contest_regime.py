"""Outcome-blind contest-regime gate for presidential vote margins.

The gate decides whether a contest has enough forecast-time evidence to move
away from the central forecast.  It preserves a conservative core floor and
reallocates only the flexible vote between the structurally dominant candidate
and the runner-up.  Third-candidate shares are not directly changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_EXPANSION_GAIN = 0.50
DEFAULT_LOG_SHIFT_CAP = 0.40
# Preserve the historical public-call behavior. The active v10 policy passes
# its asymmetric critical/swing response explicitly so older experiments that
# invert the legacy transform remain reproducible.
DEFAULT_CRITICAL_ELASTICITY = 1.00
DEFAULT_SWING_ELASTICITY = 1.00
DEFAULT_SWING_LOG_SHIFT_CAP = 0.40
V10_CRITICAL_ELASTICITY = 0.75
V10_SWING_ELASTICITY = 1.25
V10_SWING_LOG_SHIFT_CAP = 0.50
MIN_RELIABILITY = 0.50
FULL_RELIABILITY = 0.80
MIN_DIRECTIONAL_ADVANTAGE = 0.02
DIRECTIONAL_ADVANTAGE_WIDTH = 0.08
MIN_SCORE_GAP = 0.015
SCORE_GAP_WIDTH = 0.030
PARTY_EROSION_WIDTH = 0.08
CONVERSION_BUFFER = 0.15
RUPTURE_SCORE_REFERENCE = 0.25


def conservative_core_floor(frame: pd.DataFrame) -> pd.Series:
    """Return a discounted lower floor, never the full estimated core mass."""

    effective = pd.to_numeric(
        frame.get("core_voting_mass_effective", 0.0), errors="coerce"
    )
    direct = pd.to_numeric(
        frame.get("direct_party_core_raw", 0.0), errors="coerce"
    )
    reliability = pd.to_numeric(
        frame.get("direct_party_reliability", 0.0), errors="coerce"
    )
    if not isinstance(effective, pd.Series):
        effective = pd.Series(float(effective), index=frame.index)
    if not isinstance(direct, pd.Series):
        direct = pd.Series(float(direct), index=frame.index)
    if not isinstance(reliability, pd.Series):
        reliability = pd.Series(float(reliability), index=frame.index)
    return (
        np.minimum(effective.fillna(0.0), direct.fillna(0.0))
        * reliability.fillna(0.0).clip(0.0, 1.0)
    ).clip(lower=0.0)


def derive_contest_regimes(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    slot_column: str = "source_slot",
    rejection_double_discount: bool = False,
) -> pd.DataFrame:
    """Derive one continuous regime activation per election without outcomes."""

    columns = [
        "election_id",
        "dominant_slot",
        "runner_up_slot",
        "contest_regime",
        "dominance_activation",
        "regime_base_activation",
        "regime_rejection_activation",
        "regime_certainty",
        "regime_reliability",
        "structural_score_gap",
        "directional_advantage",
        "cumulative_rejection_advantage",
        "dominant_cumulative_rejection",
        "runner_cumulative_rejection",
    ]
    required = {
        "election_id",
        slot_column,
        prediction_column,
        "direct_party_recent_base",
        "direct_party_reliability",
        "core_voting_mass_effective",
        "direct_party_core_raw",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"contest regime frame missing columns: {sorted(missing)}")

    working = frame.copy()
    working["_regime_core_floor"] = conservative_core_floor(working)
    optional = {
        "government_direction_score": 0.0,
        "direct_mega_score": 0.0,
        "incumbent_shock_log_shift": 0.0,
        "mega_issue_intensity_response": 1.0,
        "government_rejection_strength": 0.0,
    }
    for column, default in optional.items():
        if column not in working.columns:
            working[column] = default
        working[column] = pd.to_numeric(
            working[column], errors="coerce"
        ).fillna(default)

    summary = working.groupby(["election_id", slot_column], as_index=False).agg(
        party_base=("direct_party_recent_base", "mean"),
        core_floor=("_regime_core_floor", "mean"),
        preliminary_share=(prediction_column, "mean"),
        government_direction=("government_direction_score", "mean"),
        direct_mega=("direct_mega_score", "mean"),
        incumbent_shock=("incumbent_shock_log_shift", "mean"),
        reliability=("direct_party_reliability", "mean"),
        intensity=("mega_issue_intensity_response", "mean"),
        rejection_strength=("government_rejection_strength", "mean"),
    )
    summary["strongest_party_base"] = summary.groupby("election_id")[
        "party_base"
    ].transform("max")
    summary["party_erosion_gate"] = (
        (summary["strongest_party_base"] - summary["party_base"])
        / PARTY_EROSION_WIDTH
    ).clip(0.0, 1.0)
    summary["conversion_resistance"] = (
        (summary["preliminary_share"] - summary["party_base"])
        / CONVERSION_BUFFER
    ).clip(0.0, 1.0)
    summary["erosion_rejection_route"] = (
        summary["party_erosion_gate"] * (1.0 - summary["conversion_resistance"])
    )
    summary["rupture_rejection_route"] = (
        (summary["intensity"] - 1.0).clip(0.0, 1.0)
        * ((-summary["direct_mega"]) / RUPTURE_SCORE_REFERENCE).clip(0.0, 1.0)
    )
    summary["cumulative_rejection"] = (
        summary["rejection_strength"]
        * summary[["erosion_rejection_route", "rupture_rejection_route"]].max(axis=1)
        * summary["reliability"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    summary["structural_score"] = (
        0.45 * summary["party_base"]
        + 0.20 * summary["core_floor"]
        + 0.15 * summary["preliminary_share"]
        + 0.10 * summary["government_direction"]
        + 0.10 * summary["direct_mega"]
        - 0.10 * summary["cumulative_rejection"]
    )

    rows: list[dict[str, object]] = []
    for election_id, candidates in summary.groupby("election_id", sort=False):
        ranked = candidates.sort_values("structural_score", ascending=False).reset_index(
            drop=True
        )
        if len(ranked) < 2:
            continue
        dominant = ranked.iloc[0]
        runner = ranked.iloc[1]
        score_gap = float(dominant["structural_score"] - runner["structural_score"])
        directional_advantage = float(
            0.60 * (dominant["party_base"] - runner["party_base"])
            + 0.30 * (dominant["core_floor"] - runner["core_floor"])
            + (dominant["incumbent_shock"] - runner["incumbent_shock"])
            + 0.10
            * (dominant["preliminary_share"] - runner["preliminary_share"])
        )
        cumulative_rejection_advantage = float(
            runner["cumulative_rejection"] - dominant["cumulative_rejection"]
        )
        directional_advantage += cumulative_rejection_advantage
        reliability = float(
            0.50 * (dominant["reliability"] + runner["reliability"])
        )
        reliability_gate = float(
            np.clip(
                (reliability - MIN_RELIABILITY)
                / max(FULL_RELIABILITY - MIN_RELIABILITY, 1e-6),
                0.0,
                1.0,
            )
        )
        base_activation = reliability_gate * float(
            np.clip(
                (directional_advantage - MIN_DIRECTIONAL_ADVANTAGE)
                / DIRECTIONAL_ADVANTAGE_WIDTH,
                0.0,
                1.0,
            )
        )
        # Cumulative rejection already includes source reliability. Do not
        # discount the same evidence a second time through reliability_gate.
        rejection_activation = (
            0.0
            if reliability < MIN_RELIABILITY
            else float(
                np.clip(
                    cumulative_rejection_advantage / PARTY_EROSION_WIDTH,
                    0.0,
                    1.0,
                )
            )
        )
        if rejection_double_discount:
            rejection_activation = 0.0
        structural_certainty = float(
            np.clip((score_gap - MIN_SCORE_GAP) / SCORE_GAP_WIDTH, 0.0, 1.0)
        )
        rupture_certainty = float(
            np.clip(max(dominant["intensity"], runner["intensity"]) - 1.0, 0.0, 1.0)
        )
        certainty = max(structural_certainty, rupture_certainty)
        activation = float(
            np.clip(max(base_activation, rejection_activation) * certainty, 0.0, 1.0)
        )
        if activation <= 0.0:
            regime = "balanced_two_bloc"
        elif rupture_certainty > 0.0:
            regime = "rupture_landslide"
        elif cumulative_rejection_advantage > 0.0:
            regime = "cumulative_rejection_landslide"
        else:
            regime = "asymmetric_two_bloc"
        rows.append(
            {
                "election_id": str(election_id),
                "dominant_slot": str(dominant[slot_column]),
                "runner_up_slot": str(runner[slot_column]),
                "contest_regime": regime,
                "dominance_activation": activation,
                "regime_base_activation": base_activation,
                "regime_rejection_activation": rejection_activation,
                "regime_certainty": certainty,
                "regime_reliability": reliability,
                "structural_score_gap": score_gap,
                "directional_advantage": directional_advantage,
                "cumulative_rejection_advantage": cumulative_rejection_advantage,
                "dominant_cumulative_rejection": float(
                    dominant["cumulative_rejection"]
                ),
                "runner_cumulative_rejection": float(
                    runner["cumulative_rejection"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def apply_contest_regime_response(
    frame: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    prediction_column: str,
    slot_column: str = "source_slot",
    output_column: str | None = None,
    expansion_gain: float = DEFAULT_EXPANSION_GAIN,
    log_shift_cap: float = DEFAULT_LOG_SHIFT_CAP,
    critical_elasticity: float = DEFAULT_CRITICAL_ELASTICITY,
    swing_elasticity: float = DEFAULT_SWING_ELASTICITY,
    swing_log_shift_cap: float = DEFAULT_SWING_LOG_SHIFT_CAP,
) -> pd.DataFrame:
    """Expand dominant/runner critical and swing pools while preserving core."""

    output_column = output_column or prediction_column
    required = {"election_id", "region_id", slot_column, prediction_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"contest regime response missing columns: {sorted(missing)}")

    out = frame.copy().reset_index(drop=True)
    generated = [
        "contest_regime",
        "dominant_slot",
        "runner_up_slot",
        "dominance_activation",
        "regime_base_activation",
        "regime_rejection_activation",
        "regime_certainty",
        "regime_reliability",
        "structural_score_gap",
        "directional_advantage",
        "cumulative_rejection_advantage",
        "dominant_cumulative_rejection",
        "runner_cumulative_rejection",
        "regime_core_floor",
        "contest_regime_log_shift",
        "critical_regime_log_shift",
        "swing_regime_log_shift",
    ]
    out = out.drop(columns=[column for column in generated if column in out.columns])
    out["regime_core_floor"] = conservative_core_floor(out)
    out = out.merge(regimes, on="election_id", how="left")
    out["dominance_activation"] = pd.to_numeric(
        out["dominance_activation"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["contest_regime_log_shift"] = 0.0
    out["critical_regime_log_shift"] = 0.0
    out["swing_regime_log_shift"] = 0.0
    predictions = (
        pd.to_numeric(out[prediction_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        .copy()
    )
    gain = max(float(expansion_gain), 0.0)
    cap = abs(float(log_shift_cap))
    critical_elasticity = max(float(critical_elasticity), 0.0)
    swing_elasticity = max(float(swing_elasticity), 0.0)
    swing_cap = abs(float(swing_log_shift_cap))

    for _, group in out.groupby(["election_id", "region_id"], sort=False):
        activation = float(group["dominance_activation"].iloc[0])
        if activation <= 0.0 or gain <= 0.0:
            continue
        dominant_slot = str(group["dominant_slot"].iloc[0])
        runner_slot = str(group["runner_up_slot"].iloc[0])
        dominant_index = group.index[group[slot_column].astype(str).eq(dominant_slot)]
        runner_index = group.index[group[slot_column].astype(str).eq(runner_slot)]
        if len(dominant_index) != 1 or len(runner_index) != 1:
            continue
        dominant_index = int(dominant_index[0])
        runner_index = int(runner_index[0])
        pair_index = np.array([dominant_index, runner_index], dtype=int)
        pair_pred = predictions[pair_index].copy()
        pair_floor = out.loc[pair_index, "regime_core_floor"].to_numpy(float)
        pair_total = float(pair_pred.sum())
        floor_sum = float(pair_floor.sum())
        if floor_sum >= 0.95 * pair_total and floor_sum > 0.0:
            pair_floor *= 0.95 * pair_total / floor_sum
        flexible = np.clip(pair_pred - pair_floor, 0.0, None)
        shift = float(np.clip(gain * activation, 0.0, cap))
        critical_available = pd.to_numeric(
            out.loc[pair_index].get(
                "critical_voting_mass_effective",
                pd.Series(0.0, index=pair_index),
            ),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        critical_pool = np.minimum(critical_available, flexible)
        swing_pool = np.clip(flexible - critical_pool, 0.0, None)
        critical_shift = float(np.clip(shift * critical_elasticity, 0.0, cap))
        swing_shift = float(
            np.clip(shift * swing_elasticity, 0.0, swing_cap)
        )

        def move_pool(pool: np.ndarray, pool_shift: float) -> np.ndarray:
            total = float(pool.sum())
            if total <= 1e-12 or pool_shift <= 0.0:
                return pool
            moved_pool = pool * np.exp(np.array([pool_shift, -pool_shift]))
            return moved_pool * total / max(float(moved_pool.sum()), 1e-12)

        moved_critical = move_pool(critical_pool, critical_shift)
        moved_swing = move_pool(swing_pool, swing_shift)
        predictions[pair_index] = pair_floor + moved_critical + moved_swing
        out.loc[dominant_index, "contest_regime_log_shift"] = shift
        out.loc[runner_index, "contest_regime_log_shift"] = -shift
        out.loc[dominant_index, "critical_regime_log_shift"] = critical_shift
        out.loc[runner_index, "critical_regime_log_shift"] = -critical_shift
        out.loc[dominant_index, "swing_regime_log_shift"] = swing_shift
        out.loc[runner_index, "swing_regime_log_shift"] = -swing_shift

    out[output_column] = predictions
    return out
