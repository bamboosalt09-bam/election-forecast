"""Outcome-safe V24 regional-shape and predictive-interval experiments.

This module is deliberately downstream of the frozen V23 point forecasts.  It
never reads a target election outcome while constructing a regional correction
or its predictive interval.  Target outcomes are supplied only to evaluation
code outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EPSILON = 1e-12


def normalize_region_weights(weights: pd.Series, regions: pd.Series) -> np.ndarray:
    """Align positive forecast-time region weights to a frame and normalize."""

    aligned = regions.astype(str).map(weights.astype(float))
    positive = pd.to_numeric(aligned, errors="coerce")
    fallback = float(positive[positive.gt(0.0)].median()) if positive.gt(0.0).any() else 1.0
    values = positive.fillna(fallback).clip(lower=EPSILON).to_numpy(float)
    return values / values.sum()


def prior_region_weights(
    target: pd.DataFrame,
    prior: pd.DataFrame | None,
) -> tuple[pd.Series, str]:
    """Build region weights from the latest strictly prior election.

    The first scored election has no prior regional vote-volume artifact in the
    frozen panel, so it receives an explicit equal-region fallback.
    """

    regions = pd.Index(target["region_id"].astype(str).drop_duplicates())
    if prior is None or prior.empty:
        return pd.Series(1.0 / len(regions), index=regions), "equal_region_no_prior"
    volume = (
        prior.assign(region_id=prior["region_id"].astype(str))
        .groupby("region_id", sort=False)["contest_votes"]
        .first()
        .astype(float)
    )
    aligned = volume.reindex(regions)
    fallback = float(aligned[aligned.gt(0.0)].median()) if aligned.gt(0.0).any() else 1.0
    aligned = aligned.fillna(fallback).clip(lower=EPSILON)
    return aligned / aligned.sum(), "latest_prior_presidential_contest_votes"


def _rake_matrix(
    seed: np.ndarray,
    row_targets: np.ndarray,
    column_targets: np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> np.ndarray:
    """Rake a strictly positive matrix to fixed row and column margins."""

    matrix = np.maximum(np.asarray(seed, dtype=float), EPSILON)
    row_targets = np.asarray(row_targets, dtype=float)
    column_targets = np.asarray(column_targets, dtype=float)
    for _ in range(max_iterations):
        matrix *= (row_targets / matrix.sum(axis=1))[:, None]
        matrix *= (column_targets / matrix.sum(axis=0))[None, :]
        row_error = float(np.max(np.abs(matrix.sum(axis=1) - row_targets)))
        column_error = float(np.max(np.abs(matrix.sum(axis=0) - column_targets)))
        if max(row_error, column_error) <= tolerance:
            break
    else:  # pragma: no cover - defensive guard for malformed structural zeros
        raise RuntimeError("regional shape raking did not converge")
    matrix *= (row_targets / matrix.sum(axis=1))[:, None]
    return matrix


def apply_national_preserving_regional_shape(
    frame: pd.DataFrame,
    region_weights: pd.Series,
    *,
    gain: float,
    prediction_column: str = "pred",
) -> pd.DataFrame:
    """Tilt regional candidate shares while preserving forecast national totals."""

    required = {
        "election_id",
        "region_id",
        "slot",
        prediction_column,
        "regional_accent_signal_scaled",
        "regional_accent_reliability",
        "core_voting_mass_effective",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"regional shape frame missing columns: {sorted(missing)}")
    if frame["election_id"].nunique() != 1:
        raise ValueError("regional shape must be applied one election at a time")

    out = frame.copy().reset_index(drop=True)
    region_order = out["region_id"].astype(str).drop_duplicates().tolist()
    slot_order = out["slot"].astype(str).drop_duplicates().tolist()
    expected = pd.MultiIndex.from_product([region_order, slot_order])
    observed = pd.MultiIndex.from_frame(out[["region_id", "slot"]].astype(str))
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise ValueError("regional shape requires a complete region-by-slot panel")

    indexed = out.assign(
        region_id=out["region_id"].astype(str),
        slot=out["slot"].astype(str),
    ).set_index(["region_id", "slot"])
    base = indexed[prediction_column].unstack("slot").reindex(
        index=region_order,
        columns=slot_order,
    ).to_numpy(float, copy=True)
    base /= base.sum(axis=1, keepdims=True)

    reliability = indexed["regional_accent_reliability"].unstack("slot").reindex(
        index=region_order,
        columns=slot_order,
    ).fillna(0.0).to_numpy(float)
    core = indexed["core_voting_mass_effective"].unstack("slot").reindex(
        index=region_order,
        columns=slot_order,
    ).fillna(0.0).to_numpy(float)
    signal = indexed["regional_accent_signal_scaled"].unstack("slot").reindex(
        index=region_order,
        columns=slot_order,
    ).fillna(0.0).to_numpy(float)
    mobility = np.clip(1.0 - core, 0.0, 1.0)
    score = signal * np.clip(reliability, 0.0, 1.0) * mobility

    weights = normalize_region_weights(region_weights, pd.Series(region_order))
    baseline_mass = weights[:, None] * base
    national_targets = baseline_mass.sum(axis=0)
    if abs(float(gain)) <= EPSILON:
        adjusted = base.copy()
    else:
        tilted = base * np.exp(np.clip(float(gain) * score, -20.0, 20.0))
        tilted /= tilted.sum(axis=1, keepdims=True)
        adjusted_mass = _rake_matrix(
            weights[:, None] * tilted,
            weights,
            national_targets,
        )
        adjusted = adjusted_mass / weights[:, None]

    adjusted_long = pd.DataFrame(adjusted, index=region_order, columns=slot_order).stack()
    score_long = pd.DataFrame(score, index=region_order, columns=slot_order).stack()
    keys = list(zip(out["region_id"].astype(str), out["slot"].astype(str)))
    out["v24_regional_shape_score"] = [float(score_long.loc[key]) for key in keys]
    out["v24_regional_shape_gain"] = float(gain)
    out["v24_regional_shape_pred"] = [float(adjusted_long.loc[key]) for key in keys]
    out["v24_regional_shape_delta"] = (
        out["v24_regional_shape_pred"] - out[prediction_column].astype(float)
    )
    out["v24_forecast_region_weight"] = out["region_id"].astype(str).map(
        pd.Series(weights, index=region_order)
    )

    candidate_after = (weights[:, None] * adjusted).sum(axis=0)
    drift = float(np.max(np.abs(candidate_after - national_targets)))
    row_drift = float(np.max(np.abs(adjusted.sum(axis=1) - 1.0)))
    if drift > 1e-10 or row_drift > 1e-10:
        raise RuntimeError("regional shape failed its mass-preservation invariant")
    out["v24_national_total_max_drift"] = drift
    return out


@dataclass(frozen=True)
class ResidualComponents:
    common_sigma: float
    regional_sigma: float
    local_sigma: float
    training_elections: int
    training_rows: int


@dataclass(frozen=True)
class RegionWeightComponents:
    log_weight_sigma: float
    training_transitions: int
    training_elections: int


def draw_region_weight_uncertainty(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    n_sim: int,
    seed: int,
) -> tuple[list[str], np.ndarray, RegionWeightComponents]:
    """Draw forecast region weights from strictly prior vote-volume transitions."""

    required_train = {"election_id", "region_id", "contest_votes"}
    required_target = {"election_id", "region_id", "v24_forecast_region_weight"}
    if required_train - set(train.columns):
        raise ValueError("region-weight training frame is missing required columns")
    if required_target - set(target.columns):
        raise ValueError("region-weight target frame is missing required columns")
    if set(train["election_id"].astype(str)) & set(target["election_id"].astype(str)):
        raise ValueError("target election appears in region-weight training data")

    target_unique = target[["region_id", "v24_forecast_region_weight"]].drop_duplicates(
        "region_id"
    )
    regions = target_unique["region_id"].astype(str).tolist()
    base = target_unique.set_index(target_unique["region_id"].astype(str))[
        "v24_forecast_region_weight"
    ].reindex(regions).astype(float).to_numpy(copy=True)
    base = np.clip(base, EPSILON, None)
    base /= base.sum()

    history = (
        train.assign(region_id=train["region_id"].astype(str))
        .groupby(["election_id", "region_id"], as_index=False)["contest_votes"]
        .first()
    )
    election_order = sorted(
        history["election_id"].astype(str).unique(),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    transition_values: list[float] = []
    transitions = 0
    for previous_id, current_id in zip(election_order, election_order[1:]):
        previous = history.loc[history["election_id"].eq(previous_id)].set_index("region_id")[
            "contest_votes"
        ].astype(float)
        current = history.loc[history["election_id"].eq(current_id)].set_index("region_id")[
            "contest_votes"
        ].astype(float)
        shared = previous.index.intersection(current.index)
        if len(shared) < 2:
            continue
        previous_share = previous.loc[shared] / previous.loc[shared].sum()
        current_share = current.loc[shared] / current.loc[shared].sum()
        change = np.log(current_share.clip(lower=EPSILON)) - np.log(
            previous_share.clip(lower=EPSILON)
        )
        change -= float(change.mean())
        transition_values.extend(change.to_numpy(float).tolist())
        transitions += 1

    if transition_values:
        raw_sigma = float(np.sqrt(np.mean(np.square(transition_values))))
        reliability = float(np.sqrt(transitions / (transitions + 2.0)))
        sigma = float(np.clip(raw_sigma * reliability, 0.0, 0.20))
    else:
        sigma = 0.0
    rng = np.random.default_rng(seed)
    noise = (
        rng.normal(0.0, sigma, size=(n_sim, len(regions)))
        if sigma > 0.0
        else np.zeros((n_sim, len(regions)), dtype=float)
    )
    noise -= noise.mean(axis=1, keepdims=True)
    draws = base[None, :] * np.exp(noise)
    draws /= draws.sum(axis=1, keepdims=True)
    return regions, draws, RegionWeightComponents(
        log_weight_sigma=sigma,
        training_transitions=transitions,
        training_elections=len(election_order),
    )


def _softmax_by_region(
    frame: pd.DataFrame,
    logits: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(logits, dtype=float)
    for indices in frame.groupby("region_id", sort=False).indices.values():
        idx = np.fromiter(indices, dtype=int)
        values = logits[..., idx]
        values = values - np.max(values, axis=-1, keepdims=True)
        exp_values = np.exp(values)
        result[..., idx] = exp_values / exp_values.sum(axis=-1, keepdims=True)
    return result


def _center_by_region(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    centered = values.copy()
    for indices in frame.groupby("region_id", sort=False).indices.values():
        idx = np.fromiter(indices, dtype=int)
        centered[..., idx] -= centered[..., idx].mean(axis=-1, keepdims=True)
    return centered


def hierarchical_residual_draws(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    n_sim: int,
    seed: int,
    prediction_column: str = "v24_regional_shape_pred",
    residual_scale: float = 1.0,
    common_multiplier: float = 1.0,
    regional_multiplier: float = 1.0,
    local_multiplier: float = 1.0,
    distribution: str = "normal",
) -> tuple[np.ndarray, ResidualComponents]:
    """Draw zero-mean log-share residuals from strictly prior elections only."""

    required = {"election_id", "region_id", "slot", "actual", prediction_column}
    missing_train = required - set(train.columns)
    missing_target = {"election_id", "region_id", "slot", prediction_column} - set(target.columns)
    if missing_train:
        raise ValueError(f"interval training frame missing columns: {sorted(missing_train)}")
    if missing_target:
        raise ValueError(f"interval target frame missing columns: {sorted(missing_target)}")
    if train.empty:
        raise ValueError("hierarchical intervals require at least one prior election")
    if set(train["election_id"].astype(str)) & set(target["election_id"].astype(str)):
        raise ValueError("target election appears in interval training data")

    history = train.copy().reset_index(drop=True)
    predicted = history[prediction_column].astype(float).clip(EPSILON, 1.0)
    actual = history["actual"].astype(float).clip(EPSILON, 1.0)
    history["_log_residual"] = np.log(actual) - np.log(predicted)
    history["_log_residual"] = _center_by_region(
        history,
        history["_log_residual"].to_numpy(float),
    )
    if "contest_votes" in history.columns:
        history["_weight"] = history["contest_votes"].astype(float).clip(lower=EPSILON)
    else:
        history["_weight"] = 1.0

    common = (
        history.groupby(["election_id", "slot"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "_common": float(
                        np.average(group["_log_residual"], weights=group["_weight"])
                    )
                }
            ),
            include_groups=False,
        )
    )
    history = history.merge(common, on=["election_id", "slot"], how="left")
    history["_after_common"] = history["_log_residual"] - history["_common"]

    camp_column = "candidate_camp" if "candidate_camp" in history.columns else "slot"
    regional = (
        history.groupby(["region_id", camp_column], as_index=False)["_after_common"]
        .agg([("_regional_mean", "mean"), ("_regional_n", "size")])
        .reset_index(drop=True)
    )
    shrinkage = regional["_regional_n"] / (regional["_regional_n"] + 3.0)
    regional["_regional_effect"] = regional["_regional_mean"] * shrinkage
    history = history.merge(regional, on=["region_id", camp_column], how="left")
    history["_local"] = history["_after_common"] - history["_regional_effect"]

    common_values = common["_common"].to_numpy(float, copy=True)
    common_values -= common_values.mean() if len(common_values) else 0.0
    regional_values = regional["_regional_effect"].to_numpy(float, copy=True)
    regional_values -= regional_values.mean() if len(regional_values) else 0.0
    local_values = history["_local"].to_numpy(float, copy=True)
    local_values -= local_values.mean() if len(local_values) else 0.0
    common_sigma = float(np.sqrt(np.mean(common_values**2))) if len(common_values) else 0.0
    regional_sigma = float(np.sqrt(np.mean(regional_values**2))) if len(regional_values) else 0.0
    local_sigma = float(np.sqrt(np.mean(local_values**2))) if len(local_values) else 0.0

    rng = np.random.default_rng(seed)
    target = target.copy().reset_index(drop=True)
    common_draw = np.zeros((n_sim, len(target)), dtype=float)
    for slot, indices in target.groupby("slot", sort=False).indices.items():
        del slot
        idx = np.fromiter(indices, dtype=int)
        if distribution == "empirical" and len(common_values):
            values = rng.choice(common_values, size=n_sim, replace=True)
        elif distribution == "normal":
            values = rng.normal(0.0, common_sigma, size=n_sim)
        else:
            raise ValueError("distribution must be 'normal' or 'empirical'")
        common_draw[:, idx] = values[:, None]
    common_draw = _center_by_region(target, common_draw)

    if distribution == "empirical" and len(regional_values):
        regional_draw = rng.choice(regional_values, size=(n_sim, len(target)), replace=True)
    else:
        regional_draw = rng.normal(0.0, regional_sigma, size=(n_sim, len(target)))
    regional_draw = _center_by_region(target, regional_draw)
    if distribution == "empirical" and len(local_values):
        local_draw = rng.choice(local_values, size=(n_sim, len(target)), replace=True)
    else:
        local_draw = rng.normal(0.0, local_sigma, size=(n_sim, len(target)))
    local_draw = _center_by_region(target, local_draw)
    noise = float(residual_scale) * (
        float(common_multiplier) * common_draw
        + float(regional_multiplier) * regional_draw
        + float(local_multiplier) * local_draw
    )

    base = target[prediction_column].astype(float).clip(EPSILON, 1.0).to_numpy(float)
    draws = _softmax_by_region(target, np.log(base)[None, :] + noise)
    components = ResidualComponents(
        common_sigma=common_sigma,
        regional_sigma=regional_sigma,
        local_sigma=local_sigma,
        training_elections=int(history["election_id"].nunique()),
        training_rows=len(history),
    )
    return draws, components
