"""Preserve the evidenced concrete share of inherited regional dispersion."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-7
DEFAULT_GAIN = 1.0


def _name_column(frame: pd.DataFrame) -> str:
    return "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"


def _logit(values: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(values, errors="raise").clip(EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _weighted_center(values: pd.Series, weights: pd.Series, groups: pd.Series) -> pd.Series:
    numerator = (values * weights).groupby(groups).transform("sum")
    denominator = weights.groupby(groups).transform("sum")
    return values - numerator / denominator


def _weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.average(values, weights=weights))
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def _calibrate(
    values: np.ndarray,
    candidate_codes: np.ndarray,
    region_codes: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    iterations: int = 200,
) -> np.ndarray:
    out = np.clip(values.astype(float), EPSILON, None)
    candidate_count = int(candidate_codes.max()) + 1
    for _ in range(iterations):
        totals = np.bincount(candidate_codes, weights=out * weights, minlength=candidate_count)
        factors = np.divide(targets, totals, out=np.ones_like(targets), where=totals > 0)
        out *= factors[candidate_codes]
        region_totals = np.bincount(region_codes, weights=out)
        out /= region_totals[region_codes]
        current = np.bincount(candidate_codes, weights=out * weights, minlength=candidate_count)
        if np.max(np.abs(current - targets)) < 1e-11:
            break
    return out


def apply_party_regionalism_dispersion(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "layer_pred",
    weight_column: str = "contest_votes",
    gain: float = DEFAULT_GAIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand only concrete-backed missing regional dispersion.

    The point-in-time bloc prior supplies width, never a regional vote target.
    Candidate national levels and regional compositions are conserved.
    """

    required = {
        "election_id", "region_id", weight_column, prediction_column,
        "recent_bloc_base", "core_voting_mass", "direct_party_reliability",
    }
    name = _name_column(frame)
    required.add(name)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"party regionalism dispersion missing columns: {sorted(missing)}")
    if gain < 0.0:
        raise ValueError("party regionalism dispersion gain must be nonnegative")

    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for election_id, source in frame.groupby("election_id", sort=False):
        out = source.copy()
        weights = pd.to_numeric(out[weight_column], errors="raise")
        if not weights.gt(0.0).any():
            raise ValueError(f"party regionalism dispersion has no positive {weight_column}")
        groups = out[name].astype(str)
        pred_logit = _logit(out[prediction_column])
        prior_logit = _logit(out["recent_bloc_base"])
        pred_contrast = _weighted_center(pred_logit, weights, groups)
        prior_contrast = _weighted_center(prior_logit, weights, groups)
        adjusted = pred_contrast.copy()

        for candidate, indices in out.groupby(name).groups.items():
            positions = np.array([out.index.get_loc(index) for index in indices])
            candidate_weights = weights.iloc[positions].to_numpy(float)
            pred_sd = _weighted_sd(pred_contrast.iloc[positions].to_numpy(float), candidate_weights)
            prior_sd = _weighted_sd(prior_contrast.iloc[positions].to_numpy(float), candidate_weights)
            core = float(np.average(
                pd.to_numeric(out.iloc[positions]["core_voting_mass"], errors="coerce").fillna(0.0),
                weights=candidate_weights,
            ))
            reliability = float(np.average(
                pd.to_numeric(out.iloc[positions]["direct_party_reliability"], errors="coerce").fillna(0.0),
                weights=candidate_weights,
            ))
            missing_ratio = max(0.0, prior_sd / max(pred_sd, 1e-9) - 1.0)
            factor = 1.0 + float(gain) * core * reliability * missing_ratio
            adjusted.iloc[positions] *= factor
            audit_rows.append({
                "election_id": str(election_id),
                "candidate_name": str(candidate),
                "prediction_logit_sd": pred_sd,
                "prior_logit_sd": prior_sd,
                "core_voting_mass": core,
                "direct_party_reliability": reliability,
                "missing_dispersion_ratio": missing_ratio,
                "dispersion_factor": factor,
                "gain": float(gain),
            })

        raw = 1.0 / (1.0 + np.exp(-(pred_logit + adjusted - pred_contrast)))
        candidate_codes, _ = pd.factorize(groups, sort=False)
        region_codes, _ = pd.factorize(out["region_id"].astype(str), sort=False)
        weight_values = weights.to_numpy(float)
        targets = np.bincount(
            candidate_codes,
            weights=out[prediction_column].to_numpy(float) * weight_values,
        )
        out[prediction_column] = _calibrate(
            raw.to_numpy(float), candidate_codes, region_codes, weight_values, targets
        )
        parts.append(out)

    result = pd.concat(parts, ignore_index=True)
    totals = result.groupby(["election_id", "region_id"])[prediction_column].sum()
    if not np.allclose(totals.to_numpy(float), 1.0, atol=1e-10):
        raise RuntimeError("party regionalism dispersion broke regional composition")
    return result, pd.DataFrame(audit_rows)
