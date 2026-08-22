"""Evaluate an outcome-free floor for inherited party-regional contrasts.

``recent_bloc_base`` already measures the point-in-time regional shape of each
candidate's political bloc.  The fitted prediction can nevertheless flatten
that shape.  This experiment preserves a reliability-weighted fraction of the
prior contrast only when the fitted prediction points in the same direction:

    |final regional logit contrast| >= gain * reliability * |prior contrast|

Opposite-signed contrasts are left alone, so a measured realignment is not
silently reversed.  After applying the floor, iterative calibration restores
both regional sums and every candidate's original vote-weighted national
share.  The experiment therefore changes regional shape, not candidate size.

The gain ladder is a development-panel sensitivity analysis, not a fitted
parameter.  No post-2022 outcome is read.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v26"
OUTPUT_DIR = ROOT / "outputs" / "party_regionalism_retention"
DEFAULT_GAINS = (0.0, 1.0, 2.0, 3.0, 4.0)
EPSILON = 1e-7


def _name_column(frame: pd.DataFrame) -> str:
    return "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"


def _logit(values: pd.Series) -> pd.Series:
    clipped = values.astype(float).clip(EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _weighted_center(values: pd.Series, weights: pd.Series, groups: pd.Series) -> pd.Series:
    numerator = (values * weights).groupby(groups).transform("sum")
    denominator = weights.groupby(groups).transform("sum")
    return values - numerator / denominator


def _calibrate(
    values: np.ndarray,
    candidate_codes: np.ndarray,
    region_codes: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    iterations: int = 200,
) -> np.ndarray:
    """Rake candidate national totals while preserving every regional sum."""

    out = np.clip(values.astype(float), EPSILON, None)
    candidate_count = int(candidate_codes.max()) + 1
    for _ in range(iterations):
        totals = np.bincount(
            candidate_codes, weights=out * weights, minlength=candidate_count
        )
        factors = np.divide(targets, totals, out=np.ones_like(targets), where=totals > 0)
        out *= factors[candidate_codes]
        region_totals = np.bincount(region_codes, weights=out)
        out /= region_totals[region_codes]
        current = np.bincount(
            candidate_codes, weights=out * weights, minlength=candidate_count
        )
        if np.max(np.abs(current - targets)) < 1e-11:
            break
    return out


def retain(frame: pd.DataFrame, gain: float) -> pd.DataFrame:
    """Apply the same-direction regional contrast floor election by election."""

    name = _name_column(frame)
    parts: list[pd.DataFrame] = []
    for _election_id, source in frame.groupby("election_id", sort=False):
        out = source.copy()
        weights = pd.to_numeric(out["contest_votes"], errors="coerce").fillna(0.0)
        groups = out[name].astype(str)
        pred_logit = _logit(out["layer_pred"])
        prior_logit = _logit(out["recent_bloc_base"])
        pred_contrast = _weighted_center(pred_logit, weights, groups)
        prior_contrast = _weighted_center(prior_logit, weights, groups)
        reliability = pd.to_numeric(
            out["direct_party_reliability"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        national = out.groupby(groups).apply(
            lambda part: np.average(part.layer_pred, weights=part.contest_votes)
        ).sort_values(ascending=False)
        third_share = float(national.iloc[2]) if len(national) > 2 else 0.0
        floor = float(gain) * third_share * reliability * prior_contrast.abs()
        same_direction = pred_contrast.mul(prior_contrast).gt(0.0)
        shallow = pred_contrast.abs().lt(floor)
        adjusted = pred_contrast.copy()
        mask = same_direction & shallow
        adjusted.loc[mask] = np.sign(pred_contrast.loc[mask]) * floor.loc[mask]

        candidate_codes, _ = pd.factorize(groups, sort=False)
        region_codes, _ = pd.factorize(out["region_id"].astype(str), sort=False)
        base = 1.0 / (1.0 + np.exp(-(pred_logit + adjusted - pred_contrast)))
        original = out["layer_pred"].to_numpy(float)
        target = np.bincount(candidate_codes, weights=original * weights.to_numpy(float))
        calibrated = _calibrate(
            base.to_numpy(float),
            candidate_codes,
            region_codes,
            weights.to_numpy(float),
            target,
        )
        out["layer_pred"] = calibrated
        out["party_regionalism_floor_bound"] = mask.to_numpy(bool)
        out["party_regionalism_third_share"] = third_share
        out["party_regionalism_prior_contrast"] = prior_contrast.to_numpy(float)
        out["party_regionalism_pred_contrast"] = pred_contrast.to_numpy(float)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def metrics(frame: pd.DataFrame, gain: float) -> pd.DataFrame:
    name = _name_column(frame)
    rows: list[dict[str, object]] = []
    for election_id, group in frame.groupby("election_id", sort=False):
        actual = group.groupby(name).apply(
            lambda part: np.average(part.actual, weights=part.contest_votes)
        )
        predicted = group.groupby(name).apply(
            lambda part: np.average(part.layer_pred, weights=part.contest_votes)
        )
        rows.append(
            {
                "gain": gain,
                "election_id": election_id,
                "regional_weighted_mae_pp": np.average(
                    (group.layer_pred - group.actual).abs(), weights=group.contest_votes
                ) * 100.0,
                "national_mae_pp": (predicted - actual).abs().mean() * 100.0,
                "winner_correct": predicted.idxmax() == actual.idxmax(),
                "bound_cells": int(group["party_regionalism_floor_bound"].sum()),
                "over_10pp_cells": int(
                    ((group.layer_pred - group.actual).abs() > 0.10).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def run(active_dir: Path = ACTIVE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(active_dir / "nested_predictions.csv", low_memory=False)
    by_election = pd.concat(
        [metrics(retain(source, gain), gain) for gain in DEFAULT_GAINS],
        ignore_index=True,
    )
    summary = (
        by_election.groupby("gain", sort=False)
        .agg(
            regional_weighted_macro_pp=("regional_weighted_mae_pp", "mean"),
            national_macro_pp=("national_mae_pp", "mean"),
            winners_correct=("winner_correct", "sum"),
            bound_cells=("bound_cells", "sum"),
            over_10pp_cells=("over_10pp_cells", "sum"),
        )
        .reset_index()
    )
    return summary, by_election


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-dir", type=Path, default=ACTIVE_DIR)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    summary, by_election = run(args.active_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    print(summary.round(4).to_string(index=False))
    print()
    print(by_election.pivot(index="election_id", columns="gain", values="regional_weighted_mae_pp").round(3))


if __name__ == "__main__":
    main()
