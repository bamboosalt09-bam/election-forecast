"""Evaluate core-weighted preservation of inherited party regional dispersion.

Unlike a regional share floor, this transform acts on a candidate's whole
regional distribution.  When the fitted logit dispersion is narrower than the
PIT ``recent_bloc_base`` dispersion, it expands the fitted contrasts by:

    factor = 1 + gain * core_mass * reliability * (prior_sd / fitted_sd - 1)

Only positive dispersion gaps are used.  Candidate national shares and each
region's unit sum are restored by iterative calibration.  Gain 1 has a direct
interpretation: the evidenced concrete share preserves its proportional part
of the missing inherited dispersion.  Other gains are sensitivity only.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_party_regionalism_retention import (
    ACTIVE_DIR,
    _calibrate,
    _logit,
    _name_column,
    _weighted_center,
    metrics,
)

OUTPUT_DIR = ROOT / "outputs" / "party_regionalism_dispersion"
DEFAULT_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def _weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.average(values, weights=weights))
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def expand(frame: pd.DataFrame, gain: float = 1.0) -> pd.DataFrame:
    """Expand only an evidenced, concrete-backed inherited dispersion gap."""

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
        adjusted = pred_contrast.copy()
        factors = pd.Series(1.0, index=out.index)

        for _candidate, indices in out.groupby(name).groups.items():
            positions = np.array([out.index.get_loc(index) for index in indices])
            candidate_weights = weights.iloc[positions].to_numpy(float)
            pred_sd = _weighted_sd(
                pred_contrast.iloc[positions].to_numpy(float), candidate_weights
            )
            prior_sd = _weighted_sd(
                prior_contrast.iloc[positions].to_numpy(float), candidate_weights
            )
            core = float(
                np.average(
                    pd.to_numeric(
                        out.iloc[positions]["core_voting_mass"], errors="coerce"
                    ).fillna(0.0),
                    weights=candidate_weights,
                )
            )
            reliability = float(
                np.average(
                    pd.to_numeric(
                        out.iloc[positions]["direct_party_reliability"], errors="coerce"
                    ).fillna(0.0),
                    weights=candidate_weights,
                )
            )
            missing_ratio = max(0.0, prior_sd / max(pred_sd, 1e-9) - 1.0)
            factor = 1.0 + float(gain) * core * reliability * missing_ratio
            adjusted.iloc[positions] *= factor
            factors.iloc[positions] = factor

        raw = 1.0 / (1.0 + np.exp(-(pred_logit + adjusted - pred_contrast)))
        candidate_codes, _ = pd.factorize(groups, sort=False)
        region_codes, _ = pd.factorize(out["region_id"].astype(str), sort=False)
        weight_values = weights.to_numpy(float)
        targets = np.bincount(
            candidate_codes,
            weights=out["layer_pred"].to_numpy(float) * weight_values,
        )
        out["layer_pred"] = _calibrate(
            raw.to_numpy(float),
            candidate_codes,
            region_codes,
            weight_values,
            targets,
        )
        out["party_regionalism_dispersion_factor"] = factors.to_numpy(float)
        # Compatibility with the shared metric helper.
        out["party_regionalism_floor_bound"] = factors.gt(1.0).to_numpy(bool)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def run(active_dir: Path = ACTIVE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(active_dir / "nested_predictions.csv", low_memory=False)
    by_election = pd.concat(
        [metrics(expand(source, gain), gain) for gain in DEFAULT_GAINS],
        ignore_index=True,
    )
    summary = (
        by_election.groupby("gain", sort=False)
        .agg(
            regional_weighted_macro_pp=("regional_weighted_mae_pp", "mean"),
            national_macro_pp=("national_mae_pp", "mean"),
            winners_correct=("winner_correct", "sum"),
            active_cells=("bound_cells", "sum"),
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


if __name__ == "__main__":
    main()
