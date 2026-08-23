"""Measure a dispersion rescale indexed on the predicted third-candidate share.

Predicted regional spread is close to realised in 2002, 2012 and 2022 and
compressed in 2007 and 2017, the two elections with a substantial third
candidate. The compression is present at the earliest modelled stage, so it
belongs to the fitted base rather than to any postprocess layer.

Shrinkage is not by itself a defect - a regularised predictor should have less
variance than the outcome, and the conditional mean does too. What marks 2007
and 2017 is that the slope of realised on predicted exceeds 1 there, near 2 for
이회창 2007, while it sits at 1 elsewhere. That is a calibration gap, not
optimal shrinkage.

The rescale tested here expands each candidate's regional deviations around
their own national level:

    scaled = level + (1 + gain * predicted_third_share) * (pred - level)

then renormalises each region. The index is the model's **predicted** third
share, which is available at forecast time, so the transform reads no outcome.
The gain is swept rather than fitted, and the sweep is the point: a gain that
helps only at one value, or that helps the two compressed elections while
hurting the other three, is not a correction.

Scope: development comparison over the through-2022 sample. The same five
outcomes measure every cell, so this is not a holdout, and the gap being closed
was found by reading those outcomes' residuals.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from presidential_issue_engine.third_share_dispersion_expansion import (  # noqa: F401
    apply_third_share_dispersion_expansion,
    predicted_third_share,  # re-exported: the sweep's guards address it here
)
from scripts.active_model_pointer import active_output_dir

ACTIVE_DIR = active_output_dir()
OUTPUT_DIR = ROOT / "outputs" / "regional_dispersion_calibration"
DEFAULT_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def _name_column(frame: pd.DataFrame) -> str:
    return "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"


def rescale(frame: pd.DataFrame, gain: float) -> pd.DataFrame:
    """Expand regional deviations around each candidate's own national level.

    Delegates to the shipped transform rather than restating it, so a sweep can
    never describe a mechanism the model does not actually apply. An earlier
    private copy here clipped negative shares and renormalised, which injected
    vote mass and made the higher gains look like they cost national accuracy
    when the cost was the clipping.
    """

    adjusted, _audit = apply_third_share_dispersion_expansion(frame, gain=gain)
    return adjusted


def metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    name = _name_column(frame)
    rows: list[dict[str, object]] = []
    for election, group in frame.groupby("election_id"):
        actual = group.groupby(name).apply(
            lambda part: np.average(part.actual, weights=part.contest_votes)
        )
        predicted = group.groupby(name).apply(
            lambda part: np.average(part.layer_pred, weights=part.contest_votes)
        )
        ratios = []
        for _candidate, part in group.groupby(name):
            spread = part.actual.std(ddof=1)
            if spread > 1e-6:
                ratios.append(part.layer_pred.std(ddof=1) / spread)
        rows.append(
            {
                "gain": label,
                "election_id": str(election),
                "regional_weighted_mae_pp": float(
                    np.average(
                        (group.layer_pred - group.actual).abs(),
                        weights=group.contest_votes,
                    )
                    * 100
                ),
                "national_mae_pp": float((predicted - actual).abs().mean() * 100),
                "dispersion_ratio": float(np.mean(ratios)) if ratios else np.nan,
                "winner_correct": bool(predicted.idxmax() == actual.idxmax()),
            }
        )
    return pd.DataFrame(rows)


def run(
    active_dir: Path = ACTIVE_DIR, gains: tuple[float, ...] = DEFAULT_GAINS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(
        active_dir / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    parts = [metrics(rescale(frame, gain), f"{gain:.2f}") for gain in gains]
    by_election = pd.concat(parts, ignore_index=True)
    summary = (
        by_election.groupby("gain", sort=False)
        .agg(
            regional_weighted_macro_pp=("regional_weighted_mae_pp", "mean"),
            national_macro_pp=("national_mae_pp", "mean"),
            mean_dispersion_ratio=("dispersion_ratio", "mean"),
            winners_correct=("winner_correct", "sum"),
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
    print("regional weighted MAE by election")
    print(
        by_election.pivot(
            index="election_id", columns="gain", values="regional_weighted_mae_pp"
        ).round(3).to_string()
    )


if __name__ == "__main__":
    main()
