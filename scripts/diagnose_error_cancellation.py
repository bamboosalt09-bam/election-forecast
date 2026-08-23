"""Report how much of the national metric is regional errors cancelling out.

The national candidate metric is a vote-weighted mean of regional predictions
compared with the same mean of realised shares. Signed regional errors therefore
offset inside it, and a prediction whose regional errors point in opposite
directions can be nationally exact while being regionally wrong everywhere.

    cancellation = 1 - |weighted signed error| / weighted absolute error

A value near 1 means the national figure is arithmetic rather than accuracy.

This matters for reading the headline, and it explains a result that took seven
attempts to understand. Every correction tried against regional compression -
dispersion rescales, personal-history caps, absolute core erosion, prior
anchoring, a historical bound - improved regional error and degraded national
error. They were not breaking the national calibration. Compression produces
offsetting errors by construction, so removing compression removes the
cancellation that the national figure was made of.

Read-only. No model change and none proposed.
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

from scripts.active_model_pointer import active_output_dir

ACTIVE_DIR = active_output_dir()
OUTPUT_DIR = ROOT / "outputs" / "error_cancellation"


def _name_column(frame: pd.DataFrame) -> str:
    return "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"


def cancellation(frame: pd.DataFrame) -> pd.DataFrame:
    """Signed against absolute regional error, per election and candidate."""

    name = _name_column(frame)
    rows: list[dict[str, object]] = []
    for (election, candidate), group in frame.groupby(["election_id", name]):
        weights = group["contest_votes"] / group["contest_votes"].sum()
        error = group["layer_pred"] - group["actual"]
        signed = float((weights * error).sum() * 100)
        absolute = float((weights * error.abs()).sum() * 100)
        rows.append(
            {
                "election_id": str(election),
                "candidate_name": str(candidate),
                "national_error_pp": signed,
                "regional_absolute_pp": absolute,
                "cancellation": 1.0 - abs(signed) / absolute if absolute > 1e-9 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-dir", type=Path, default=ACTIVE_DIR)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    frame = pd.read_csv(
        args.active_dir / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    detail = cancellation(frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT_DIR / "by_candidate.csv", index=False, encoding="utf-8-sig")

    by_election = (
        detail.groupby("election_id")
        .agg(
            mean_cancellation=("cancellation", "mean"),
            mean_national_error_pp=("national_error_pp", lambda s: s.abs().mean()),
            mean_regional_absolute_pp=("regional_absolute_pp", "mean"),
        )
        .reset_index()
    )
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")

    print(detail.sort_values("cancellation", ascending=False).round(3).to_string(index=False))
    print()
    print(by_election.round(3).to_string(index=False))
    print()
    overall = float(detail["cancellation"].mean())
    print(f"mean cancellation across the panel: {overall:.3f}")
    highest = detail.loc[detail["cancellation"].idxmax()]
    print(
        f"largest: {highest['candidate_name']} {highest['election_id']} -"
        f" national {highest['national_error_pp']:+.3f} %p against regional"
        f" {highest['regional_absolute_pp']:.3f} %p"
    )
    print()
    print(
        "A national figure built this way is not evidence that regional levels are"
        " right; it is evidence that their errors point in opposite directions."
    )


if __name__ == "__main__":
    main()
