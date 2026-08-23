"""Rebuild the frozen 2025 demonstration and compare it with the artifact.

The published forecast had no reproduction check of its own. Freezing
``kospi_context_effect`` into the KOSPI fixed dataset - a column that is the
interaction of the market aggregates with a configurable economic
responsibility score, not a market aggregate - silently disabled the runtime
override of that score. The prospective harness began rejecting every run
through its own historical reproduction guard, while every historical CI job
stayed green because none of them runs this path. This script is what would
have caught it.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_prospective_forecast_v29 as v29  # noqa: E402

FROZEN = ROOT / "outputs/prospective_pres_2025_v29"
KEYS = ["election_id", "region_id", "candidate_name"]
TOLERANCE = 1e-12


def main() -> None:
    original = v29.OUTPUT_DIR
    with tempfile.TemporaryDirectory(prefix="election_forecast_v29_prospective_") as temporary:
        destination = Path(temporary) / "prospective_pres_2025_v29"
        try:
            v29.OUTPUT_DIR = destination
            v29.run()
        finally:
            v29.OUTPUT_DIR = original

        for name in ("prospective_predictions.csv", "national_summary.csv"):
            expected = pd.read_csv(FROZEN / name, encoding="utf-8-sig")
            actual = pd.read_csv(destination / name, encoding="utf-8-sig")
            keys = [key for key in KEYS if key in expected.columns]
            merged = expected.merge(
                actual, on=keys, how="outer", suffixes=("_frozen", "_rebuilt"), indicator=True
            )
            unmatched = merged.loc[merged["_merge"].ne("both")]
            if not unmatched.empty:
                raise RuntimeError(f"{name}: row keys changed ({len(unmatched)} unmatched)")
            difference = (
                merged["predicted_share_frozen"] - merged["predicted_share_rebuilt"]
            ).abs()
            worst = float(difference.max())
            if worst > TOLERANCE:
                raise RuntimeError(
                    f"{name}: rebuilt forecast differs from the frozen artifact by {worst:.16g}"
                )

        audit = pd.read_csv(
            destination / "third_share_dispersion_expansion_audit.csv", encoding="utf-8-sig"
        )
        if not bool((audit["max_candidate_level_shift_pp"].abs() < 1e-9).all()):
            raise RuntimeError("the 2025 expansion moved a candidate national level")
    print("[V29 prospective reproduction: PASS]")


if __name__ == "__main__":
    main()
