"""Rebuild the frozen 2025 demonstration and compare it with the artifact.

The published forecast had no reproduction check of its own. Enforcing the V28
external-model boundary process-wide made its path unrunnable, and every
historical CI job stayed green throughout, because none of them runs it.

Running it also surfaced a second fact nobody had tested: the 2025 forecast
depended on the full Assembly stance extraction, which carries verbatim
excerpts and is not redistributed - so it could not be rebuilt from the public
tree at all. Every consumer of that file turned out to use only the excerpt's
length, so a derived form is published instead and the demonstration is now
reproducible from what the repository ships. Where even that is missing this
script says so and verifies nothing, rather than reporting success.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_prospective_forecast_v32 as v32  # noqa: E402

FROZEN = ROOT / "outputs/prospective_pres_2025_v32"
KEYS = ["election_id", "region_id", "candidate_name"]
TOLERANCE = 1e-12


def unresolved_inputs() -> list[str]:
    """Required 2025 inputs that resolve to nothing on this checkout.

    The collected Assembly stance rows carry verbatim excerpts and are not
    redistributed, but their derived form is, and every consumer takes only the
    excerpt's length. So the question is not whether the private file is here -
    it is whether the input resolves at all.
    """

    from scripts import run_prospective_forecast as harness

    unresolved: list[str] = []
    if not harness.OFFICIAL_2025_MINUTES.exists():
        unresolved.append(
            harness.OFFICIAL_2025_MINUTES.relative_to(ROOT).as_posix()
            + " (and no redistributable form beside it)"
        )
    return unresolved


def main() -> None:
    absent = unresolved_inputs()
    if absent:
        print("[V32 prospective reproduction: SKIPPED - required input unresolved]")
        for relative in absent:
            print(f"  missing: {relative}")
        for note in (
            "The 2025 demonstration could not be rebuilt on this checkout.",
            "Build the redistributable stance rows with",
            "scripts/build_redistributable_pres_2025_stance_rows.py, or restore the",
            "collected file. Nothing was verified.",
        ):
            print(note)
        return

    original = v32.OUTPUT_DIR
    with tempfile.TemporaryDirectory(prefix="election_forecast_v32_prospective_") as temporary:
        destination = Path(temporary) / "prospective_pres_2025_v32"
        try:
            v32.OUTPUT_DIR = destination
            v32.run()
        finally:
            v32.OUTPUT_DIR = original

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
            destination / "multiplicative_dispersion_expansion_audit.csv", encoding="utf-8-sig"
        )
        if not bool((audit["max_candidate_level_shift_pp"].abs() < 1e-9).all()):
            raise RuntimeError("the 2025 expansion moved a candidate national level")
    print("[V32 prospective reproduction: PASS]")


if __name__ == "__main__":
    main()
