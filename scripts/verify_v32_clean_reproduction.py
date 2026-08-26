"""Rebuild V32 separately and compare it with the frozen active artifact."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_active_presidential_model_v32 as v32  # noqa: E402

FROZEN = ROOT / "outputs/active_presidential_nested_v32/nested_predictions.csv"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="election_forecast_v32_") as temporary:
        destination = Path(temporary) / "active_presidential_nested_v32"
        v32.run(destination)
        expected = pd.read_csv(FROZEN, low_memory=False)
        actual = pd.read_csv(destination / "nested_predictions.csv", low_memory=False)
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, atol=1e-12, rtol=0.0
        )
        manifest = pd.read_csv(destination / "input_manifest.csv")
        paths = manifest.path.astype(str).str.replace("\\", "/", regex=False)
        if paths.str.contains("assembly_issue_character_overlay", regex=False).any():
            raise RuntimeError("clean V32 reproduction retained sentence-level overlay")
        retained = paths.str.endswith(
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        )
        if int(retained.sum()) != 1:
            raise RuntimeError("clean V32 reproduction lost its disclosed frozen profile")
        audit = pd.read_csv(
            destination / "multiplicative_dispersion_expansion_audit.csv",
            encoding="utf-8-sig",
        )
        if not bool((audit["max_candidate_level_shift_pp"].abs() < 1e-9).all()):
            raise RuntimeError("clean V32 reproduction moved a candidate national level")
    print("[clean V32 reproduction: PASS]")


if __name__ == "__main__":
    main()
