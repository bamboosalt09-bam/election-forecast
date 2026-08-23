"""Rebuild V28 separately and compare it with the frozen active artifact."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_active_presidential_model_v28 as v28  # noqa: E402

FROZEN = ROOT / "outputs/active_presidential_nested_v28/nested_predictions.csv"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="election_forecast_v28_") as temporary:
        destination = Path(temporary) / "active_presidential_nested_v28"
        v28.run(destination)
        expected = pd.read_csv(FROZEN, low_memory=False)
        actual = pd.read_csv(destination / "nested_predictions.csv", low_memory=False)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-12, rtol=0.0)
        manifest = pd.read_csv(destination / "input_manifest.csv")
        paths = manifest.path.astype(str).str.replace("\\", "/", regex=False)
        if paths.str.contains("assembly_issue_character_overlay", regex=False).any():
            raise RuntimeError("clean V28 reproduction retained sentence-level overlay")
        retained = paths.str.endswith(
            "data/raw/auto_issue_seed/candidate_issue_profile.csv"
        )
        if int(retained.sum()) != 1:
            raise RuntimeError("clean V28 reproduction lost its disclosed frozen profile")
    print("[clean V28 reproduction: PASS]")


if __name__ == "__main__":
    main()
