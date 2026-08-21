"""Write the V26 automatic controls: V23's, with the intensity ladder applied.

V26 differs from V25 in exactly two places. This script produces the first of
them - a control directory identical to ``automatic_controls_v23`` except that
``mega_issue_intensity.csv`` carries graded rather than class-quantised
intensity. The second, the event-class alignment, is applied by the V26 runner.

Keeping the laddered table as a written control rather than an in-memory patch
means the promoted model reads the same kind of input every previous version
read, and the difference from V23 is inspectable as a diff.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "presidential_issue_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from presidential_issue_engine.mega_issue_intensity_ladder import (  # noqa: E402
    ladder_intensity,
)

SOURCE = ROOT / "outputs" / "automatic_controls_v23"
DESTINATION = ROOT / "outputs" / "automatic_controls_v26"
# The classifier components are written to the v22 taxonomy audit, which is a
# canonical tracked output carrying the same four columns as the speech-derived
# diagnostics.
DIAGNOSTICS = ROOT / "outputs" / "automatic_controls_v22" / "mega_issue_taxonomy_audit.csv"
INTENSITY_NAME = "mega_issue_intensity.csv"
LADDER_NOTE = (
    "Graded institutional-crisis proximity applied to the V23 class intensity; "
    "classifier gates reused, no new constant, no election outcome"
)


def build(destination: Path = DESTINATION) -> pd.DataFrame:
    """Copy the V23 controls and replace the intensity table with the ladder."""

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(SOURCE, destination)

    diagnostics = pd.read_csv(DIAGNOSTICS, encoding="utf-8-sig")
    intensity = pd.read_csv(SOURCE / INTENSITY_NAME, encoding="utf-8-sig")
    laddered = ladder_intensity(intensity, diagnostics)
    if "notes" in laddered.columns:
        laddered["notes"] = LADDER_NOTE
    laddered.to_csv(destination / INTENSITY_NAME, index=False, encoding="utf-8-sig")
    return laddered


def main() -> None:
    laddered = build()
    before = pd.read_csv(SOURCE / INTENSITY_NAME, encoding="utf-8-sig")
    print(f"wrote {DESTINATION.relative_to(ROOT).as_posix()}")
    for election, floor, raised in zip(
        before["election_id"],
        before["mega_issue_intensity"],
        laddered["mega_issue_intensity"],
    ):
        activation = min(max(float(raised) - 1.0, 0.0), 1.0)
        print(
            f"  {election:<10} {float(floor):.2f} -> {float(raised):.6f}"
            f"   activation {activation:.4f}"
        )


if __name__ == "__main__":
    main()
