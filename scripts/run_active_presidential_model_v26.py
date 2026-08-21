"""Run V26: V25 with a graded mega-issue intensity and event-class alignment.

V25 reaches only four mega-issue intensities, and ``intensity_activation`` is
``(intensity - 1).clip(0, 1)``, so a direct political shock is either inert or
saturated. V26 changes exactly two things:

1. the intensity table is the graded ladder in ``outputs/automatic_controls_v26``
2. ``align_profile_to_event_class`` is applied before direct mega attribution,
   which the forecast path already did and the retrospective did not

The two are inseparable. Grading the intensity alone degrades the panel badly,
because raising the floors exposes the winner-take-all issue race on 2007 and
2022, where ``security_nk`` leads a class that does not contain it; the
intensity gate at 1.00 had been suppressing that. The alignment alone is
bit-identical to V25. Applied together every scored election holds or improves.

Nothing else changes: the Ridge model, its predictors, the V24 ballot panel and
the three structural postprocesses are V25's. See
``docs/EXPERIMENT_V25_INTENSITY_LADDER_20260822.md`` for the two-by-two, and
``docs/FINAL_MODEL_V26_20260822.md`` for the promotion record.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from presidential_issue_engine import mega_issue_adjustment  # noqa: E402
from scripts import run_active_presidential_model_v25 as v25  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v26"
AUTOMATIC_DIR = ROOT / "outputs" / "automatic_controls_v26"
FINAL_VARIANT = "v26_graded_mega_intensity_event_aligned"


@contextmanager
def graded_mega_runtime(automatic_dir: Path = AUTOMATIC_DIR) -> Iterator[None]:
    """Point V25 at the laddered controls and align direct attribution.

    Both patches are reverted on exit so that importing this module can never
    change how V25 behaves for a caller that runs it directly.
    """

    if not automatic_dir.exists():
        raise RuntimeError(
            f"{automatic_dir.name} is missing; run scripts/build_automatic_controls_v26.py"
        )
    taxonomy = pd.read_csv(
        automatic_dir / "mega_issue_taxonomy.csv", encoding="utf-8-sig"
    )
    original_dir = v25.AUTOMATIC_DIR
    original_variant = v25.FINAL_VARIANT
    original_compile = mega_issue_adjustment.compile_direct_mega_scores

    def aligned(profile, intensity, election_dates, **kwargs):
        return original_compile(
            mega_issue_adjustment.align_profile_to_event_class(
                profile, taxonomy, election_dates
            ),
            intensity,
            election_dates,
            **kwargs,
        )

    v25.AUTOMATIC_DIR = automatic_dir
    # The metric rows are keyed by variant, so V26 must stamp its own name or
    # it overwrites V25's row in any shared summary.
    v25.FINAL_VARIANT = FINAL_VARIANT
    mega_issue_adjustment.compile_direct_mega_scores = aligned
    try:
        yield
    finally:
        v25.AUTOMATIC_DIR = original_dir
        v25.FINAL_VARIANT = original_variant
        mega_issue_adjustment.compile_direct_mega_scores = original_compile


def run(output_dir: Path | None = None) -> Path:
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    with graded_mega_runtime():
        v25.run(output_dir=destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    destination = run(output_dir=args.output_dir)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
