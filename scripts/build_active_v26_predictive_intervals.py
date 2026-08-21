"""Build chronological predictive intervals for the bounded V26 point model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_active_v24_predictive_intervals as shared  # noqa: E402


INPUT = ROOT / "outputs" / "active_presidential_nested_v26" / "nested_predictions.csv"
OUTPUT_DIR = ROOT / "outputs" / "active_presidential_nested_v26"


def build(
    *,
    n_sim: int = 50_000,
    # The seed follows the version so that a rerun of V26 is reproducible
    # without reusing the draw sequence a different version was calibrated on.
    seed: int = 26_820,
    residual_scale: float = shared.DEFAULT_RESIDUAL_SCALE,
    levels: tuple[float, ...] = shared.DEFAULT_LEVELS,
    distribution: str = shared.DEFAULT_DISTRIBUTION,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    original_input = shared.INPUT
    try:
        shared.INPUT = INPUT
        payload = shared.build(
            n_sim=n_sim,
            seed=seed,
            residual_scale=residual_scale,
            levels=levels,
            distribution=distribution,
            output_dir=output_dir,
        )
    finally:
        shared.INPUT = original_input

    payload["schema"] = "active_v26_national_predictive_intervals_v1"
    payload["model_version"] = "v26"
    payload["development_outcome_warning"] = (
        "V26 point-model rules inherit through-2022 development choices, and "
        "the graded-intensity pairing was selected by comparing the same five "
        "scored outcomes; coverage is historical calibration, not an untouched "
        "holdout guarantee."
    )
    shared._atomic_json(payload, output_dir / "predictive_interval_manifest.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sim", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=26_820)
    parser.add_argument("--residual-scale", type=float, default=shared.DEFAULT_RESIDUAL_SCALE)
    parser.add_argument(
        "--levels",
        default=",".join(str(level) for level in shared.DEFAULT_LEVELS),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    levels = tuple(float(value) for value in args.levels.split(",") if value.strip())
    payload = build(
        n_sim=args.n_sim,
        seed=args.seed,
        residual_scale=args.residual_scale,
        levels=levels,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
