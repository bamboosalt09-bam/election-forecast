"""Build chronological predictive intervals for V30."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_active_v24_predictive_intervals as shared  # noqa: E402

INPUT = ROOT / "outputs/active_presidential_nested_v30/nested_predictions.csv"
OUTPUT_DIR = ROOT / "outputs/active_presidential_nested_v30"


def build(*, n_sim: int = 50_000, seed: int = 30_820, output_dir: Path = OUTPUT_DIR):
    original = shared.INPUT
    try:
        shared.INPUT = INPUT
        payload = shared.build(n_sim=n_sim, seed=seed, output_dir=output_dir)
    finally:
        shared.INPUT = original
    payload["schema"] = "active_v30_national_predictive_intervals_v1"
    payload["model_version"] = "v30"
    payload["development_outcome_warning"] = (
        "V30 keeps V29 s transform and weights it at forecast time."
        "expansion conserves each candidate's national level, so these national "
        "intervals are inherited from V28 rather than re-evidenced. Coverage "
        "remains historical calibration, not holdout evidence."
    )
    shared._atomic_json(payload, output_dir / "predictive_interval_manifest.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sim", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=30_820)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(n_sim=args.n_sim, seed=args.seed, output_dir=args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
