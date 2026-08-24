"""Run V29: V28 plus a third-share-indexed regional dispersion expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import third_share_dispersion_expansion  # noqa: E402
from scripts import run_active_presidential_model_v24 as v24  # noqa: E402
from scripts import run_active_presidential_model_v28 as v28  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v29"
FINAL_VARIANT = "v29_third_share_dispersion_expansion"


def run(output_dir: Path | None = None, gain: float | None = None) -> Path:
    """Run V29, optionally at a gain other than the promoted one.

    The promoted gain is 1.0, where the expansion factor is the predicted third
    share itself and no constant is selected. A swept gain of 0.50 scores better
    on the five scored outcomes and was rejected for exactly that reason - it is
    a constant chosen on the panel it is then measured against. It stays
    available because "better on the panel" is still worth being able to
    measure; it is not pre-registered and adopting it would need its own
    promotion.

    Writing a non-default gain into the promoted directory is refused. A
    frozen artifact that silently holds something other than the promoted
    configuration is the one outcome this must not allow.
    """

    selected = third_share_dispersion_expansion.DEFAULT_GAIN if gain is None else float(gain)
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    if selected != third_share_dispersion_expansion.DEFAULT_GAIN and destination == DEFAULT_OUTPUT:
        raise SystemExit(
            f"refusing to write gain {selected} into the promoted directory "
            f"{DEFAULT_OUTPUT.relative_to(ROOT).as_posix()}; pass --output-dir"
        )
    destination.mkdir(parents=True, exist_ok=True)
    v28.run(output_dir=destination)

    path = destination / "nested_predictions.csv"
    predictions = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    predictions["v28_pre_third_share_expansion_pred"] = predictions["layer_pred"]
    predictions, audit = third_share_dispersion_expansion.apply_third_share_dispersion_expansion(
        predictions, gain=selected
    )
    v24._atomic_csv_crlf(predictions, path)
    v24._atomic_csv_crlf(audit, destination / "third_share_dispersion_expansion_audit.csv")

    from scripts import run_active_presidential_model as active

    summary, by_election, national = active.nested._metrics(
        predictions, "layer_pred", FINAL_VARIANT
    )
    v24._atomic_csv_crlf(by_election, destination / "by_election.csv")
    v24._atomic_csv_crlf(national, destination / "national_predictions.csv")

    summary_path = destination / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["pre_v29_extension_metrics"] = payload.get("metrics")
    payload["metrics"] = summary
    payload["policy_version"] = "active_v29_third_share_dispersion_expansion"
    payload["predecessor"] = "v28"
    payload["third_share_dispersion_expansion"] = {
        "gain": third_share_dispersion_expansion.DEFAULT_GAIN,
        "index": "model_predicted_third_placed_national_level",
        "candidate_national_level_preserved": True,
        "regional_composition_preserved": True,
        "gain_selection": "parameter_free_unit_gain_not_swept",
        "better_scoring_gain_rejected": 0.5,
        "outcome_fields_used": [],
    }
    # V28 stamps these; the expansion changes neither, and re-deriving the
    # metrics above must not silently drop them.
    payload["external_neural_model_runtime"] = False
    payload["post_2022_outcomes_used"] = False
    v24._atomic_json_crlf(payload, summary_path)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help=(
            "expansion gain; defaults to the promoted 1.0. Any other value "
            "requires --output-dir and is stamped as non-promoted in the summary."
        ),
    )
    args = parser.parse_args()
    destination = run(args.output_dir, gain=args.gain)
    print(v24.report(destination).to_string(index=False))


if __name__ == "__main__":
    main()
