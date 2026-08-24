"""Run V30: V29 with both terminal transforms on forecast-time weights.

V27 and V29 each weight a candidate's national level by ``contest_votes`` - the
target election's own regional turnout, which exists only once the votes are
counted. A postprocess using it consumes an outcome of the election it is
predicting. The 2025 prospective path already refuses that number and
substitutes the previous election's volumes; V30 makes the scored panel do the
same, so the historical figures describe something a forecast could actually
have produced.

Nothing else changes. The Ridge stack, the predictors, the shock structure, the
V28 external-model boundary and both transforms' functional forms are V29's.
Only the weight each transform reads is different.

The cost was measured before the change rather than discovered after it:

    2002  +0.0866    (no predecessor in the panel; equal regions)
    2007  -0.0149
    2012   0.0000    (no third candidate; the expansion does not act)
    2017  -0.0115
    2022  -0.0007
    macro +0.0119 percentage points

Almost all of it is 2002 giving up a volume-weighted level for an equal-region
one. The leak was wide open and carried nearly nothing - which is the argument
for closing it, not for leaving it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import forecast_time_region_weights  # noqa: E402
from presidential_issue_engine import third_share_dispersion_expansion  # noqa: E402
from scripts import run_active_presidential_model_v24 as v24  # noqa: E402
from scripts import run_active_presidential_model_v27 as v27  # noqa: E402
from scripts import run_active_presidential_model_v28 as v28  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v30"
FINAL_VARIANT = "v30_forecast_time_weighted_dispersion"
WEIGHT_COLUMN = forecast_time_region_weights.WEIGHT_COLUMN


def _rename_error_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Make err_pp mean the shipped model's error, and name the baseline honestly.

    `official_pred` was the pre-layer baseline, renamed from `pred` in
    evaluate_electorate_layers - but the name reads as "the official
    prediction", and `err_pp` / `abs_err_pp` were computed from it. Every one of
    the 232 rows differed from the shipped `layer_pred`, by 5.89 percentage
    points on average, so anyone checking the published figure against the
    artifact's own error columns got 6.35 where the headline says 2.57.

    The numbers were never wrong for what they measured. The names were wrong
    about what that was.
    """

    out = predictions.copy()
    if "official_pred" in out.columns:
        out = out.rename(
            columns={
                "official_pred": "baseline_pre_layer_pred",
                "err_pp": "baseline_pre_layer_err_pp",
                "abs_err_pp": "baseline_pre_layer_abs_err_pp",
            }
        )
    out["err_pp"] = (out["layer_pred"] - out["actual"]) * 100.0
    out["abs_err_pp"] = out["err_pp"].abs()
    return out



def run(output_dir: Path | None = None, gain: float | None = None) -> Path:
    """Build V30. A non-default gain may not be written into the promoted directory."""

    selected = third_share_dispersion_expansion.DEFAULT_GAIN if gain is None else float(gain)
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    if selected != third_share_dispersion_expansion.DEFAULT_GAIN and destination == DEFAULT_OUTPUT:
        raise SystemExit(
            f"refusing to write gain {selected} into the promoted directory "
            f"{DEFAULT_OUTPUT.relative_to(ROOT).as_posix()}; pass --output-dir"
        )
    destination.mkdir(parents=True, exist_ok=True)

    original_weight = v27.WEIGHT_COLUMN
    try:
        # the V27 transform runs deep inside the chain; this is where its weight
        # is chosen, and it is restored on the way out so V27 itself is untouched
        v27.WEIGHT_COLUMN = WEIGHT_COLUMN
        v28.run(output_dir=destination)
    finally:
        v27.WEIGHT_COLUMN = original_weight

    path = destination / "nested_predictions.csv"
    predictions = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if WEIGHT_COLUMN not in predictions.columns:
        predictions[WEIGHT_COLUMN] = forecast_time_region_weights.build(predictions)
    predictions["v28_pre_third_share_expansion_pred"] = predictions["layer_pred"]
    predictions, audit = third_share_dispersion_expansion.apply_third_share_dispersion_expansion(
        predictions, weight_column=WEIGHT_COLUMN, gain=selected
    )
    predictions = _rename_error_columns(predictions)
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
    payload["pre_v30_extension_metrics"] = payload.get("metrics")
    payload["metrics"] = summary
    payload["policy_version"] = "active_v30_forecast_time_weighted_dispersion"
    payload["predecessor"] = "v29"
    payload["forecast_time_region_weights"] = {
        "weight_column": WEIGHT_COLUMN,
        "rule": "previous scored election's regional volumes; equal regions for the first",
        "replaces": "contest_votes, the target election's own turnout",
        "applies_to": [
            "party_regionalism_dispersion",
            "third_share_dispersion_expansion",
        ],
        "target_election_outcome_fields_used": [],
    }
    payload["third_share_dispersion_expansion"] = {
        "gain": selected,
        "gain_selection": (
            "parameter_free_unit_gain_not_swept"
            if selected == third_share_dispersion_expansion.DEFAULT_GAIN
            else f"explicitly_requested_non_promoted_gain_{selected}"
        ),
        "index": "model_predicted_third_placed_national_level",
        "outcome_fields_used": [],
    }
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
        help="expansion gain; defaults to the promoted 1.0 and otherwise requires --output-dir",
    )
    args = parser.parse_args()
    destination = run(args.output_dir, gain=args.gain)
    print(v24.report(destination).to_string(index=False))


if __name__ == "__main__":
    main()
