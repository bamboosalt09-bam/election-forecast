"""V31: expand the regional dispersion multiplicatively, so nothing hits zero.

V29 added an additive expansion of each candidate's regional deviations around
that candidate's own national level, and capped the factor per election so no
share went negative. V30 changed the weight that locates the level. Both stand.

What neither noticed is what the cap does when it binds. It is defined as the
factor at which some region reaches zero, so the region that sets the cap lands
on exactly zero every time - not as an estimate, but as the point where the
arithmetic stopped. On the scored panel that is 홍준표's 광주 in 2017, where the
stage feeding the transform says 3.55% and the realised share is 1.68%. In the
published 2025 demonstration it is 김문수's 광주: 2.67% in, 0.00% out, with the
displaced mass landing on the other two candidates in that region.

V31 replaces the additive expansion with a multiplicative one::

    scaled = level * (pred / level) ** factor

which cannot reach zero from a positive input, so the cap is not needed and is
gone. The multiplicative form does not conserve the weighted national level on
its own, so the two constraints - regional sums of one, and each candidate's
input level - are alternated to convergence. Neither step introduces a
constant; the factor and its gain are V29's.

Everything else is V30's: the Ridge stack, the predictors, the shock structure,
the V28 external-model boundary, V27's regional transform, and the forecast-time
weighting.

Measured before the decision, on the scored panel:

    regional macro   2.566445 -> 2.500701   (-0.065744)
    national macro   0.720437 -> 0.724291   (+0.003854)
    2017 광주        0.0001%  -> 1.9688%    (realised 1.6800%)
    2017 전남        0.8470%  -> 2.5013%    (realised 2.6360%)
    feasibility cap  binds in 2017          -> never binds; removed

The national figure gets slightly worse and that is not a reason to decline. A
prediction of exactly zero for a major-party candidate in a metropolitan region
is wrong in kind, not in degree, and this version would have been taken had
both figures moved against it.

This runner also refuses two conditions the V30 chain tolerated. V30's shared
weights module falls back to equal regions when the 1997 warmup table is absent
and picks one value when an election-region's turnout disagrees across its
candidate rows. Both are frozen into V30 by hash and cannot be changed there, so
the checks live here instead: a missing warmup table or an inconsistent volume
stops this version rather than quietly running a different one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import forecast_time_region_weights  # noqa: E402
from presidential_issue_engine import multiplicative_dispersion_expansion as expansion  # noqa: E402
from scripts import run_active_presidential_model_v24 as v24  # noqa: E402
from scripts import run_active_presidential_model_v27 as v27  # noqa: E402
from scripts import run_active_presidential_model_v28 as v28  # noqa: E402
from scripts import run_active_presidential_model_v30 as v30  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v31"
FINAL_VARIANT = "v31_multiplicative_dispersion_expansion"
WEIGHT_COLUMN = forecast_time_region_weights.WEIGHT_COLUMN
PRE_EXPANSION_COLUMN = "v30_pre_multiplicative_expansion_pred"


def require_forecast_time_inputs(predictions: pd.DataFrame) -> None:
    """Fail rather than silently weighting a different way.

    The shared weights module is permissive by necessity - it is frozen into
    V30 - so this version states its own preconditions. Both are conditions the
    repository never actually reaches; the point is that reaching them stops
    the run instead of producing an artifact that looks ordinary.
    """

    warmup = forecast_time_region_weights.WARMUP_TURNOUT
    if not warmup.is_file():
        # not relative_to(ROOT): the path is whatever the module points at, and
        # an error path that raises its own exception reports nothing
        raise FileNotFoundError(
            f"the warmup turnout table {warmup} is missing; it is part of what "
            "this version is, and equal-region weights would be a different model"
        )
    table = pd.read_csv(warmup, encoding="utf-8-sig")
    for election in sorted(set(forecast_time_region_weights.WARMUP_PREDECESSOR.values())):
        if table["election_id"].astype(str).eq(election).sum() == 0:
            raise ValueError(f"the warmup table carries no rows for {election}")

    spread = predictions.groupby(["election_id", "region_id"])["contest_votes"].nunique()
    conflicting = spread.loc[spread > 1]
    if not conflicting.empty:
        election, region = conflicting.index[0]
        raise ValueError(
            f"contest_votes disagrees within {election} {region} across its "
            f"candidate rows ({int(conflicting.iloc[0])} distinct values)"
        )


def run(output_dir: Path | None = None, gain: float | None = None) -> Path:
    """Build V31. A non-default gain may not be written into the promoted directory."""

    selected = expansion.DEFAULT_GAIN if gain is None else float(gain)
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    if selected != expansion.DEFAULT_GAIN and destination == DEFAULT_OUTPUT:
        raise SystemExit(
            f"refusing to write gain {selected} into the promoted directory "
            f"{DEFAULT_OUTPUT.relative_to(ROOT).as_posix()}; pass --output-dir"
        )
    destination.mkdir(parents=True, exist_ok=True)

    original_weight = v27.WEIGHT_COLUMN
    try:
        # V27's transform runs deep inside the chain; this is where its weight is
        # chosen, and it is restored on the way out so V27 itself is untouched.
        v27.WEIGHT_COLUMN = WEIGHT_COLUMN
        v28.run(output_dir=destination)
    finally:
        v27.WEIGHT_COLUMN = original_weight

    path = destination / "nested_predictions.csv"
    predictions = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    require_forecast_time_inputs(predictions)
    if WEIGHT_COLUMN not in predictions.columns:
        predictions[WEIGHT_COLUMN] = forecast_time_region_weights.build(predictions)
    predictions[PRE_EXPANSION_COLUMN] = predictions["layer_pred"]
    predictions, audit = expansion.apply_multiplicative_dispersion_expansion(
        predictions, weight_column=WEIGHT_COLUMN, gain=selected
    )
    predictions = v30._rename_error_columns(predictions)
    v24._atomic_csv_crlf(predictions, path)
    v24._atomic_csv_crlf(audit, destination / "multiplicative_dispersion_expansion_audit.csv")

    from scripts import run_active_presidential_model as active

    summary, by_election, national = active.nested._metrics(
        predictions, "layer_pred", FINAL_VARIANT
    )
    v24._atomic_csv_crlf(by_election, destination / "by_election.csv")
    v24._atomic_csv_crlf(national, destination / "national_predictions.csv")

    summary_path = destination / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["pre_v31_extension_metrics"] = payload.get("metrics")
    payload["metrics"] = summary
    payload["policy_version"] = "active_v31_multiplicative_dispersion_expansion"
    payload["predecessor"] = "v30"
    payload["forecast_time_region_weights"] = {
        "weight_column": WEIGHT_COLUMN,
        # V30's runner published "equal regions for the first", which its own
        # code had already stopped doing once 1997 was sourced. Stated correctly
        # here; the frozen V30 record keeps its wording and the error is noted
        # in the V31 experiment document.
        "rule": (
            "previous scored election's regional volumes; the first scored election "
            "uses the 1997 warmup table"
        ),
        "warmup_table": (
            "presidential_issue_engine/fixed_dataset/pres_1997_regional_turnout.csv"
        ),
        "missing_predecessor_behaviour": "raise",
        "replaces": "contest_votes, the target election's own turnout",
        "applies_to": [
            "party_regionalism_dispersion",
            "multiplicative_dispersion_expansion",
        ],
        "target_election_outcome_fields_used": [],
    }
    payload["multiplicative_dispersion_expansion"] = {
        "form": "level * (pred / level) ** (1 + gain * predicted_third_share)",
        "gain": selected,
        "gain_selection": (
            "parameter_free_unit_gain_not_swept"
            if selected == expansion.DEFAULT_GAIN
            else f"explicitly_requested_non_promoted_gain_{selected}"
        ),
        "index": "model_predicted_third_placed_national_level",
        "feasibility_cap": "not required; the form cannot reach zero",
        "level_reconciliation": "alternating regional sums and candidate levels to convergence",
        "replaces": "v29 additive expansion with a per-election zero cap",
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
