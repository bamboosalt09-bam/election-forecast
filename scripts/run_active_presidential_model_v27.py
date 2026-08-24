"""Run V27: V26 plus core-weighted inherited regional dispersion."""

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
from presidential_issue_engine import party_regionalism_dispersion  # noqa: E402
from scripts import run_active_presidential_model_v24 as v24  # noqa: E402
from scripts import run_active_presidential_model_v26 as v26  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs" / "active_presidential_nested_v27"
FINAL_VARIANT = "v27_core_weighted_party_regional_dispersion"


#: Which column weights each candidate's national level in the terminal
#: transform. V27 froze this as contest_votes - the target election's own
#: turnout, which exists only after the count. V30 overrides it with
#: forecast-time weights; V27 itself stays as it was frozen.
WEIGHT_COLUMN = "contest_votes"


def run(output_dir: Path | None = None) -> Path:
    destination = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    v26.run(output_dir=destination)
    path = destination / "nested_predictions.csv"
    predictions = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    predictions["v26_pre_regional_polarization_pred"] = predictions["layer_pred"]
    if WEIGHT_COLUMN not in predictions.columns:
        predictions[WEIGHT_COLUMN] = forecast_time_region_weights.build(predictions)
    predictions, audit = party_regionalism_dispersion.apply_party_regionalism_dispersion(
        predictions, weight_column=WEIGHT_COLUMN,
        gain=party_regionalism_dispersion.DEFAULT_GAIN
    )
    v24._atomic_csv_crlf(predictions, path)
    v24._atomic_csv_crlf(audit, destination / "party_regionalism_dispersion_audit.csv")

    from scripts import run_active_presidential_model as active
    summary, by_election, national = active.nested._metrics(
        predictions, "layer_pred", FINAL_VARIANT
    )
    v24._atomic_csv_crlf(by_election, destination / "by_election.csv")
    v24._atomic_csv_crlf(national, destination / "national_predictions.csv")
    payload = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    payload["pre_v27_extension_metrics"] = payload.get("metrics")
    payload["metrics"] = summary
    payload["policy_version"] = "active_v27_core_weighted_party_regional_dispersion"
    payload["predecessor"] = "v26"
    payload["party_regionalism_dispersion"] = {
        "gain": party_regionalism_dispersion.DEFAULT_GAIN,
        "prior_width": "recent_bloc_base_vote_weighted_logit_sd",
        "retained_share": "core_voting_mass_times_direct_party_reliability",
        "candidate_national_level_preserved": True,
        "regional_composition_preserved": True,
        "outcome_fields_used": [],
    }
    payload["post_2022_outcomes_used"] = False
    v24._atomic_json_crlf(payload, destination / "summary.json")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    destination = run(args.output_dir)
    print(v24.report(destination).to_string(index=False))


if __name__ == "__main__":
    main()
