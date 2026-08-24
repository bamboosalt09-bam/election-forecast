"""Run the outcome-free 2025 demonstration through the promoted V30 layer."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import third_share_dispersion_expansion  # noqa: E402
from scripts import run_prospective_forecast as base  # noqa: E402
from scripts import run_prospective_forecast_v28 as v28  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "prospective_pres_2025_v30"
# The scored panel weights each candidate's national level by that election's
# own turnout. A forecast cannot, so the previous election's regional volumes
# stand in - the same substitution V27 makes for the same reason.
WEIGHT_COLUMN = "v30_prior_election_vote_weight"


def run() -> Path:
    original_output = v28.OUTPUT_DIR
    try:
        v28.OUTPUT_DIR = OUTPUT_DIR
        v28.run()
    finally:
        v28.OUTPUT_DIR = original_output
    _expand_in_place()
    return OUTPUT_DIR


def _expand_in_place() -> None:
    stage = pd.read_csv(OUTPUT_DIR / "prediction_stage_audit.csv", low_memory=False)
    weights = base._prior_region_volume("v25")
    stage[WEIGHT_COLUMN] = stage["region_id"].astype(str).map(weights).fillna(0.0)
    stage["v28_pre_third_share_expansion_pred"] = stage["layer_pred"]
    stage, audit = third_share_dispersion_expansion.apply_third_share_dispersion_expansion(
        stage, prediction_column="layer_pred", weight_column=WEIGHT_COLUMN
    )
    stage["predicted_share"] = stage["layer_pred"]

    predictions = pd.read_csv(
        OUTPUT_DIR / "prospective_predictions.csv", encoding="utf-8-sig"
    )
    keyed = stage.set_index(["region_id", "candidate_name"])["predicted_share"]
    predictions["predicted_share"] = pd.MultiIndex.from_arrays(
        [predictions["region_id"], predictions["candidate_name"]]
    ).map(keyed)
    if predictions["predicted_share"].isna().any():
        raise ValueError("the expansion did not cover every forecast row")

    national = base._national_summary(predictions, "v25")
    predictions.to_csv(
        OUTPUT_DIR / "prospective_predictions.csv", index=False, encoding="utf-8-sig"
    )
    national.to_csv(
        OUTPUT_DIR / "national_summary.csv", index=False, encoding="utf-8-sig"
    )
    stage.to_csv(
        OUTPUT_DIR / "prediction_stage_audit.csv", index=False, encoding="utf-8-sig"
    )
    audit.to_csv(
        OUTPUT_DIR / "third_share_dispersion_expansion_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_path = OUTPUT_DIR / "input_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    added = [
        ROOT / "presidential_issue_engine/third_share_dispersion_expansion.py",
        ROOT / "scripts/run_prospective_forecast_v30.py",
        ROOT / "scripts/run_active_presidential_model_v30.py",
    ]
    manifest = (
        pd.concat(
            [
                manifest,
                pd.DataFrame(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": base._sha256(path),
                    }
                    for path in added
                ),
            ],
            ignore_index=True,
        )
        .drop_duplicates("path", keep="last")
        .sort_values("path")
    )
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    run_manifest_path = OUTPUT_DIR / "run_manifest.json"
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    payload["version"] = "v30"
    payload["predecessor_runtime"] = "v28"
    payload["v30_third_share_dispersion_expansion"] = {
        "gain": third_share_dispersion_expansion.DEFAULT_GAIN,
        "index": "model_predicted_third_placed_national_level",
        "weight_source": "pres_2022_valid_vote_volume",
        "gain_selection": "parameter_free_unit_gain_not_swept",
        "outcome_fields_used": [],
    }
    payload["model_parameters_changed"] = False
    payload["performance_metrics_computed"] = False
    run_manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    print(run().relative_to(ROOT).as_posix())
