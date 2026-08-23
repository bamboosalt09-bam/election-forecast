"""Run the outcome-free 2025 demonstration through the promoted V27 layer."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import party_regionalism_dispersion  # noqa: E402
from scripts import run_prospective_forecast as base  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "prospective_pres_2025_v27"


#: Overridden by the V28 wrapper, which runs inside the external-model
#: boundary and therefore cannot reproduce a V25 history frozen before it.
CANONICAL_HISTORY_DIR: Path | None = None


def run() -> Path:
    # V26 and V25 are identical for the already-saturated 2025 crisis target;
    # V26's historical ladder difference is separately pinned by V27 rollback.
    with tempfile.TemporaryDirectory(prefix="prospective_v27_base_") as temporary:
        source_dir = base.run(
            "v25",
            output_dir_override=Path(temporary),
            canonical_dir=CANONICAL_HISTORY_DIR,
        )
        _promote_from_source(source_dir)
    return OUTPUT_DIR


def _promote_from_source(source_dir: Path) -> None:
    stage = pd.read_csv(source_dir / "prediction_stage_audit.csv", low_memory=False)
    target = stage.loc[stage["election_id"].astype(str).eq(base.TARGET_ELECTION)].copy()
    weights = base._prior_region_volume("v25")
    target["v27_prior_election_vote_weight"] = (
        target["region_id"].astype(str).map(weights).fillna(0.0)
    )
    target["v26_pre_regional_polarization_pred"] = target["layer_pred"]
    adjusted, audit = party_regionalism_dispersion.apply_party_regionalism_dispersion(
        target,
        prediction_column="layer_pred",
        weight_column="v27_prior_election_vote_weight",
    )
    adjusted["predicted_share"] = adjusted["layer_pred"]
    adjusted = adjusted.drop(columns=["slot"], errors="ignore").rename(
        columns={"source_slot": "slot"}
    )
    name = "candidate_name_x" if "candidate_name_x" in adjusted.columns else "candidate_name"
    adjusted = adjusted.rename(columns={name: "candidate_name"})
    adjusted = adjusted.loc[:, ~adjusted.columns.duplicated()]
    predictions = adjusted[list(base.OUTPUT_COLUMNS)].copy()
    national = base._national_summary(predictions, "v25")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "prospective_predictions.csv", index=False, encoding="utf-8-sig")
    national.to_csv(OUTPUT_DIR / "national_summary.csv", index=False, encoding="utf-8-sig")
    adjusted.to_csv(OUTPUT_DIR / "prediction_stage_audit.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUTPUT_DIR / "party_regionalism_dispersion_audit.csv", index=False, encoding="utf-8-sig")
    manifest = pd.read_csv(source_dir / "input_manifest.csv")
    v27_inputs = [
        ROOT / "presidential_issue_engine/party_regionalism_dispersion.py",
        ROOT / "scripts/run_prospective_forecast_v27.py",
        ROOT / "scripts/run_active_presidential_model_v27.py",
    ]
    manifest = pd.concat(
        [
            manifest,
            pd.DataFrame(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": base._sha256(path),
                }
                for path in v27_inputs
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("path", keep="last").sort_values("path")
    manifest.to_csv(OUTPUT_DIR / "input_manifest.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "target_feature_audit.csv").write_bytes(
        (source_dir / "target_feature_audit.csv").read_bytes()
    )
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
    source_manifest.update({
        "version": "v27",
        "predecessor_runtime": "v26",
        "v27_party_regionalism_dispersion": {
            "gain": party_regionalism_dispersion.DEFAULT_GAIN,
            "weight_source": "pres_2022_valid_vote_volume",
            "outcome_fields_used": [],
            "module_sha256": base._sha256(
                ROOT / "presidential_issue_engine/party_regionalism_dispersion.py"
            ),
        },
        "model_parameters_changed": False,
        "performance_metrics_computed": False,
    })
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    print(run().relative_to(ROOT).as_posix())
