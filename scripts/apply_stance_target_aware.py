"""Apply the frozen target-aware v8 classifier to the 5,000-row corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.apply_stance_precision_ensemble import (  # noqa: E402
    DEFAULT_INPUT,
    validate_shadow_corpus,
)
from scripts.train_stance_target_aware import (  # noqa: E402
    DirectPolicy,
    _apply,
    _inputs,
    _ordered_probabilities,
)


DEFAULT_ARTIFACT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_target_aware_v8"
    / "stance_target_aware_v8.joblib"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_target_aware_v8"
    / "application_5000"
)
TARGET_TYPES = {"person", "party", "government"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.input.resolve(), encoding="utf-8-sig").fillna("")
    validate_shadow_corpus(frame)
    artifact = joblib.load(args.artifact.resolve())
    if artifact.get("active_forecast_integration") is not False:
        raise ValueError("artifact is not marked shadow-only")
    matrix = artifact["features"].transform(_inputs(frame))
    probabilities = _ordered_probabilities(artifact["model"], matrix)
    policy = DirectPolicy(**artifact["policy"])
    prediction = _apply(
        probabilities, frame["text_excerpt"].astype(str).to_numpy(), policy
    )
    prediction[~frame["target_type"].isin(TARGET_TYPES).to_numpy()] = "neutral"
    output = frame.copy()
    output["probability_negative"] = probabilities[:, 0]
    output["probability_neutral"] = probabilities[:, 1]
    output["probability_positive"] = probabilities[:, 2]
    output["target_aware_prediction"] = prediction
    output["target_aware_confidence"] = probabilities.max(axis=1)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "target_aware_predictions_5000.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    directional = output.loc[output["target_aware_prediction"].ne("neutral")]
    state = {
        "status": "shadow_application_complete",
        "active_forecast_changed": False,
        "rows": len(output),
        "directional_rows": len(directional),
        "directional_rate": len(directional) / max(len(output), 1),
        "prediction_counts": output["target_aware_prediction"].value_counts().to_dict(),
        "directional_by_target": directional["target_type"].value_counts().to_dict(),
        "directional_by_election": directional["election_id"].value_counts().sort_index().to_dict(),
        "output": str(output_path),
    }
    (output_dir / "application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
