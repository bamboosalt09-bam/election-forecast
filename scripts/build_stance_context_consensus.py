"""Build a conservative majority consensus from three shadow classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V8 = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_target_aware_v8"
    / "application_5000"
    / "target_aware_predictions_5000.csv"
)
DEFAULT_KLUE = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_klue_context_v9"
    / "application_5000"
    / "context_predictions_5000.csv"
)
DEFAULT_NLI = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_ko_nli_context_v10"
    / "application_5000"
    / "context_predictions_5000.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_consensus_v11"
    / "application_5000"
)


def majority_consensus(predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("predictions must have exactly three columns")
    negative = (predictions == "negative").sum(axis=1)
    positive = (predictions == "positive").sum(axis=1)
    output = np.full(len(predictions), "neutral", dtype="<U8")
    output[negative >= 2] = "negative"
    output[positive >= 2] = "positive"
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8", type=Path, default=DEFAULT_V8)
    parser.add_argument("--klue", type=Path, default=DEFAULT_KLUE)
    parser.add_argument("--nli", type=Path, default=DEFAULT_NLI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    v8 = pd.read_csv(args.v8.resolve(), encoding="utf-8-sig").fillna("")
    klue = pd.read_csv(args.klue.resolve(), encoding="utf-8-sig").fillna("")
    nli = pd.read_csv(args.nli.resolve(), encoding="utf-8-sig").fillna("")
    output = v8.copy()
    output = output.merge(
        klue[
            [
                "text_sha256",
                "context_prediction",
                "context_confidence",
            ]
        ].rename(
            columns={
                "context_prediction": "klue_prediction",
                "context_confidence": "klue_confidence",
            }
        ),
        on="text_sha256",
        validate="one_to_one",
    )
    output = output.merge(
        nli[
            [
                "text_sha256",
                "context_prediction",
                "context_confidence",
            ]
        ].rename(
            columns={
                "context_prediction": "nli_prediction",
                "context_confidence": "nli_confidence",
            }
        ),
        on="text_sha256",
        validate="one_to_one",
    )
    children = output[
        ["target_aware_prediction", "klue_prediction", "nli_prediction"]
    ].to_numpy(dtype=str)
    output["context_consensus_prediction"] = majority_consensus(children)
    output["context_consensus_votes"] = np.maximum(
        (children == "negative").sum(axis=1),
        (children == "positive").sum(axis=1),
    )
    output["context_consensus_confidence"] = output[
        ["target_aware_confidence", "klue_confidence", "nli_confidence"]
    ].median(axis=1)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "context_consensus_predictions_5000.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    directional = output.loc[
        output["context_consensus_prediction"].ne("neutral")
    ]
    conflicts = (
        ((children == "negative").sum(axis=1) > 0)
        & ((children == "positive").sum(axis=1) > 0)
    )
    state = {
        "status": "shadow_consensus_complete",
        "model_version": "stance_context_consensus_v11",
        "active_forecast_changed": False,
        "rows": len(output),
        "directional_rows": len(directional),
        "prediction_counts": output["context_consensus_prediction"].value_counts().to_dict(),
        "child_directional_conflicts": int(conflicts.sum()),
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
