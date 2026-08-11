"""Apply a frozen shadow context encoder to the through-2022 corpus."""

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
from scripts.train_stance_context_encoder import (  # noqa: E402
    DirectPolicy,
    _apply,
    _inputs,
    _ordered_probabilities,
    encode_texts,
)


TARGET_TYPES = {"person", "party", "government"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.input.resolve(), encoding="utf-8-sig").fillna("")
    validate_shadow_corpus(frame)
    artifact = joblib.load(args.artifact.resolve())
    if artifact.get("active_forecast_integration") is not False:
        raise ValueError("artifact is not marked shadow-only")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache = output_dir / "application_embeddings.npz"
    if embedding_cache.exists():
        matrix = np.load(embedding_cache)["embeddings"]
        if len(matrix) != len(frame):
            raise ValueError("application embedding cache row count does not match input")
    else:
        matrix = encode_texts(
            _inputs(frame),
            artifact["encoder"],
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
        np.savez_compressed(embedding_cache, embeddings=matrix)
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
    output["context_prediction"] = prediction
    output["context_confidence"] = probabilities.max(axis=1)
    output_path = output_dir / "context_predictions_5000.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    directional = output.loc[output["context_prediction"].ne("neutral")]
    state = {
        "status": "shadow_application_complete",
        "model_version": artifact["model_version"],
        "active_forecast_changed": False,
        "rows": len(output),
        "directional_rows": len(directional),
        "directional_rate": len(directional) / max(len(output), 1),
        "prediction_counts": output["context_prediction"].value_counts().to_dict(),
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
