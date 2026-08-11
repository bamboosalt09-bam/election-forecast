"""Apply the frozen shadow encoder with resumable embedding checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.apply_stance_precision_ensemble import validate_shadow_corpus  # noqa: E402
from scripts.train_stance_context_encoder import (  # noqa: E402
    DirectPolicy,
    _apply,
    _inputs,
    _ordered_probabilities,
    encode_texts,
)


TARGET_TYPES = {"person", "party", "government"}


def _chunk_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in frame["text_sha256"].astype(str):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    frame = pd.read_csv(source, encoding="utf-8-sig", low_memory=False).fillna("")
    validate_shadow_corpus(frame)
    artifact = joblib.load(args.artifact.resolve())
    if artifact.get("active_forecast_integration") is not False:
        raise ValueError("artifact is not marked shadow-only")

    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "embedding_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "context_predictions.csv"
    if output_path.exists():
        raise FileExistsError(output_path)

    matrices: list[np.ndarray] = []
    chunk_records: list[dict[str, object]] = []
    for chunk_index, start in enumerate(range(0, len(frame), args.chunk_size)):
        stop = min(start + args.chunk_size, len(frame))
        chunk = frame.iloc[start:stop]
        expected_hash = _chunk_hash(chunk)
        cache_path = cache_dir / f"embeddings_{chunk_index:04d}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            matrix = cached["embeddings"]
            cached_hash = str(cached["text_hash"].item())
            if len(matrix) != len(chunk) or cached_hash != expected_hash:
                raise ValueError(f"invalid embedding checkpoint: {cache_path}")
            status = "reused"
        else:
            matrix = encode_texts(
                _inputs(chunk),
                artifact["encoder"],
                batch_size=args.batch_size,
                local_files_only=args.local_files_only,
            )
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    embeddings=matrix,
                    text_hash=np.asarray(expected_hash),
                )
            temporary.replace(cache_path)
            status = "computed"
        matrices.append(matrix)
        chunk_records.append(
            {
                "chunk": chunk_index,
                "start": start,
                "stop": stop,
                "rows": len(chunk),
                "sha256": expected_hash,
                "status": status,
            }
        )
        (output_dir / "checkpoint_state.json").write_text(
            json.dumps(
                {
                    "status": "embedding_in_progress",
                    "input": str(source),
                    "rows": len(frame),
                    "completed_rows": stop,
                    "chunks": chunk_records,
                    "active_forecast_changed": False,
                    "vote_outcomes_used": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[chunked encoder] {stop:,}/{len(frame):,} rows ({status})", flush=True)

    matrix = np.concatenate(matrices, axis=0)
    probabilities = _ordered_probabilities(artifact["model"], matrix)
    policy = DirectPolicy(**artifact["policy"])
    prediction = _apply(probabilities, frame["text_excerpt"].astype(str).to_numpy(), policy)
    prediction[~frame["target_type"].isin(TARGET_TYPES).to_numpy()] = "neutral"

    output = frame.copy()
    output["probability_negative"] = probabilities[:, 0]
    output["probability_neutral"] = probabilities[:, 1]
    output["probability_positive"] = probabilities[:, 2]
    output["context_prediction"] = prediction
    output["context_confidence"] = probabilities.max(axis=1)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    directional = output.loc[output["context_prediction"].ne("neutral")]
    state = {
        "status": "shadow_application_complete",
        "model_version": artifact["model_version"],
        "active_forecast_changed": False,
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "rows": len(output),
        "directional_rows": len(directional),
        "directional_rate": len(directional) / max(len(output), 1),
        "prediction_counts": output["context_prediction"].value_counts().to_dict(),
        "directional_by_target": directional["target_type"].value_counts().to_dict(),
        "directional_by_election": directional["election_id"].value_counts().sort_index().to_dict(),
        "chunks": chunk_records,
        "output": str(output_path),
    }
    (output_dir / "application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
