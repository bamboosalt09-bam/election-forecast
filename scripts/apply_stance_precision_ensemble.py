"""Apply the frozen precision-first stance ensemble to a through-2022 corpus.

This is a shadow inference path. It writes auditable sentence-level output and
never modifies active forecast inputs.
"""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import (  # noqa: E402
    ConsensusPolicy,
    KoreanPrecisionStructureFeatures,
    PrecisionPolicy,
    apply_precision_policy,
    combine_precision_children,
    compose_embedding_input,
    compose_precision_input,
    neutral_information_features,
)
from election_forecast.context_corpus import (  # noqa: E402
    ELECTION_CUTOFFS,
    outcome_like_columns,
)
from scripts.train_stance_precision_first import (  # noqa: E402
    ALLOWED_ELECTIONS,
    _predict_heads as predict_tfidf_heads,
)


DEFAULT_INPUT = ROOT / "data" / "shadow" / "stance_context_5000_through2022.csv"
DEFAULT_TFIDF = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "precision_augmented_v3"
    / "stance_precision_augmented_v3.joblib"
)
DEFAULT_EMBEDDING = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "precision_embedding_v1"
    / "stance_precision_embedding_v1.joblib"
)
DEFAULT_ENSEMBLE_METRICS = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "precision_augmented_ensemble_v4"
    / "metrics.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "precision_augmented_ensemble_v4"
    / "application_5000"
)
REQUIRED_COLUMNS = {
    "election_id",
    "text_excerpt",
    "text_sha256",
    "target_type",
    "target_name",
    "meeting_date",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shadow_corpus(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"shadow corpus is missing required columns: {missing}")
    forbidden_elections = sorted(
        set(frame["election_id"].astype(str)).difference(ALLOWED_ELECTIONS)
    )
    if forbidden_elections:
        raise ValueError(
            f"shadow corpus contains forbidden elections: {forbidden_elections}"
        )
    forbidden_columns = sorted(outcome_like_columns(frame.columns))
    if forbidden_columns:
        raise ValueError(
            f"shadow corpus contains outcome-like columns: {forbidden_columns}"
        )
    if frame["text_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").eq(False).any():
        raise ValueError("shadow corpus contains invalid text hashes")
    if frame["text_sha256"].duplicated().any():
        raise ValueError("shadow corpus contains duplicate text hashes")
    dates = pd.to_datetime(frame["meeting_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("shadow corpus contains invalid meeting_date values")
    cutoffs = frame["election_id"].astype(str).map(ELECTION_CUTOFFS)
    if (dates > cutoffs).any():
        raise ValueError("shadow corpus contains post-cutoff meeting rows")


def _validate_artifact(artifact: dict[str, object], label: str) -> None:
    if bool(artifact.get("active_forecast_integration", True)):
        raise ValueError(f"{label} artifact is not marked shadow-only")
    allowed = set(str(value) for value in artifact.get("allowed_elections", []))
    if allowed != set(ALLOWED_ELECTIONS):
        raise ValueError(f"{label} artifact election boundary does not match")


def _tfidf_predictions(
    artifact: dict[str, object], frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = pd.Series(
        [
            compose_precision_input(row, str(artifact["representation"]))
            for row in frame.to_dict(orient="records")
        ],
        dtype=str,
    )
    matrix = artifact["features"].transform(inputs)
    direction, polarity = predict_tfidf_heads(
        artifact["direction_model"], artifact["polarity_model"], matrix
    )
    prediction = apply_precision_policy(
        direction,
        polarity,
        frame["text_excerpt"].astype(str).to_numpy(),
        PrecisionPolicy(**artifact["policy"]),
    )
    return direction, polarity, prediction


def _embedding_predictions(
    artifact: dict[str, object], frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "embedding inference requires the isolated .venv-stance environment"
        ) from exc

    snapshot = snapshot_download(
        str(artifact["encoder_model_id"]),
        revision=str(artifact["encoder_revision"]),
        local_files_only=True,
    )
    model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True)
    inputs = [
        compose_embedding_input(row, str(artifact["representation"]))
        for row in frame.to_dict(orient="records")
    ]
    embeddings = np.asarray(
        model.encode(
            inputs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float64,
    )
    variant = str(artifact["feature_variant"])
    if variant == "embedding":
        matrix = embeddings
    elif variant == "embedding_structure":
        structure = KoreanPrecisionStructureFeatures().transform(inputs).toarray()
        matrix = np.column_stack([embeddings, structure])
    else:
        raise ValueError(f"unsupported embedding feature variant: {variant}")
    matrix = artifact["scaler"].transform(matrix)
    raw_direction = artifact["direction_model"].predict_proba(matrix)
    direction_position = list(artifact["direction_model"].classes_).index(1)
    direction = raw_direction[:, direction_position]
    raw_polarity = artifact["polarity_model"].predict_proba(matrix)
    positions = {
        str(value): index
        for index, value in enumerate(artifact["polarity_model"].classes_)
    }
    polarity = np.column_stack(
        [
            raw_polarity[:, positions["negative"]],
            raw_polarity[:, positions["positive"]],
        ]
    )
    prediction = apply_precision_policy(
        direction,
        polarity,
        frame["text_excerpt"].astype(str).to_numpy(),
        PrecisionPolicy(**artifact["policy"]),
    )
    return direction, polarity, prediction


def _summary_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("").astype(str).value_counts().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tfidf-artifact", type=Path, default=DEFAULT_TFIDF)
    parser.add_argument("--embedding-artifact", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--ensemble-metrics", type=Path, default=DEFAULT_ENSEMBLE_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.input.resolve()
    tfidf_path = args.tfidf_artifact.resolve()
    embedding_path = args.embedding_artifact.resolve()
    ensemble_metrics_path = args.ensemble_metrics.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path, encoding="utf-8-sig").fillna("")
    validate_shadow_corpus(frame)
    tfidf_artifact = joblib.load(tfidf_path)
    embedding_artifact = joblib.load(embedding_path)
    _validate_artifact(tfidf_artifact, "TF-IDF")
    _validate_artifact(embedding_artifact, "embedding")
    ensemble_metrics = json.loads(ensemble_metrics_path.read_text(encoding="utf-8"))
    if ensemble_metrics.get("adoption", {}).get("active_forecast_changed") is not False:
        raise ValueError("ensemble metrics are not marked shadow-only")
    policy = ConsensusPolicy(**ensemble_metrics["selection"]["policy"])

    first_direction, first_polarity, first_prediction = _tfidf_predictions(
        tfidf_artifact, frame
    )
    second_direction, second_polarity, second_prediction = _embedding_predictions(
        embedding_artifact, frame
    )
    prediction, source = combine_precision_children(
        first_prediction,
        second_prediction,
        first_direction,
        second_direction,
        first_polarity,
        second_polarity,
        frame["text_excerpt"].astype(str).to_numpy(),
        policy,
    )

    output = frame.copy()
    output["tfidf_direction_probability"] = first_direction
    output["tfidf_polarity_negative"] = first_polarity[:, 0]
    output["tfidf_polarity_positive"] = first_polarity[:, 1]
    output["tfidf_prediction"] = first_prediction
    output["embedding_direction_probability"] = second_direction
    output["embedding_polarity_negative"] = second_polarity[:, 0]
    output["embedding_polarity_positive"] = second_polarity[:, 1]
    output["embedding_prediction"] = second_prediction
    output["ensemble_prediction"] = prediction
    output["ensemble_source"] = source
    output["ensemble_direction_score"] = np.sqrt(first_direction * second_direction)
    output["ensemble_polarity_score"] = np.minimum(
        first_polarity.max(axis=1), second_polarity.max(axis=1)
    )
    output["ensemble_sign_margin"] = np.minimum(
        np.abs(first_polarity[:, 1] - first_polarity[:, 0]),
        np.abs(second_polarity[:, 1] - second_polarity[:, 0]),
    )
    sign = np.where(prediction == "negative", -1.0, np.where(prediction == "positive", 1.0, 0.0))
    output["ensemble_signed_score"] = (
        sign * output["ensemble_direction_score"] * output["ensemble_polarity_score"]
    )
    information = pd.DataFrame(
        [neutral_information_features(text) for text in output["text_excerpt"]]
    )
    for column in information.columns:
        output[column] = information[column].to_numpy()
    output.loc[
        output["ensemble_prediction"].ne("neutral"), "neutral_information_score"
    ] = 0.0

    output_path = output_dir / "stance_precision_predictions_5000.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    directional = output.loc[output["ensemble_prediction"].ne("neutral")].copy()
    audit = directional.sort_values(
        ["ensemble_prediction", "ensemble_direction_score", "text_sha256"],
        ascending=[True, False, True],
    ).groupby(["election_id", "ensemble_prediction"], group_keys=False).head(10)
    audit.to_csv(
        output_dir / "directional_audit_sample.csv", index=False, encoding="utf-8-sig"
    )
    neutral = output.loc[output["ensemble_prediction"].eq("neutral")].nlargest(
        100, "neutral_information_score"
    )
    neutral.to_csv(
        output_dir / "high_information_neutral_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    state = {
        "status": "shadow_application_complete",
        "active_forecast_changed": False,
        "rows": int(len(output)),
        "unique_text_hashes": int(output["text_sha256"].nunique()),
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "tfidf_artifact_sha256": _sha256(tfidf_path),
        "embedding_artifact_sha256": _sha256(embedding_path),
        "ensemble_metrics_sha256": _sha256(ensemble_metrics_path),
        "allowed_elections": list(ALLOWED_ELECTIONS),
        "post_2022_rows_present": False,
        "outcome_columns_present": False,
        "prediction_counts": _summary_counts(output, "ensemble_prediction"),
        "prediction_source_counts": _summary_counts(output, "ensemble_source"),
        "target_type_counts": _summary_counts(output, "target_type"),
        "directional_rows": int(len(directional)),
        "directional_rate": float(len(directional) / max(len(output), 1)),
        "directional_by_election": {
            str(key): int(value)
            for key, value in directional["election_id"].value_counts().sort_index().items()
        },
        "directional_by_target_type": {
            str(key): int(value)
            for key, value in directional["target_type"].value_counts().items()
        },
        "neutral_information_nonzero": int(
            output["neutral_information_score"].gt(0).sum()
        ),
        "output": str(output_path),
    }
    (output_dir / "application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
