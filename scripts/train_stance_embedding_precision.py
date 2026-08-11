"""Evaluate a fixed Korean sentence embedding for precision-first stance.

Run this script with `.venv-stance`. The pretrained encoder is never fine-tuned;
only small logistic heads and conservative abstention thresholds are selected.
The output is shadow-only and cannot modify active forecast inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import (  # noqa: E402
    KoreanPrecisionStructureFeatures,
    PrecisionPolicy,
    apply_precision_policy,
    compose_embedding_input,
    neutral_information_features,
    precision_first_metrics,
    stance_adoption_assessment,
)
from scripts.train_stance_precision_first import (  # noqa: E402
    ALLOWED_ELECTIONS,
    RANDOM_STATE,
    _groups,
    _load_gold,
)


MODEL_ID = "jhgan/ko-sroberta-multitask"
DEFAULT_GOLD = ROOT / "data" / "shadow" / "stance_precision_gold_through2022.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "assembly_stance" / "precision_embedding_v1"
REPRESENTATIONS = ("current_only", "risk_aware_context")
FEATURE_VARIANTS = ("embedding", "embedding_structure")
C_VALUES = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs(frame: pd.DataFrame, mode: str) -> list[str]:
    return [compose_embedding_input(row, mode) for row in frame.to_dict(orient="records")]


def _encode(
    model: SentenceTransformer,
    frame: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    values = _inputs(frame, mode)
    return np.asarray(
        model.encode(
            values,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float64,
    )


def _features(
    embeddings: np.ndarray,
    texts: list[str],
    variant: str,
) -> np.ndarray:
    if variant == "embedding":
        return embeddings
    if variant == "embedding_structure":
        structure = KoreanPrecisionStructureFeatures().transform(texts).toarray()
        return np.column_stack([embeddings, structure])
    raise ValueError(f"unsupported feature variant: {variant}")


def _classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=5_000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def _fit_heads(matrix: np.ndarray, labels: np.ndarray, c_value: float):
    directional = (labels != "neutral").astype(int)
    direction = _classifier(c_value).fit(matrix, directional)
    mask = labels != "neutral"
    polarity = _classifier(c_value).fit(matrix[mask], labels[mask])
    return direction, polarity


def _predict_heads(direction, polarity, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_direction = direction.predict_proba(matrix)
    direction_position = list(direction.classes_).index(1)
    raw_polarity = polarity.predict_proba(matrix)
    positions = {str(value): index for index, value in enumerate(polarity.classes_)}
    return (
        raw_direction[:, direction_position],
        np.column_stack(
            [
                raw_polarity[:, positions["negative"]],
                raw_polarity[:, positions["positive"]],
            ]
        ),
    )


def _oof(
    development: pd.DataFrame,
    encoded: dict[str, np.ndarray],
) -> dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]]:
    labels = development["review_label"].astype(str).to_numpy()
    groups = _groups(development)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    stores = {
        (mode, variant, c_value): (
            np.zeros(len(development), dtype=float),
            np.zeros((len(development), 2), dtype=float),
        )
        for mode in REPRESENTATIONS
        for variant in FEATURE_VARIANTS
        for c_value in C_VALUES
    }
    for mode in REPRESENTATIONS:
        texts = _inputs(development, mode)
        matrix = _features(encoded[mode], texts, "embedding_structure")
        embedding_width = encoded[mode].shape[1]
        for train_index, validation_index in splitter.split(development, labels, groups):
            for variant in FEATURE_VARIANTS:
                width = embedding_width if variant == "embedding" else matrix.shape[1]
                train_raw = matrix[train_index, :width]
                validation_raw = matrix[validation_index, :width]
                scaler = StandardScaler().fit(train_raw)
                train_matrix = scaler.transform(train_raw)
                validation_matrix = scaler.transform(validation_raw)
                for c_value in C_VALUES:
                    direction_model, polarity_model = _fit_heads(
                        train_matrix,
                        labels[train_index],
                        c_value,
                    )
                    direction, polarity = _predict_heads(
                        direction_model,
                        polarity_model,
                        validation_matrix,
                    )
                    stores[(mode, variant, c_value)][0][validation_index] = direction
                    stores[(mode, variant, c_value)][1][validation_index] = polarity
    return stores


def _select(
    development: pd.DataFrame,
    stores: dict[tuple[str, str, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.Series, PrecisionPolicy, pd.DataFrame, np.ndarray]:
    labels = development["review_label"].astype(str).to_numpy()
    texts = development["text_excerpt"].astype(str).to_numpy()
    thresholds = tuple(np.round(np.arange(0.30, 0.951, 0.025), 3))
    rows: list[dict[str, object]] = []
    for (mode, variant, c_value), (direction, polarity) in stores.items():
        for direction_threshold in thresholds:
            for polarity_threshold in thresholds:
                for surcharge in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
                    policy = PrecisionPolicy(
                        float(direction_threshold),
                        float(polarity_threshold),
                        float(surcharge),
                    )
                    prediction = apply_precision_policy(
                        direction,
                        polarity,
                        texts,
                        policy,
                    )
                    metrics = precision_first_metrics(labels, prediction)
                    rows.append(
                        {
                            "representation": mode,
                            "feature_variant": variant,
                            "c_value": c_value,
                            **asdict(policy),
                            **metrics,
                            "observed_zero_harmful_errors": metrics["harmful_error_count"] == 0,
                        }
                    )
    table = pd.DataFrame(rows).sort_values(
        [
            "observed_zero_harmful_errors",
            "harmful_error_count",
            "correct_direction_coverage",
            "directional_precision",
            "direction_threshold",
            "polarity_threshold",
            "risk_surcharge",
        ],
        ascending=[False, True, False, False, False, False, False],
    ).reset_index(drop=True)
    selected = table.iloc[0]
    policy = PrecisionPolicy(
        float(selected["direction_threshold"]),
        float(selected["polarity_threshold"]),
        float(selected["risk_surcharge"]),
    )
    key = (
        str(selected["representation"]),
        str(selected["feature_variant"]),
        float(selected["c_value"]),
    )
    direction, polarity = stores[key]
    prediction = apply_precision_policy(direction, polarity, texts, policy)
    return selected, policy, table, prediction


def _fit_artifact(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    selected: pd.Series,
    policy: PrecisionPolicy,
) -> dict[str, object]:
    mode = str(selected["representation"])
    variant = str(selected["feature_variant"])
    values = _inputs(frame, mode)
    matrix = _features(embeddings, values, variant)
    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    labels = frame["review_label"].astype(str).to_numpy()
    direction, polarity = _fit_heads(scaled, labels, float(selected["c_value"]))
    return {
        "model_version": "stance_precision_embedding_v1",
        "encoder_model_id": MODEL_ID,
        "representation": mode,
        "feature_variant": variant,
        "c_value": float(selected["c_value"]),
        "policy": asdict(policy),
        "scaler": scaler,
        "direction_model": direction,
        "polarity_model": polarity,
        "active_forecast_integration": False,
    }


def _predict_artifact(
    artifact: dict[str, object],
    frame: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _inputs(frame, str(artifact["representation"]))
    matrix = _features(embeddings, values, str(artifact["feature_variant"]))
    scaled = artifact["scaler"].transform(matrix)
    direction, polarity = _predict_heads(
        artifact["direction_model"],
        artifact["polarity_model"],
        scaled,
    )
    policy = PrecisionPolicy(**artifact["policy"])
    prediction = apply_precision_policy(
        direction,
        polarity,
        frame["text_excerpt"].astype(str).to_numpy(),
        policy,
    )
    return direction, polarity, prediction


def _prediction_frame(
    frame: pd.DataFrame,
    direction: np.ndarray,
    polarity: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "audit_id",
        "election_id",
        "meeting_date",
        "committee",
        "speaker",
        "issue_name",
        "target_type",
        "target_name",
        "text_excerpt",
        "review_label",
    ]
    out = frame[columns].copy()
    out["direction_probability"] = direction
    out["polarity_probability_negative"] = polarity[:, 0]
    out["polarity_probability_positive"] = polarity[:, 1]
    out["precision_prediction"] = prediction
    information = pd.DataFrame(
        [neutral_information_features(text) for text in out["text_excerpt"]]
    )
    for column in information:
        out[column] = information[column].to_numpy()
    out.loc[out["precision_prediction"].ne("neutral"), "neutral_information_score"] = 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    gold_path = args.gold_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development, holdout = _load_gold(gold_path)
    all_gold = pd.concat([development, holdout], ignore_index=True)
    snapshot_path = Path(
        snapshot_download(MODEL_ID, local_files_only=args.local_files_only)
    )
    model = SentenceTransformer(
        str(snapshot_path),
        device="cpu",
        local_files_only=True,
    )
    development_encoded = {
        mode: _encode(model, development, mode) for mode in REPRESENTATIONS
    }
    holdout_encoded = {mode: _encode(model, holdout, mode) for mode in REPRESENTATIONS}

    stores = _oof(development, development_encoded)
    selected, policy, search, development_prediction = _select(development, stores)
    key = (
        str(selected["representation"]),
        str(selected["feature_variant"]),
        float(selected["c_value"]),
    )
    development_direction, development_polarity = stores[key]
    development_metrics = precision_first_metrics(
        development["review_label"],
        development_prediction,
    )

    development_artifact = _fit_artifact(
        development,
        development_encoded[str(selected["representation"])],
        selected,
        policy,
    )
    holdout_direction, holdout_polarity, holdout_prediction = _predict_artifact(
        development_artifact,
        holdout,
        holdout_encoded[str(selected["representation"])],
    )
    holdout_metrics = precision_first_metrics(holdout["review_label"], holdout_prediction)

    all_encoded = np.vstack(
        [
            development_encoded[str(selected["representation"])],
            holdout_encoded[str(selected["representation"])],
        ]
    )
    final_artifact = _fit_artifact(all_gold, all_encoded, selected, policy)
    final_artifact["encoder_revision"] = snapshot_path.name
    final_artifact["allowed_elections"] = list(ALLOWED_ELECTIONS)
    joblib.dump(final_artifact, output_dir / "stance_precision_embedding_v1.joblib")

    _prediction_frame(
        development,
        development_direction,
        development_polarity,
        development_prediction,
    ).to_csv(
        output_dir / "development_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _prediction_frame(
        holdout,
        holdout_direction,
        holdout_polarity,
        holdout_prediction,
    ).to_csv(
        output_dir / "engineering_holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    search.to_csv(output_dir / "policy_search.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "shadow_not_active",
        "model_version": "stance_precision_embedding_v1",
        "scope": {
            "allowed_elections": list(ALLOWED_ELECTIONS),
            "post_2022_rows_present": False,
            "vote_outcomes_used": False,
            "encoder_fine_tuned": False,
        },
        "data": {
            "source_split": str(gold_path.relative_to(ROOT)),
            "source_sha256": _sha256(gold_path),
            "development_rows": len(development),
            "engineering_holdout_rows": len(holdout),
            "holdout_warning": "historical engineering holdout; not publication-independent",
        },
        "encoder": {
            "model_id": MODEL_ID,
            "revision": snapshot_path.name,
            "embedding_dimensions": int(all_encoded.shape[1]),
        },
        "selection": {
            "objective": "zero observed harmful errors before maximizing correct directional coverage",
            "representation": str(selected["representation"]),
            "feature_variant": str(selected["feature_variant"]),
            "c_value": float(selected["c_value"]),
            "policy": asdict(policy),
        },
        "development_oof": development_metrics,
        "engineering_holdout": holdout_metrics,
        "adoption": {
            **stance_adoption_assessment(
                holdout_metrics,
                independent_audit=False,
                target_attribution_audited=False,
                point_in_time_audited=True,
                rolling_non_degradation=False,
            ),
            "active_forecast_changed": False,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
