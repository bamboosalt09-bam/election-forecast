"""Train a shadow embedding classifier with weighted contrastive expansion."""

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
    PrecisionPolicy,
    apply_precision_policy,
    compose_embedding_input,
    neutral_information_features,
    precision_first_metrics,
    stance_adoption_assessment,
)
from scripts.train_stance_precision_augmented import _load_manual  # noqa: E402
from scripts.train_stance_precision_first import (  # noqa: E402
    ALLOWED_ELECTIONS,
    RANDOM_STATE,
    _groups,
    _load_gold,
)


MODEL_ID = "jhgan/ko-sroberta-multitask"
DEFAULT_GOLD = ROOT / "data" / "shadow" / "stance_precision_gold_through2022.csv"
DEFAULT_EXPANSION = ROOT / "data" / "shadow" / "stance_contrastive_expansion_v2.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "assembly_stance" / "precision_embedding_augmented_v6"
MODEL_VERSION = "stance_precision_embedding_augmented_v6"
MODE = "risk_aware_context"
C_VALUES = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
EXPANSION_SCALES = (0.50, 1.00, 1.50)
THRESHOLDS = tuple(np.round(np.arange(0.25, 0.951, 0.025), 3))
RISK_SURCHARGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs(frame: pd.DataFrame) -> list[str]:
    return [
        compose_embedding_input(row, MODE) for row in frame.to_dict(orient="records")
    ]


def _encode(model: SentenceTransformer, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        model.encode(
            _inputs(frame),
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float64,
    )


def _classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=5_000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def _fit_heads(
    matrix: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    sample_weight: np.ndarray,
):
    directional = (labels != "neutral").astype(int)
    direction = _classifier(c_value).fit(
        matrix, directional, sample_weight=sample_weight
    )
    mask = labels != "neutral"
    polarity = _classifier(c_value).fit(
        matrix[mask], labels[mask], sample_weight=sample_weight[mask]
    )
    return direction, polarity


def _predict_heads(direction, polarity, matrix: np.ndarray):
    raw_direction = direction.predict_proba(matrix)
    direction_probability = raw_direction[:, list(direction.classes_).index(1)]
    raw_polarity = polarity.predict_proba(matrix)
    positions = {str(value): index for index, value in enumerate(polarity.classes_)}
    return direction_probability, np.column_stack(
        [
            raw_polarity[:, positions["negative"]],
            raw_polarity[:, positions["positive"]],
        ]
    )


def _oof(
    development: pd.DataFrame,
    expansion: pd.DataFrame,
    development_embedding: np.ndarray,
    expansion_embedding: np.ndarray,
):
    labels = development["review_label"].astype(str).to_numpy()
    expansion_labels = expansion["review_label"].astype(str).to_numpy()
    groups = _groups(development)
    expansion_groups = _groups(expansion)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    stores = {
        (c_value, scale): (
            np.zeros(len(development), dtype=float),
            np.zeros((len(development), 2), dtype=float),
        )
        for c_value in C_VALUES
        for scale in EXPANSION_SCALES
    }
    rows_per_fold: list[int] = []
    for train_index, validation_index in splitter.split(development, labels, groups):
        allowed = ~expansion_groups.isin(set(groups.iloc[validation_index]))
        rows_per_fold.append(int(allowed.sum()))
        raw_train = np.vstack(
            [development_embedding[train_index], expansion_embedding[allowed.to_numpy()]]
        )
        combined_labels = np.concatenate(
            [labels[train_index], expansion_labels[allowed.to_numpy()]]
        )
        base_expansion_weight = expansion.loc[allowed, "training_weight"].to_numpy(
            dtype=float
        )
        for c_value in C_VALUES:
            for scale in EXPANSION_SCALES:
                scaler = StandardScaler().fit(raw_train)
                train_matrix = scaler.transform(raw_train)
                validation_matrix = scaler.transform(development_embedding[validation_index])
                sample_weight = np.concatenate(
                    [
                        np.ones(len(train_index), dtype=float),
                        base_expansion_weight * scale,
                    ]
                )
                direction, polarity = _fit_heads(
                    train_matrix, combined_labels, c_value, sample_weight
                )
                dprob, pprob = _predict_heads(direction, polarity, validation_matrix)
                stores[(c_value, scale)][0][validation_index] = dprob
                stores[(c_value, scale)][1][validation_index] = pprob
    return stores, rows_per_fold


def _select(development: pd.DataFrame, stores):
    labels = development["review_label"].astype(str).to_numpy()
    texts = development["text_excerpt"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[float, float, float, float, float], np.ndarray] = {}
    for (c_value, scale), (direction, polarity) in stores.items():
        for direction_threshold in THRESHOLDS:
            for polarity_threshold in THRESHOLDS:
                for surcharge in RISK_SURCHARGES:
                    policy = PrecisionPolicy(
                        float(direction_threshold),
                        float(polarity_threshold),
                        float(surcharge),
                    )
                    prediction = apply_precision_policy(
                        direction, polarity, texts, policy
                    )
                    metrics = precision_first_metrics(labels, prediction)
                    key = (
                        c_value,
                        scale,
                        policy.direction_threshold,
                        policy.polarity_threshold,
                        policy.risk_surcharge,
                    )
                    predictions[key] = prediction
                    rows.append(
                        {
                            "c_value": c_value,
                            "expansion_scale": scale,
                            **asdict(policy),
                            **metrics,
                            "observed_zero_harmful_errors": metrics["harmful_error_count"]
                            == 0,
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
        float(selected.direction_threshold),
        float(selected.polarity_threshold),
        float(selected.risk_surcharge),
    )
    key = (
        float(selected.c_value),
        float(selected.expansion_scale),
        policy.direction_threshold,
        policy.polarity_threshold,
        policy.risk_surcharge,
    )
    return selected, policy, table, predictions[key]


def _fit_artifact(
    gold: pd.DataFrame,
    expansion: pd.DataFrame,
    gold_embedding: np.ndarray,
    expansion_embedding: np.ndarray,
    selected: pd.Series,
    policy: PrecisionPolicy,
):
    raw = np.vstack([gold_embedding, expansion_embedding])
    labels = np.concatenate(
        [
            gold["review_label"].astype(str).to_numpy(),
            expansion["review_label"].astype(str).to_numpy(),
        ]
    )
    scaler = StandardScaler().fit(raw)
    matrix = scaler.transform(raw)
    sample_weight = np.concatenate(
        [
            np.ones(len(gold), dtype=float),
            expansion["training_weight"].to_numpy(dtype=float)
            * float(selected.expansion_scale),
        ]
    )
    direction, polarity = _fit_heads(
        matrix, labels, float(selected.c_value), sample_weight
    )
    return {
        "model_version": MODEL_VERSION,
        "encoder_model_id": MODEL_ID,
        "representation": MODE,
        "feature_variant": "embedding",
        "c_value": float(selected.c_value),
        "expansion_scale": float(selected.expansion_scale),
        "policy": asdict(policy),
        "scaler": scaler,
        "direction_model": direction,
        "polarity_model": polarity,
        "active_forecast_integration": False,
    }


def _predict_artifact(artifact, frame, embedding):
    matrix = artifact["scaler"].transform(embedding)
    direction, polarity = _predict_heads(
        artifact["direction_model"], artifact["polarity_model"], matrix
    )
    prediction = apply_precision_policy(
        direction,
        polarity,
        frame["text_excerpt"].astype(str).to_numpy(),
        PrecisionPolicy(**artifact["policy"]),
    )
    return direction, polarity, prediction


def _prediction_frame(frame, direction, polarity, prediction):
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
    output = frame[columns].copy()
    output["direction_probability"] = direction
    output["polarity_probability_negative"] = polarity[:, 0]
    output["polarity_probability_positive"] = polarity[:, 1]
    output["precision_prediction"] = prediction
    information = pd.DataFrame(
        [neutral_information_features(text) for text in output["text_excerpt"]]
    )
    for column in information:
        output[column] = information[column].to_numpy()
    output.loc[
        output["precision_prediction"].ne("neutral"), "neutral_information_score"
    ] = 0.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--expansion-file", type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    gold_path = args.gold_file.resolve()
    expansion_path = args.expansion_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development, holdout = _load_gold(gold_path)
    all_gold = pd.concat([development, holdout], ignore_index=True)
    expansion = _load_manual(expansion_path, all_gold)
    snapshot = Path(
        snapshot_download(MODEL_ID, local_files_only=args.local_files_only)
    )
    model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True)
    development_embedding = _encode(model, development)
    holdout_embedding = _encode(model, holdout)
    expansion_embedding = _encode(model, expansion)
    stores, rows_per_fold = _oof(
        development, expansion, development_embedding, expansion_embedding
    )
    selected, policy, search, development_prediction = _select(development, stores)
    key = (float(selected.c_value), float(selected.expansion_scale))
    development_direction, development_polarity = stores[key]
    development_metrics = precision_first_metrics(
        development["review_label"], development_prediction
    )

    development_artifact = _fit_artifact(
        development,
        expansion,
        development_embedding,
        expansion_embedding,
        selected,
        policy,
    )
    holdout_direction, holdout_polarity, holdout_prediction = _predict_artifact(
        development_artifact, holdout, holdout_embedding
    )
    holdout_metrics = precision_first_metrics(
        holdout["review_label"], holdout_prediction
    )
    final_artifact = _fit_artifact(
        all_gold,
        expansion,
        np.vstack([development_embedding, holdout_embedding]),
        expansion_embedding,
        selected,
        policy,
    )
    final_artifact["encoder_revision"] = snapshot.name
    final_artifact["allowed_elections"] = list(ALLOWED_ELECTIONS)
    joblib.dump(final_artifact, output_dir / f"{MODEL_VERSION}.joblib")
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
        holdout, holdout_direction, holdout_polarity, holdout_prediction
    ).to_csv(
        output_dir / "engineering_holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    search.to_csv(output_dir / "policy_search.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "shadow_not_active",
        "model_version": MODEL_VERSION,
        "scope": {
            "allowed_elections": list(ALLOWED_ELECTIONS),
            "post_2022_rows_present": False,
            "vote_outcomes_used": False,
            "encoder_fine_tuned": False,
        },
        "data": {
            "gold_sha256": _sha256(gold_path),
            "expansion_sha256": _sha256(expansion_path),
            "expansion_rows": len(expansion),
            "expansion_rows_per_oof_fold": rows_per_fold,
            "engineering_holdout_warning": (
                "historical engineering holdout; not publication-independent"
            ),
        },
        "encoder": {
            "model_id": MODEL_ID,
            "revision": snapshot.name,
            "embedding_dimensions": int(expansion_embedding.shape[1]),
        },
        "selection": {
            "objective": "zero observed harmful errors before maximizing coverage",
            "representation": MODE,
            "c_value": float(selected.c_value),
            "expansion_scale": float(selected.expansion_scale),
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
