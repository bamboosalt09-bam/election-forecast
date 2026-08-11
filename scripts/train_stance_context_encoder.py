"""Train fixed-encoder target-aware stance candidates in shadow mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import (  # noqa: E402
    compose_embedding_input,
    precision_first_metrics,
    risk_flags,
    stance_adoption_assessment,
)
from election_forecast.context_corpus import (  # noqa: E402
    ELECTION_CUTOFFS,
    outcome_like_columns,
)
from scripts.train_stance_precision_first import (  # noqa: E402
    ALLOWED_ELECTIONS,
    RANDOM_STATE,
    _groups,
)
from scripts.train_stance_target_aware import (  # noqa: E402
    DirectPolicy,
    _apply,
    _ordered_probabilities,
)


AUDITS = (
    (
        ROOT / "data" / "shadow" / "stance_locked_audit_v1.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v1_labels.csv",
    ),
    (
        ROOT / "data" / "shadow" / "stance_locked_audit_v2.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v2_labels.csv",
    ),
)
DEFAULT_SUPPORT = ROOT / "data" / "shadow" / "stance_target_aware_expansion_v3.csv"
ENCODERS = {
    "klue-small": {
        "backend": "transformers_mean_pool",
        "model_id": "klue/roberta-small",
        "revision": "b6b4c36d827e0293ae2fcf04d527072f10a23064",
        "model_version": "stance_klue_context_v9",
    },
    "ko-nli": {
        "backend": "sentence_transformer",
        "model_id": "jhgan/ko-sroberta-nli",
        "revision": "c4e15f24df2aceadfc931e2a57094726b2409861",
        "model_version": "stance_ko_nli_context_v10",
    },
}
MODE = "risk_aware_context"
C_VALUES = (0.10, 0.50, 1.00, 2.00)
SUPPORT_SCALES = (0.25, 0.50, 1.00)
THRESHOLDS = tuple(np.round(np.arange(0.40, 0.951, 0.025), 3))
RISK_SURCHARGES = (0.0, 0.1, 0.2, 0.3)


def _validate_training_frame(frame: pd.DataFrame, label: str) -> None:
    forbidden = sorted(set(frame["election_id"].astype(str)).difference(ALLOWED_ELECTIONS))
    if forbidden:
        raise ValueError(f"{label} contains forbidden elections: {forbidden}")
    outcome_columns = outcome_like_columns(frame.columns)
    if outcome_columns:
        raise ValueError(f"{label} contains outcome-like columns: {outcome_columns}")
    dates = pd.to_datetime(frame["meeting_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{label} contains invalid meeting_date values")
    cutoffs = frame["election_id"].astype(str).map(ELECTION_CUTOFFS)
    if (dates > cutoffs).any():
        raise ValueError(f"{label} contains post-cutoff meeting rows")


def _development() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for audit_path, label_path in AUDITS:
        audit = pd.read_csv(audit_path, encoding="utf-8-sig").fillna("")
        audit = audit.drop(
            columns=[
                "audit_locked_label",
                "audit_target_correct",
                "audit_quotation_owner",
                "audit_notes",
            ],
            errors="ignore",
        )
        labels = pd.read_csv(label_path, encoding="utf-8-sig").fillna("")
        frame = audit.merge(labels, on="text_sha256", validate="one_to_one")
        target_correct = (
            frame["audit_target_correct"].astype(str).str.lower().eq("true")
        )
        frame["review_label"] = frame["audit_locked_label"].where(
            target_correct, "neutral"
        )
        frame["development_source"] = audit_path.stem
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["text_sha256"].duplicated().any():
        raise ValueError("development audits contain duplicate text hashes")
    _validate_training_frame(combined, "development")
    return combined


def _support(path: Path, development: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    frame = frame.loc[
        ~frame["text_sha256"].astype(str).isin(development["text_sha256"].astype(str))
    ].copy()
    _validate_training_frame(frame, "support")
    frame["training_weight"] = pd.to_numeric(
        frame["training_weight"], errors="coerce"
    )
    if frame["training_weight"].isna().any():
        raise ValueError("support contains invalid training weights")
    return frame.reset_index(drop=True)


def _inputs(frame: pd.DataFrame) -> list[str]:
    return [
        compose_embedding_input(row, MODE)
        for row in frame.to_dict(orient="records")
    ]


def _cache_key(texts: Sequence[str], encoder: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(encoder, sort_keys=True).encode("utf-8"))
    for value in texts:
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def encode_texts(
    texts: Sequence[str],
    encoder: dict[str, str],
    *,
    batch_size: int,
    local_files_only: bool,
) -> np.ndarray:
    if encoder["backend"] == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            encoder["model_id"],
            revision=encoder["revision"],
            local_files_only=local_files_only,
            device="cpu",
        )
        matrix = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(matrix, dtype=np.float32)

    if encoder["backend"] != "transformers_mean_pool":
        raise ValueError(f"unsupported encoder backend: {encoder['backend']}")
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        encoder["model_id"],
        revision=encoder["revision"],
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        encoder["model_id"],
        revision=encoder["revision"],
        local_files_only=local_files_only,
        use_safetensors=True,
    )
    model.eval()
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        batches.append(pooled.cpu().numpy().astype(np.float32))
        print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)}")
    return normalize(np.vstack(batches), norm="l2").astype(np.float32)


def _embeddings(
    frame: pd.DataFrame,
    encoder: dict[str, str],
    cache_dir: Path,
    *,
    batch_size: int,
    local_files_only: bool,
) -> np.ndarray:
    texts = _inputs(frame)
    key = _cache_key(texts, encoder)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.exists():
        return np.load(path)["embeddings"]
    matrix = encode_texts(
        texts,
        encoder,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
    np.savez_compressed(path, embeddings=matrix)
    return matrix


def _classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=2_000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def _oof(
    development: pd.DataFrame,
    support: pd.DataFrame,
    development_matrix: np.ndarray,
    support_matrix: np.ndarray,
):
    labels = development["review_label"].astype(str).to_numpy()
    groups = _groups(development)
    stores = {
        (c_value, scale): np.zeros((len(development), 3), dtype=float)
        for c_value in C_VALUES
        for scale in SUPPORT_SCALES
    }
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    rows_per_fold: list[int] = []
    for train_index, validation_index in splitter.split(development, labels, groups):
        validation_hashes = set(development.iloc[validation_index]["text_sha256"])
        allowed = ~support["text_sha256"].isin(validation_hashes)
        rows_per_fold.append(int(allowed.sum()))
        train_matrix = np.vstack(
            [development_matrix[train_index], support_matrix[allowed.to_numpy()]]
        )
        train_labels = np.concatenate(
            [labels[train_index], support.loc[allowed, "review_label"].astype(str)]
        )
        support_weight = support.loc[allowed, "training_weight"].to_numpy(dtype=float)
        for c_value in C_VALUES:
            for scale in SUPPORT_SCALES:
                weights = np.concatenate(
                    [np.ones(len(train_index)), support_weight * scale]
                )
                model = _classifier(c_value).fit(
                    train_matrix, train_labels, sample_weight=weights
                )
                stores[(c_value, scale)][validation_index] = _ordered_probabilities(
                    model, development_matrix[validation_index]
                )
    return stores, rows_per_fold


def _select(development: pd.DataFrame, stores):
    truth = development["review_label"].astype(str).to_numpy()
    texts = development["text_excerpt"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[float, float, float, float], np.ndarray] = {}
    for (c_value, scale), probabilities in stores.items():
        for threshold in THRESHOLDS:
            for surcharge in RISK_SURCHARGES:
                policy = DirectPolicy(float(threshold), float(surcharge))
                prediction = _apply(probabilities, texts, policy)
                metrics = precision_first_metrics(truth, prediction)
                key = (c_value, scale, threshold, surcharge)
                predictions[key] = prediction
                rows.append(
                    {
                        "c_value": c_value,
                        "support_scale": scale,
                        "probability_threshold": threshold,
                        "risk_surcharge": surcharge,
                        **metrics,
                        "zero_harmful": metrics["harmful_error_count"] == 0,
                    }
                )
    table = pd.DataFrame(rows).sort_values(
        [
            "zero_harmful",
            "harmful_error_count",
            "correct_direction_count",
            "directional_precision",
            "probability_threshold",
            "risk_surcharge",
        ],
        ascending=[False, True, False, False, False, False],
    ).reset_index(drop=True)
    selected = table.iloc[0]
    policy = DirectPolicy(
        float(selected.probability_threshold), float(selected.risk_surcharge)
    )
    key = (
        float(selected.c_value),
        float(selected.support_scale),
        policy.probability_threshold,
        policy.risk_surcharge,
    )
    return selected, policy, table, predictions[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--support-file", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    encoder = ENCODERS[args.encoder].copy()
    model_version = encoder["model_version"]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "outputs" / "assembly_stance" / model_version
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    development = _development()
    support = _support(args.support_file.resolve(), development)
    cache_dir = output_dir / "embedding_cache"
    development_matrix = _embeddings(
        development,
        encoder,
        cache_dir,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    support_matrix = _embeddings(
        support,
        encoder,
        cache_dir,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    stores, rows_per_fold = _oof(
        development, support, development_matrix, support_matrix
    )
    selected, policy, search, prediction = _select(development, stores)
    probabilities = stores[(float(selected.c_value), float(selected.support_scale))]
    metrics = precision_first_metrics(development["review_label"], prediction)
    final_matrix = np.vstack([development_matrix, support_matrix])
    final_labels = np.concatenate(
        [
            development["review_label"].astype(str),
            support["review_label"].astype(str),
        ]
    )
    final_weights = np.concatenate(
        [
            np.ones(len(development)),
            support["training_weight"].to_numpy(dtype=float)
            * float(selected.support_scale),
        ]
    )
    model = _classifier(float(selected.c_value)).fit(
        final_matrix, final_labels, sample_weight=final_weights
    )
    artifact = {
        "model_version": model_version,
        "encoder": encoder,
        "representation": MODE,
        "c_value": float(selected.c_value),
        "support_scale": float(selected.support_scale),
        "policy": {
            "probability_threshold": policy.probability_threshold,
            "risk_surcharge": policy.risk_surcharge,
        },
        "model": model,
        "active_forecast_integration": False,
        "allowed_elections": list(ALLOWED_ELECTIONS),
    }
    joblib.dump(artifact, output_dir / f"{model_version}.joblib")
    output = development.copy()
    output["probability_negative"] = probabilities[:, 0]
    output["probability_neutral"] = probabilities[:, 1]
    output["probability_positive"] = probabilities[:, 2]
    output["context_prediction"] = prediction
    output.to_csv(
        output_dir / "development_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    search.to_csv(output_dir / "policy_search.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "shadow_not_active",
        "model_version": model_version,
        "encoder": encoder,
        "scope": {
            "allowed_elections": list(ALLOWED_ELECTIONS),
            "post_2022_rows_present": False,
            "vote_outcomes_used": False,
            "active_forecast_changed": False,
        },
        "data": {
            "development_rows": len(development),
            "development_sources": development["development_source"].value_counts().to_dict(),
            "support_rows": len(support),
            "support_rows_per_oof_fold": rows_per_fold,
            "v1_and_v2_audits_reused_as_development": True,
        },
        "selection": {
            "objective": "zero harmful target/sign errors before maximizing emissions",
            "c_value": float(selected.c_value),
            "support_scale": float(selected.support_scale),
            "policy": artifact["policy"],
        },
        "development_oof": metrics,
        "adoption": {
            **stance_adoption_assessment(
                metrics,
                independent_audit=False,
                target_attribution_audited=True,
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
