"""Evaluate a frozen external Korean NLI model as a shadow stance cascade.

The experiment follows a target-first selective-classification design:

1. decide whether the assigned target is actually evaluated;
2. decide whether the current speaker owns the evaluation;
3. emit polarity only when both role gates and the stance gate agree.

No vote outcomes are loaded.  Audit versions 1 through ``--train-through`` are
used for grouped development; later audit versions are reported as a separate
pseudo-holdout.  A new locked audit is still required before promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from election_forecast.context_corpus import (  # noqa: E402
    ELECTION_CUTOFFS,
    outcome_like_columns,
)
from election_forecast.stance_precision import (  # noqa: E402
    clean_text,
    precision_first_metrics,
    stance_adoption_assessment,
)


MODEL_ID = "pongjin/roberta_with_kornli"
MODEL_REVISION = "138378c1fb502754eb27a699a8ad71955c4d9668"
MODEL_LICENSE = "apache-2.0"
MODEL_VERSION = "stance_external_kornli_role_cascade_v26s"
ALLOWED_ELECTIONS = {"pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022"}
LABELS = np.asarray(["negative", "neutral", "positive"])
HYPOTHESIS_KEYS = (
    "direct_target",
    "speaker_owned",
    "reported_external",
    "negative",
    "positive",
    "factual_neutral",
)


@dataclass(frozen=True)
class CascadePolicy:
    target_threshold: float
    owner_threshold: float
    stance_threshold: float
    stance_margin: float


def _audit_paths(version: int, root: Path) -> list[Path]:
    exact = root / f"stance_locked_audit_v{version}.csv"
    if exact.exists():
        return [exact]
    return sorted(root.glob(f"stance_locked_audit_v{version}_part_*.csv"))


def load_audit_version(version: int, root: Path) -> pd.DataFrame:
    """Load one locked audit and its separately stored adjudication labels."""

    paths = _audit_paths(version, root)
    label_path = root / f"stance_locked_audit_v{version}_labels.csv"
    if not paths or not label_path.exists():
        raise FileNotFoundError(f"locked audit v{version} is incomplete")
    audit = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig").fillna("") for path in paths],
        ignore_index=True,
    ).drop(
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
    if len(frame) != len(labels):
        raise ValueError(f"audit v{version} does not match its labels")

    correction_path = root / f"stance_locked_audit_v{version}_adjudication_correction.csv"
    if correction_path.exists():
        corrections = pd.read_csv(correction_path, encoding="utf-8-sig").fillna("")
        for row in corrections.to_dict(orient="records"):
            matched = frame["text_sha256"].astype(str).eq(str(row["text_sha256"]))
            if int(matched.sum()) != 1:
                raise ValueError(f"invalid correction row in {correction_path}")
            frame.loc[matched, "audit_locked_label"] = row["new_label"]
            frame.loc[matched, "audit_target_correct"] = row["new_target_correct"]

    frame["audit_version"] = version
    target_correct = frame["audit_target_correct"].astype(str).str.lower().eq("true")
    frame["target_truth"] = target_correct.astype(int)
    frame["owner_truth"] = frame["audit_quotation_owner"].astype(str).eq("speaker").astype(int)
    frame["review_label"] = frame["audit_locked_label"].where(target_correct, "neutral")
    frame.loc[~frame["review_label"].isin(LABELS), "review_label"] = "neutral"
    return frame.reset_index(drop=True)


def load_audits(versions: Iterable[int], root: Path) -> pd.DataFrame:
    frames = [load_audit_version(version, root) for version in versions]
    combined = pd.concat(frames, ignore_index=True)
    duplicated = combined["text_sha256"].astype(str).duplicated(keep=False)
    if duplicated.any():
        consistency_columns = [
            "target_type",
            "target_name",
            "review_label",
            "target_truth",
            "owner_truth",
        ]
        inconsistent = (
            combined.loc[duplicated]
            .groupby("text_sha256")[consistency_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if inconsistent.any():
            raise ValueError("duplicate audit texts have inconsistent adjudications")
        combined = combined.drop_duplicates("text_sha256", keep="first").reset_index(drop=True)
    forbidden = sorted(set(combined["election_id"].astype(str)).difference(ALLOWED_ELECTIONS))
    if forbidden:
        raise ValueError(f"audits contain forbidden elections: {forbidden}")
    leaked = outcome_like_columns(combined.columns)
    if leaked:
        raise ValueError(f"audits contain outcome-like columns: {leaked}")
    dates = pd.to_datetime(combined["meeting_date"], errors="coerce")
    cutoffs = combined["election_id"].astype(str).map(ELECTION_CUTOFFS)
    if dates.isna().any() or cutoffs.isna().any() or (dates > cutoffs).any():
        raise ValueError("audits violate point-in-time cutoffs")
    return combined


def compose_premise(row: dict[str, object]) -> str:
    current = clean_text(row.get("text_excerpt", ""))[:1_200]
    before = clean_text(row.get("context_before", ""))[-400:]
    after = clean_text(row.get("context_after", ""))[:400]
    agenda = clean_text(row.get("agenda", ""))[:200]
    parts = [f"[현재 발언] {current}"]
    if before:
        parts.append(f"[앞 문맥] {before}")
    if after:
        parts.append(f"[뒤 문맥] {after}")
    if agenda:
        parts.append(f"[의제] {agenda}")
    return " ".join(parts)


def compose_hypotheses(row: dict[str, object]) -> tuple[str, ...]:
    target = clean_text(row.get("target_name", "")) or "지정된 대상"
    return (
        f"이 발언은 {target}을 직접 평가한다.",
        "현재 발언자는 이 평가를 자신의 판단으로 직접 밝힌다.",
        "현재 발언자는 다른 사람이나 기관의 평가를 단순히 인용하거나 전달한다.",
        f"현재 발언자는 {target}을 비판하거나 부정적으로 평가한다.",
        f"현재 발언자는 {target}을 지지하거나 긍정적으로 평가한다.",
        f"현재 발언자는 {target}을 평가하지 않고 사실, 절차 또는 가정만 설명한다.",
    )


def _row_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in frame["text_sha256"].astype(str):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def extract_nli_features(
    frame: pd.DataFrame,
    cache_path: Path,
    *,
    batch_size: int,
    max_length: int,
    row_chunk_size: int,
    local_files_only: bool,
) -> np.ndarray:
    """Return entailment/neutral/contradiction probabilities per hypothesis."""

    expected_hash = _row_hash(frame)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            str(cached["row_hash"].item()) != expected_hash
            or str(cached["model_revision"].item()) != MODEL_REVISION
            or tuple(cached["hypothesis_keys"].astype(str)) != HYPOTHESIS_KEYS
            or int(cached["max_length"].item()) != max_length
        ):
            raise ValueError(f"invalid external NLI cache: {cache_path}")
        return cached["probabilities"]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
    )
    model.eval()
    label_to_id = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
    order = [label_to_id[label] for label in ("entailment", "neutral", "contradiction")]

    chunk_dir = cache_path.parent / "external_nli_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    matrices: list[np.ndarray] = []
    for chunk_index, row_start in enumerate(range(0, len(frame), row_chunk_size)):
        row_stop = min(row_start + row_chunk_size, len(frame))
        chunk = frame.iloc[row_start:row_stop]
        chunk_hash = _row_hash(chunk)
        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.npz"
        if chunk_path.exists():
            cached = np.load(chunk_path, allow_pickle=False)
            if (
                str(cached["row_hash"].item()) != chunk_hash
                or str(cached["model_revision"].item()) != MODEL_REVISION
                or int(cached["max_length"].item()) != max_length
            ):
                raise ValueError(f"invalid external NLI chunk: {chunk_path}")
            matrices.append(cached["probabilities"])
            print(f"[external nli] rows {row_stop:,}/{len(frame):,} reused", flush=True)
            continue

        premises: list[str] = []
        hypotheses: list[str] = []
        for row in chunk.to_dict(orient="records"):
            premise = compose_premise(row)
            row_hypotheses = compose_hypotheses(row)
            premises.extend([premise] * len(row_hypotheses))
            hypotheses.extend(row_hypotheses)

        batches: list[np.ndarray] = []
        for start in range(0, len(premises), batch_size):
            stop = min(start + batch_size, len(premises))
            encoded = tokenizer(
                premises[start:stop],
                hypotheses[start:stop],
                padding=True,
                truncation="only_first",
                max_length=max_length,
                return_tensors="pt",
            )
            # This KorNLI checkpoint is RoBERTa (type_vocab_size=1), while its
            # BertTokenizer emits segment id 1 for paired inputs. Pair
            # separators remain; dropping token_type_ids restores RoBERTa's
            # all-zero segment convention.
            encoded.pop("token_type_ids", None)
            with torch.inference_mode():
                logits = model(**encoded).logits[:, order]
                probabilities = torch.softmax(logits, dim=1)
            batches.append(probabilities.cpu().numpy().astype(np.float32))
        chunk_matrix = np.vstack(batches).reshape(len(chunk), len(HYPOTHESIS_KEYS), 3)
        temporary_chunk = chunk_path.with_suffix(".tmp")
        with temporary_chunk.open("wb") as handle:
            np.savez_compressed(
                handle,
                probabilities=chunk_matrix,
                row_hash=np.asarray(chunk_hash),
                model_revision=np.asarray(MODEL_REVISION),
                max_length=np.asarray(max_length),
            )
        temporary_chunk.replace(chunk_path)
        matrices.append(chunk_matrix)
        print(f"[external nli] rows {row_stop:,}/{len(frame):,} computed", flush=True)

    matrix = np.concatenate(matrices, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            probabilities=matrix,
            row_hash=np.asarray(expected_hash),
            model_revision=np.asarray(MODEL_REVISION),
            hypothesis_keys=np.asarray(HYPOTHESIS_KEYS),
            max_length=np.asarray(max_length),
        )
    temporary.replace(cache_path)
    return matrix


def feature_matrix(probabilities: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    flat = probabilities.reshape(len(probabilities), -1)
    target_types = pd.get_dummies(
        pd.Categorical(frame["target_type"], categories=["person", "party", "government"]),
        dtype=float,
    ).to_numpy()
    return np.hstack([flat, target_types]).astype(np.float64)


def _classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=4_000,
        solver="lbfgs",
        random_state=1729,
    )


def _positive_probability(model: LogisticRegression, matrix: np.ndarray) -> np.ndarray:
    positions = {int(label): index for index, label in enumerate(model.classes_)}
    return model.predict_proba(matrix)[:, positions[1]]


def _stance_probabilities(model: LogisticRegression, matrix: np.ndarray) -> np.ndarray:
    positions = {str(label): index for index, label in enumerate(model.classes_)}
    raw = model.predict_proba(matrix)
    return np.column_stack([raw[:, positions[label]] for label in LABELS])


def apply_cascade(
    target_probability: Sequence[float],
    owner_probability: Sequence[float],
    stance_probability: np.ndarray,
    policy: CascadePolicy,
) -> np.ndarray:
    target_probability = np.asarray(target_probability, dtype=float)
    owner_probability = np.asarray(owner_probability, dtype=float)
    stance_probability = np.asarray(stance_probability, dtype=float)
    best = stance_probability.argmax(axis=1)
    prediction = LABELS[best].astype("<U8")
    best_probability = stance_probability[np.arange(len(prediction)), best]
    ordered = np.sort(stance_probability, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    directional = prediction != "neutral"
    accepted = (
        directional
        & (target_probability >= policy.target_threshold)
        & (owner_probability >= policy.owner_threshold)
        & (best_probability >= policy.stance_threshold)
        & (margin >= policy.stance_margin)
    )
    prediction[~accepted] = "neutral"
    return prediction


def _groups(frame: pd.DataFrame) -> np.ndarray:
    source = frame.get("source_file", pd.Series("unknown", index=frame.index)).astype(str)
    return (frame["election_id"].astype(str) + "|" + source).to_numpy()


def oof_predictions(frame: pd.DataFrame, matrix: np.ndarray, c_value: float):
    target = frame["target_truth"].astype(int).to_numpy()
    owner = frame["owner_truth"].astype(int).to_numpy()
    stance = frame["review_label"].astype(str).to_numpy()
    target_probability = np.zeros(len(frame), dtype=float)
    owner_probability = np.zeros(len(frame), dtype=float)
    stance_probability = np.zeros((len(frame), 3), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1729)
    groups = _groups(frame)
    for train, validation in splitter.split(matrix, stance, groups):
        target_model = _classifier(c_value).fit(matrix[train], target[train])
        owner_model = _classifier(c_value).fit(matrix[train], owner[train])
        stance_model = _classifier(c_value).fit(matrix[train], stance[train])
        target_probability[validation] = _positive_probability(target_model, matrix[validation])
        owner_probability[validation] = _positive_probability(owner_model, matrix[validation])
        stance_probability[validation] = _stance_probabilities(stance_model, matrix[validation])
    return target_probability, owner_probability, stance_probability


def select_policy(frame: pd.DataFrame, stores: dict[float, tuple[np.ndarray, ...]]):
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[float, float, float, float, float], np.ndarray] = {}
    truth = frame["review_label"].astype(str).to_numpy()
    for c_value, (target_probability, owner_probability, stance_probability) in stores.items():
        for target_threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
            for owner_threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
                for stance_threshold in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
                    for stance_margin in (0.00, 0.10, 0.20, 0.30):
                        policy = CascadePolicy(
                            target_threshold,
                            owner_threshold,
                            stance_threshold,
                            stance_margin,
                        )
                        prediction = apply_cascade(
                            target_probability,
                            owner_probability,
                            stance_probability,
                            policy,
                        )
                        metrics = precision_first_metrics(truth, prediction)
                        key = (
                            c_value,
                            target_threshold,
                            owner_threshold,
                            stance_threshold,
                            stance_margin,
                        )
                        predictions[key] = prediction
                        rows.append({"c_value": c_value, **asdict(policy), **metrics})
    search = pd.DataFrame(rows)
    eligible = search["predicted_directional_rows"].ge(59)
    search["eligible_emissions"] = eligible
    search = search.sort_values(
        [
            "eligible_emissions",
            "harmful_error_count",
            "harmful_error_upper_95",
            "correct_direction_count",
            "predicted_directional_rows",
        ],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)
    selected = search.iloc[0]
    policy = CascadePolicy(
        float(selected.target_threshold),
        float(selected.owner_threshold),
        float(selected.stance_threshold),
        float(selected.stance_margin),
    )
    key = (
        float(selected.c_value),
        policy.target_threshold,
        policy.owner_threshold,
        policy.stance_threshold,
        policy.stance_margin,
    )
    return float(selected.c_value), policy, predictions[key], search


def fit_models(frame: pd.DataFrame, matrix: np.ndarray, c_value: float) -> dict[str, object]:
    return {
        "target": _classifier(c_value).fit(matrix, frame["target_truth"].astype(int)),
        "owner": _classifier(c_value).fit(matrix, frame["owner_truth"].astype(int)),
        "stance": _classifier(c_value).fit(matrix, frame["review_label"].astype(str)),
    }


def predict_models(models: dict[str, object], matrix: np.ndarray, policy: CascadePolicy):
    target_probability = _positive_probability(models["target"], matrix)
    owner_probability = _positive_probability(models["owner"], matrix)
    stance_probability = _stance_probabilities(models["stance"], matrix)
    prediction = apply_cascade(
        target_probability,
        owner_probability,
        stance_probability,
        policy,
    )
    return target_probability, owner_probability, stance_probability, prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=ROOT / "data" / "shadow")
    parser.add_argument("--train-through", type=int, default=15)
    parser.add_argument("--evaluate-through", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "assembly_stance" / MODEL_VERSION,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--row-chunk-size", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.train_through >= args.evaluate_through:
        raise ValueError("evaluate-through must be later than train-through")

    versions = list(range(1, args.evaluate_through + 1))
    frame = load_audits(versions, args.audit_root.resolve())
    development = frame[frame["audit_version"].le(args.train_through)].reset_index(drop=True)
    holdout = frame[frame["audit_version"].gt(args.train_through)].reset_index(drop=True)
    overlap = holdout["text_sha256"].astype(str).isin(set(development["text_sha256"].astype(str)))
    removed_holdout_overlap_rows = int(overlap.sum())
    if overlap.any():
        holdout = holdout.loc[~overlap].reset_index(drop=True)
        frame = pd.concat([development, holdout], ignore_index=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = extract_nli_features(
        frame,
        output_dir / "external_nli_probabilities.npz",
        batch_size=args.batch_size,
        max_length=args.max_length,
        row_chunk_size=args.row_chunk_size,
        local_files_only=args.local_files_only,
    )
    matrix = feature_matrix(probabilities, frame)
    development_matrix = matrix[: len(development)]
    holdout_matrix = matrix[len(development) :]

    stores = {
        c_value: oof_predictions(development, development_matrix, c_value)
        for c_value in (0.10, 0.50, 1.00, 2.00, 5.00)
    }
    c_value, policy, development_prediction, search = select_policy(development, stores)
    models = fit_models(development, development_matrix, c_value)
    target_p, owner_p, stance_p, holdout_prediction = predict_models(
        models, holdout_matrix, policy
    )

    development_metrics = precision_first_metrics(
        development["review_label"], development_prediction
    )
    holdout_metrics = precision_first_metrics(holdout["review_label"], holdout_prediction)
    holdout_output = holdout.copy()
    holdout_output["target_probability"] = target_p
    holdout_output["owner_probability"] = owner_p
    holdout_output["probability_negative"] = stance_p[:, 0]
    holdout_output["probability_neutral"] = stance_p[:, 1]
    holdout_output["probability_positive"] = stance_p[:, 2]
    holdout_output["cascade_prediction"] = holdout_prediction
    holdout_output.to_csv(
        output_dir / "pseudo_holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    search.to_csv(output_dir / "policy_search.csv", index=False, encoding="utf-8-sig")

    artifact = {
        "model_version": MODEL_VERSION,
        "external_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
        },
        "hypothesis_keys": HYPOTHESIS_KEYS,
        "c_value": c_value,
        "policy": asdict(policy),
        "models": models,
        "active_forecast_integration": False,
        "vote_outcomes_used": False,
        "post_2022_rows_present": False,
    }
    joblib.dump(artifact, output_dir / f"{MODEL_VERSION}.joblib")
    payload = {
        "status": "shadow_not_active",
        "model_version": MODEL_VERSION,
        "external_model": artifact["external_model"],
        "method": "target_owner_stance_cascade_with_abstention",
        "external_nli_max_length": args.max_length,
        "scope": {
            "development_audit_versions": list(range(1, args.train_through + 1)),
            "pseudo_holdout_audit_versions": list(
                range(args.train_through + 1, args.evaluate_through + 1)
            ),
            "development_rows": len(development),
            "pseudo_holdout_rows": len(holdout),
            "removed_holdout_overlap_rows": removed_holdout_overlap_rows,
            "vote_outcomes_used": False,
            "post_2022_rows_present": False,
            "active_forecast_changed": False,
        },
        "selection": {"c_value": c_value, "policy": asdict(policy)},
        "development_grouped_oof": development_metrics,
        "pseudo_holdout": holdout_metrics,
        "pseudo_holdout_by_audit": {
            f"v{version}": precision_first_metrics(
                holdout_output.loc[holdout_output["audit_version"].eq(version), "review_label"],
                holdout_output.loc[
                    holdout_output["audit_version"].eq(version), "cascade_prediction"
                ],
            )
            for version in sorted(holdout_output["audit_version"].unique())
        },
        "adoption": {
            **stance_adoption_assessment(
                holdout_metrics,
                independent_audit=False,
                target_attribution_audited=True,
                point_in_time_audited=True,
                rolling_non_degradation=False,
            ),
            "reason": "v16-v17 are pseudo-holdouts; a fresh locked audit is required",
            "active_forecast_changed": False,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
