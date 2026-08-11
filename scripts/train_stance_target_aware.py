"""Train a direct target-aware, speaker-owned shadow stance classifier."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import (  # noqa: E402
    build_precision_features,
    compose_precision_input,
    precision_first_metrics,
    risk_flags,
    stance_adoption_assessment,
)
from scripts.train_stance_precision_first import (  # noqa: E402
    ALLOWED_ELECTIONS,
    RANDOM_STATE,
    _groups,
)


DEFAULT_AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v1.csv"
DEFAULT_LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v1_labels.csv"
DEFAULT_SUPPORT = ROOT / "data" / "shadow" / "stance_target_aware_expansion_v3.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "assembly_stance" / "stance_target_aware_v8"
MODEL_VERSION = "stance_target_aware_v8"
MODE = "current_context"
C_VALUES = (0.10, 0.50, 1.00, 2.00)
SUPPORT_SCALES = (0.50, 1.00, 1.50)
THRESHOLDS = tuple(np.round(np.arange(0.40, 0.951, 0.025), 3))
RISK_SURCHARGES = (0.0, 0.1, 0.2, 0.3, 0.4)


@dataclass(frozen=True)
class DirectPolicy:
    probability_threshold: float
    risk_surcharge: float


def _development(audit_path: Path, label_path: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_path, encoding="utf-8-sig").fillna("").drop(
        columns=[
            "audit_locked_label",
            "audit_target_correct",
            "audit_quotation_owner",
            "audit_notes",
        ]
    )
    labels = pd.read_csv(label_path, encoding="utf-8-sig").fillna("")
    frame = audit.merge(labels, on="text_sha256", validate="one_to_one")
    target_correct = frame["audit_target_correct"].astype(str).str.lower().eq("true")
    frame["review_label"] = frame["audit_locked_label"].where(
        target_correct, "neutral"
    )
    return frame.reset_index(drop=True)


def _support(path: Path, development: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    frame = frame.loc[
        ~frame["text_sha256"].astype(str).isin(development["text_sha256"].astype(str))
    ].copy()
    forbidden = sorted(set(frame["election_id"]).difference(ALLOWED_ELECTIONS))
    if forbidden:
        raise ValueError(f"support contains forbidden elections: {forbidden}")
    frame["training_weight"] = pd.to_numeric(
        frame["training_weight"], errors="coerce"
    )
    if frame["training_weight"].isna().any():
        raise ValueError("support contains invalid training weights")
    return frame.reset_index(drop=True)


def _inputs(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            compose_precision_input(row, MODE)
            for row in frame.to_dict(orient="records")
        ],
        dtype=str,
    )


def _classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=2_000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )


def _ordered_probabilities(model, matrix) -> np.ndarray:
    raw = model.predict_proba(matrix)
    positions = {str(value): index for index, value in enumerate(model.classes_)}
    return np.column_stack(
        [raw[:, positions[label]] for label in ("negative", "neutral", "positive")]
    )


def _apply(probabilities: np.ndarray, texts, policy: DirectPolicy) -> np.ndarray:
    best = np.argmax(probabilities, axis=1)
    labels = np.asarray(["negative", "neutral", "positive"])[best]
    confidence = probabilities[np.arange(len(probabilities)), best]
    required = np.minimum(
        policy.probability_threshold
        + risk_flags(texts).astype(float) * policy.risk_surcharge,
        0.999,
    )
    labels = labels.astype("<U8")
    labels[confidence < required] = "neutral"
    return labels


def _oof(development: pd.DataFrame, support: pd.DataFrame):
    labels = development["review_label"].astype(str).to_numpy()
    groups = _groups(development)
    development_inputs = _inputs(development)
    support_inputs = _inputs(support)
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
        values = pd.concat(
            [development_inputs.iloc[train_index], support_inputs.loc[allowed]],
            ignore_index=True,
        )
        combined_labels = np.concatenate(
            [labels[train_index], support.loc[allowed, "review_label"].astype(str)]
        )
        features = build_precision_features()
        train_matrix = features.fit_transform(values)
        validation_matrix = features.transform(
            development_inputs.iloc[validation_index]
        )
        support_weight = support.loc[allowed, "training_weight"].to_numpy(dtype=float)
        for c_value in C_VALUES:
            for scale in SUPPORT_SCALES:
                sample_weight = np.concatenate(
                    [
                        np.ones(len(train_index), dtype=float),
                        support_weight * scale,
                    ]
                )
                model = _classifier(c_value).fit(
                    train_matrix, combined_labels, sample_weight=sample_weight
                )
                stores[(c_value, scale)][validation_index] = _ordered_probabilities(
                    model, validation_matrix
                )
    return stores, rows_per_fold


def _select(development: pd.DataFrame, stores):
    truth = development["review_label"].astype(str).to_numpy()
    texts = development["text_excerpt"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    predictions = {}
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
                        "observed_zero_harmful_errors": metrics["harmful_error_count"]
                        == 0,
                    }
                )
    table = pd.DataFrame(rows).sort_values(
        [
            "observed_zero_harmful_errors",
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


def _fit_artifact(development, support, selected, policy):
    values = pd.concat([_inputs(development), _inputs(support)], ignore_index=True)
    labels = np.concatenate(
        [
            development["review_label"].astype(str),
            support["review_label"].astype(str),
        ]
    )
    features = build_precision_features()
    matrix = features.fit_transform(values)
    sample_weight = np.concatenate(
        [
            np.ones(len(development), dtype=float),
            support["training_weight"].to_numpy(dtype=float)
            * float(selected.support_scale),
        ]
    )
    model = _classifier(float(selected.c_value)).fit(
        matrix, labels, sample_weight=sample_weight
    )
    return {
        "model_version": MODEL_VERSION,
        "representation": MODE,
        "c_value": float(selected.c_value),
        "support_scale": float(selected.support_scale),
        "policy": {
            "probability_threshold": policy.probability_threshold,
            "risk_surcharge": policy.risk_surcharge,
        },
        "features": features,
        "model": model,
        "allowed_elections": list(ALLOWED_ELECTIONS),
        "active_forecast_integration": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-file", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--support-file", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    development = _development(args.audit_file.resolve(), args.label_file.resolve())
    support = _support(args.support_file.resolve(), development)
    stores, rows_per_fold = _oof(development, support)
    selected, policy, search, prediction = _select(development, stores)
    probabilities = stores[(float(selected.c_value), float(selected.support_scale))]
    metrics = precision_first_metrics(development["review_label"], prediction)
    artifact = _fit_artifact(development, support, selected, policy)
    joblib.dump(artifact, output_dir / f"{MODEL_VERSION}.joblib")
    output = development.copy()
    output["probability_negative"] = probabilities[:, 0]
    output["probability_neutral"] = probabilities[:, 1]
    output["probability_positive"] = probabilities[:, 2]
    output["target_aware_prediction"] = prediction
    output.to_csv(
        output_dir / "development_oof_predictions.csv",
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
            "activation_semantics": "speaker-owned stance toward a valid target",
        },
        "data": {
            "development_rows": len(development),
            "support_rows": len(support),
            "support_rows_per_oof_fold": rows_per_fold,
            "audit_v1_reused_as_development": True,
        },
        "selection": {
            "objective": "zero harmful target/sign errors before maximizing emissions",
            "representation": MODE,
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
