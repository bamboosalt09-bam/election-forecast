"""Train and audit a metadata-free quantitative Korean stance model."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_text_model import (  # noqa: E402
    apply_rule_correction,
    build_stance_pipeline,
)


AUDIT_DIR = ROOT / "outputs" / "assembly_stance" / "stance_metadata_blind_audit_300"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "stance_text_model_v1"
RANDOM_STATE = 20260714


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_frame() -> pd.DataFrame:
    blind = pd.read_csv(AUDIT_DIR / "blind_review.csv", encoding="utf-8-sig")
    annotations = pd.read_csv(AUDIT_DIR / "blind_annotations.csv", encoding="utf-8-sig")
    key = pd.read_csv(AUDIT_DIR / "hidden_key.csv", encoding="utf-8-sig")
    frame = blind[["audit_id", "text_excerpt"]].merge(
        annotations[["audit_id", "review_label"]], on="audit_id", validate="one_to_one"
    ).merge(
        key[["audit_id", "stance_polarity", "text_sha256"]], on="audit_id", validate="one_to_one"
    )
    return frame.loc[frame["review_label"].isin(["negative", "neutral", "positive"])].reset_index(drop=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    labels = ["negative", "neutral", "positive"]
    directional_mask = y_pred != "neutral"
    directional_precision = (
        precision_score(y_true[directional_mask], y_pred[directional_mask], average="micro", zero_division=0)
        if directional_mask.any()
        else 0.0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "directional_coverage": float(directional_mask.mean()),
        "directional_precision": float(directional_precision),
        "confusion_matrix_true_rows_pred_columns": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _policy_candidates(
    probabilities: np.ndarray,
    classes: np.ndarray,
    legacy_predictions: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for margin in (0.05, 0.10, 0.15, 0.20):
            predicted = apply_rule_correction(
                probabilities,
                classes,
                legacy_predictions,
                min_override_probability=threshold,
                min_probability_margin=margin,
                allow_neutral_source=False,
            )
            metric = _metrics(y_true, predicted)
            rows.append(
                {
                    "min_override_probability": threshold,
                    "min_probability_margin": margin,
                    "macro_f1": float(metric["macro_f1"]),
                    "accuracy": float(metric["accuracy"]),
                    "directional_coverage": float(metric["directional_coverage"]),
                    "directional_precision": float(metric["directional_precision"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_training_frame()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_indices, holdout_indices = next(splitter.split(frame["text_excerpt"], frame["review_label"]))
    train = frame.iloc[train_indices].reset_index(drop=True)
    holdout = frame.iloc[holdout_indices].reset_index(drop=True)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_stance_pipeline(),
        param_grid={"classifier__C": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]},
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    x_train = train["text_excerpt"].fillna("").astype(str).to_numpy()
    y_train = train["review_label"].astype(str).to_numpy()
    search.fit(x_train, y_train)

    best_c = float(search.best_params_["classifier__C"])
    oof_model = build_stance_pipeline(c_value=best_c)
    oof_probabilities = cross_val_predict(
        oof_model,
        x_train,
        y_train,
        cv=inner_cv,
        method="predict_proba",
        n_jobs=1,
    )
    oof_model.fit(x_train, y_train)
    classes = oof_model.named_steps["classifier"].classes_
    rule_train = train["stance_polarity"].map({-1: "negative", 0: "neutral", 1: "positive"}).to_numpy()
    baseline_directional_precision = float(_metrics(y_train, rule_train)["directional_precision"])
    policy_table = _policy_candidates(oof_probabilities, classes, rule_train, y_train)
    eligible = policy_table.loc[
        policy_table["directional_precision"].ge(baseline_directional_precision)
    ]
    ranked = eligible if not eligible.empty else policy_table
    best_policy = ranked.sort_values(
        ["macro_f1", "directional_precision", "directional_coverage"], ascending=False
    ).iloc[0]

    x_holdout = holdout["text_excerpt"].fillna("").astype(str).to_numpy()
    y_holdout = holdout["review_label"].astype(str).to_numpy()
    holdout_probabilities = oof_model.predict_proba(x_holdout)
    raw_holdout = classes[np.argmax(holdout_probabilities, axis=1)]
    rule_holdout = holdout["stance_polarity"].map({-1: "negative", 0: "neutral", 1: "positive"}).to_numpy()
    policy_holdout = apply_rule_correction(
        holdout_probabilities,
        classes,
        rule_holdout,
        min_override_probability=float(best_policy["min_override_probability"]),
        min_probability_margin=float(best_policy["min_probability_margin"]),
        allow_neutral_source=False,
    )

    probability_frame = pd.DataFrame(
        holdout_probabilities,
        columns=[f"probability_{label}" for label in classes],
    )
    holdout_output = pd.concat(
        [
            holdout[["audit_id", "text_sha256", "text_excerpt", "review_label"]].reset_index(drop=True),
            pd.DataFrame(
                {
                    "rule_prediction": rule_holdout,
                    "raw_model_prediction": raw_holdout,
                    "model_prediction": policy_holdout,
                }
            ),
            probability_frame,
        ],
        axis=1,
    )
    holdout_output.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(search.cv_results_).to_csv(OUTPUT_DIR / "hyperparameter_cv.csv", index=False)
    policy_table.to_csv(OUTPUT_DIR / "abstention_policy_cv.csv", index=False)
    pd.concat(
        [
            train.assign(split="train"),
            holdout.assign(split="holdout"),
        ],
        ignore_index=True,
    ).to_csv(OUTPUT_DIR / "frozen_split.csv", index=False, encoding="utf-8-sig")

    final_model = build_stance_pipeline(c_value=best_c)
    final_model.fit(frame["text_excerpt"].fillna("").astype(str), frame["review_label"].astype(str))
    artifact = {
        "model": final_model,
        "classes": final_model.named_steps["classifier"].classes_.tolist(),
        "min_override_probability": float(best_policy["min_override_probability"]),
        "min_probability_margin": float(best_policy["min_probability_margin"]),
        "policy": "legacy_rule_with_probability_override",
        "allow_neutral_source": False,
        "metadata_columns": [],
        "training_rows": int(len(frame)),
        "training_source_sha256": _sha256(AUDIT_DIR / "blind_annotations.csv"),
        "model_version": "stance_text_tfidf_logreg_v1",
    }
    joblib.dump(artifact, OUTPUT_DIR / "stance_text_model_v1.joblib")

    metrics = {
        "status": "trained",
        "training_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "best_c": best_c,
        "inner_cv_best_macro_f1": float(search.best_score_),
        "selected_policy": {
            "min_override_probability": float(best_policy["min_override_probability"]),
            "min_probability_margin": float(best_policy["min_probability_margin"]),
            "oof_macro_f1": float(best_policy["macro_f1"]),
            "oof_directional_precision": float(best_policy["directional_precision"]),
            "oof_directional_coverage": float(best_policy["directional_coverage"]),
        },
        "holdout_rule": _metrics(y_holdout, rule_holdout),
        "holdout_raw_model": _metrics(y_holdout, raw_holdout),
        "holdout_corrected_model": _metrics(y_holdout, policy_holdout),
        "metadata_columns": [],
        "warning": "Post-audit engineering holdout; a second independently annotated sample is still required.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
