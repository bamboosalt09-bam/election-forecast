"""Train a context-aware stance classifier with 5,000 weak and 273 gold rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_context_model import compose_context_input, weak_context_label  # noqa: E402
from election_forecast.stance_text_model import apply_rule_correction, build_stance_pipeline  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
WEAK_INPUT = DATA_DIR / "stance_context_5000.csv"
GOLD_INPUT = DATA_DIR / "gold_context_273.csv"
RANDOM_STATE = 20260714
LABELS = ["negative", "neutral", "positive"]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    directional = y_pred != "neutral"
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
        "directional_coverage": float(directional.mean()),
        "directional_precision": float(
            precision_score(y_true[directional], y_pred[directional], average="micro", zero_division=0)
            if directional.any()
            else 0.0
        ),
        "confusion_matrix_true_rows_pred_columns": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    weak = pd.read_csv(WEAK_INPUT, encoding="utf-8-sig").fillna("")
    gold = pd.read_csv(GOLD_INPUT, encoding="utf-8-sig").fillna("")
    weak_labels = [weak_context_label(row) for row in weak.to_dict(orient="records")]
    weak["training_label"] = [item.label for item in weak_labels]
    weak["weak_confidence"] = [item.confidence for item in weak_labels]
    weak["weak_reason"] = [item.reason for item in weak_labels]
    weak["model_input"] = [compose_context_input(row) for row in weak.to_dict(orient="records")]
    gold["model_input"] = [compose_context_input(row) for row in gold.to_dict(orient="records")]
    weak.to_csv(DATA_DIR / "weak_context_labels_5000.csv", index=False, encoding="utf-8-sig")
    return weak, gold


def fit_fold_model(
    gold_train: pd.DataFrame,
    weak: pd.DataFrame,
    *,
    c_value: float,
    weak_mass_ratio: float,
):
    combined_x = pd.concat([gold_train["model_input"], weak["model_input"]], ignore_index=True)
    template = build_stance_pipeline(c_value=c_value)
    features = template.named_steps["features"]
    classifier = template.named_steps["classifier"]
    feature_matrix = features.fit_transform(combined_x.astype(str))
    if weak_mass_ratio > 0:
        combined_y = pd.concat([gold_train["review_label"], weak["training_label"]], ignore_index=True)
        weak_raw = weak["weak_confidence"].to_numpy(float)
        weak_weights = weak_raw * (weak_mass_ratio * len(gold_train) / weak_raw.sum())
        sample_weight = np.concatenate([np.ones(len(gold_train)), weak_weights])
        classifier.fit(feature_matrix, combined_y.astype(str), sample_weight=sample_weight)
    else:
        classifier.fit(
            feature_matrix[: len(gold_train)],
            gold_train["review_label"].astype(str),
            sample_weight=np.ones(len(gold_train)),
        )
    return Pipeline([("features", features), ("classifier", classifier)])


def select_model(
    gold_train: pd.DataFrame,
    weak: pd.DataFrame,
) -> tuple[float, float, np.ndarray, np.ndarray, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float]] = []
    saved: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    y = gold_train["review_label"].astype(str).to_numpy()
    parameter_keys = [
        (c_value, weak_mass_ratio)
        for c_value in (0.10, 0.25, 0.50, 1.00)
        for weak_mass_ratio in (0.0, 0.25, 0.50, 1.00)
    ]
    probability_store = {
        key: np.zeros((len(gold_train), len(LABELS)), dtype=float) for key in parameter_keys
    }
    for train_index, validation_index in cv.split(gold_train["model_input"], y):
        fold_gold = gold_train.iloc[train_index]
        combined_x = pd.concat([fold_gold["model_input"], weak["model_input"]], ignore_index=True)
        feature_template = build_stance_pipeline(c_value=1.0)
        features = feature_template.named_steps["features"]
        combined_matrix = features.fit_transform(combined_x.astype(str))
        validation_matrix = features.transform(
            gold_train.iloc[validation_index]["model_input"].astype(str)
        )
        gold_matrix = combined_matrix[: len(fold_gold)]
        weak_raw = weak["weak_confidence"].to_numpy(float)
        for c_value, weak_mass_ratio in parameter_keys:
            classifier = build_stance_pipeline(c_value=c_value).named_steps["classifier"]
            if weak_mass_ratio > 0:
                training_matrix = combined_matrix
                training_y = pd.concat(
                    [fold_gold["review_label"], weak["training_label"]], ignore_index=True
                ).astype(str)
                weak_weights = weak_raw * (weak_mass_ratio * len(fold_gold) / weak_raw.sum())
                sample_weight = np.concatenate([np.ones(len(fold_gold)), weak_weights])
            else:
                training_matrix = gold_matrix
                training_y = fold_gold["review_label"].astype(str)
                sample_weight = np.ones(len(fold_gold))
            classifier.fit(training_matrix, training_y, sample_weight=sample_weight)
            fold_probabilities = classifier.predict_proba(validation_matrix)
            class_positions = {label: index for index, label in enumerate(classifier.classes_)}
            for label_index, label in enumerate(LABELS):
                probability_store[(c_value, weak_mass_ratio)][validation_index, label_index] = (
                    fold_probabilities[:, class_positions[label]]
                )
    for c_value, weak_mass_ratio in parameter_keys:
        probabilities = probability_store[(c_value, weak_mass_ratio)]
        predictions = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
        result = metrics(y, predictions)
        rows.append(
            {
                "c_value": c_value,
                "weak_mass_ratio": weak_mass_ratio,
                "oof_accuracy": float(result["accuracy"]),
                "oof_macro_f1": float(result["macro_f1"]),
                "oof_kappa": float(result["cohen_kappa"]),
            }
        )
        saved[(c_value, weak_mass_ratio)] = (probabilities, predictions)
    table = pd.DataFrame(rows).sort_values(
        ["oof_macro_f1", "oof_accuracy", "weak_mass_ratio", "c_value"],
        ascending=[False, False, True, True],
    )
    best = table.iloc[0]
    key = (float(best["c_value"]), float(best["weak_mass_ratio"]))
    probabilities, predictions = saved[key]
    return key[0], key[1], probabilities, predictions, table


def select_correction_policy(
    probabilities: np.ndarray,
    gold_train: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    legacy = gold_train["stance_polarity"].astype(int).map({-1: "negative", 0: "neutral", 1: "positive"}).to_numpy()
    y = gold_train["review_label"].astype(str).to_numpy()
    rows: list[dict[str, float]] = []
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for margin in (0.05, 0.10, 0.15, 0.20):
            predicted = apply_rule_correction(
                probabilities,
                LABELS,
                legacy,
                min_override_probability=threshold,
                min_probability_margin=margin,
                allow_neutral_source=False,
            )
            result = metrics(y, predicted)
            rows.append(
                {
                    "min_override_probability": threshold,
                    "min_probability_margin": margin,
                    "oof_accuracy": float(result["accuracy"]),
                    "oof_macro_f1": float(result["macro_f1"]),
                    "oof_directional_precision": float(result["directional_precision"]),
                    "oof_directional_coverage": float(result["directional_coverage"]),
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["oof_macro_f1", "oof_accuracy", "oof_directional_precision"], ascending=False
    )
    return table.iloc[0].to_dict(), table


def main() -> None:
    weak, gold = prepare()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_index, holdout_index = next(splitter.split(gold["model_input"], gold["review_label"]))
    gold_train = gold.iloc[train_index].reset_index(drop=True)
    holdout = gold.iloc[holdout_index].reset_index(drop=True)
    c_value, weak_mass_ratio, oof_probabilities, oof_raw, search = select_model(gold_train, weak)
    policy, policy_table = select_correction_policy(oof_probabilities, gold_train)

    model = fit_fold_model(
        gold_train,
        weak,
        c_value=c_value,
        weak_mass_ratio=weak_mass_ratio,
    )
    holdout_probabilities = model.predict_proba(holdout["model_input"].astype(str))
    classes = model.named_steps["classifier"].classes_
    raw_holdout = classes[np.argmax(holdout_probabilities, axis=1)]
    legacy_holdout = holdout["stance_polarity"].astype(int).map(
        {-1: "negative", 0: "neutral", 1: "positive"}
    ).to_numpy()
    corrected_holdout = apply_rule_correction(
        holdout_probabilities,
        classes,
        legacy_holdout,
        min_override_probability=float(policy["min_override_probability"]),
        min_probability_margin=float(policy["min_probability_margin"]),
        allow_neutral_source=False,
    )
    y_holdout = holdout["review_label"].astype(str).to_numpy()

    final_model = fit_fold_model(
        gold,
        weak,
        c_value=c_value,
        weak_mass_ratio=weak_mass_ratio,
    )
    artifact = {
        "model": final_model,
        "classes": final_model.named_steps["classifier"].classes_.tolist(),
        "model_version": "stance_context_5000_v1",
        "c_value": c_value,
        "weak_mass_ratio": weak_mass_ratio,
        "min_override_probability": float(policy["min_override_probability"]),
        "min_probability_margin": float(policy["min_probability_margin"]),
        "allow_neutral_source": False,
        "metadata_columns": ["target_type", "target_name", "target_alias"],
        "metadata_policy": "target identity is masked to [TARGET]; only target type and explicit span replacement are used",
        "gold_rows": int(len(gold)),
        "weak_rows": int(len(weak)),
    }
    joblib.dump(artifact, DATA_DIR / "stance_context_5000_v1.joblib")

    holdout_output = holdout[
        ["audit_id", "text_sha256", "text_excerpt", "context_before", "context_after", "review_label"]
    ].copy()
    holdout_output["legacy_prediction"] = legacy_holdout
    holdout_output["raw_context_prediction"] = raw_holdout
    holdout_output["corrected_context_prediction"] = corrected_holdout
    for class_index, label in enumerate(classes):
        holdout_output[f"probability_{label}"] = holdout_probabilities[:, class_index]
    holdout_output.to_csv(DATA_DIR / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    search.to_csv(DATA_DIR / "model_selection_oof.csv", index=False)
    policy_table.to_csv(DATA_DIR / "correction_policy_oof.csv", index=False)
    pd.concat(
        [gold_train.assign(split="train"), holdout.assign(split="holdout")], ignore_index=True
    ).to_csv(DATA_DIR / "frozen_gold_split.csv", index=False, encoding="utf-8-sig")

    weak_distribution = weak.groupby(["training_label", "weak_reason"]).size().reset_index(name="rows")
    weak_distribution.to_csv(DATA_DIR / "weak_label_distribution.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "trained",
        "weak_rows": int(len(weak)),
        "gold_train_rows": int(len(gold_train)),
        "holdout_rows": int(len(holdout)),
        "selected_c": c_value,
        "selected_weak_mass_ratio": weak_mass_ratio,
        "selected_oof_raw": metrics(gold_train["review_label"].to_numpy(), oof_raw),
        "selected_correction_policy": policy,
        "holdout_legacy": metrics(y_holdout, legacy_holdout),
        "holdout_raw_context": metrics(y_holdout, raw_holdout),
        "holdout_corrected_context": metrics(y_holdout, corrected_holdout),
        "warning": "Single-reviewer gold labels and post-audit engineering holdout; not independent publication evidence.",
    }
    (DATA_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
