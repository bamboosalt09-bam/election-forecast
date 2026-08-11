"""Train and evaluate stance v3 with explicit class balancing and context selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from election_forecast.stance_text_model import build_stance_pipeline  # noqa: E402
from election_forecast.stance_v3 import (  # noqa: E402
    compose_v3_input,
    directional_abstention,
    ownership_abstention,
    temperature_scale,
)


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
OUTPUT_DIR = DATA_DIR / "stance_v3"
LABELS = np.asarray(["negative", "neutral", "positive"])
RANDOM_STATE = 20260714
MODES = ("current_only", "nearest_context", "risk_aware_nearest")
WEAK_MASS_RATIOS = (0.0, 0.25, 0.50)


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    neutral_mask = truth == "neutral"
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(truth, prediction, labels=LABELS)),
        "neutral_false_direction_rate": float(
            np.mean(prediction[neutral_mask] != "neutral") if neutral_mask.any() else 0.0
        ),
    }


def _explicit_weights(
    gold_labels: pd.Series,
    weak: pd.DataFrame,
    weak_mass_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    gold_labels = gold_labels.astype(str)
    gold_counts = gold_labels.value_counts()
    gold_weights = gold_labels.map(
        {label: len(gold_labels) / (len(LABELS) * gold_counts[label]) for label in LABELS}
    ).to_numpy(float)
    weak_weights = np.zeros(len(weak), dtype=float)
    if weak_mass_ratio <= 0.0:
        return gold_weights, weak_weights
    total_weak_mass = weak_mass_ratio * gold_weights.sum()
    for label in LABELS:
        mask = weak["training_label"].astype(str).eq(label).to_numpy()
        raw = pd.to_numeric(weak.loc[mask, "weak_confidence"], errors="coerce").fillna(0.0).to_numpy(float)
        if raw.sum() > 0.0:
            weak_weights[mask] = raw * ((total_weak_mass / len(LABELS)) / raw.sum())
    return gold_weights, weak_weights


def _fit_from_matrices(
    features,
    combined_matrix,
    gold: pd.DataFrame,
    weak: pd.DataFrame,
    weak_mass_ratio: float,
) -> Pipeline:
    gold_weights, weak_weights = _explicit_weights(gold["review_label"], weak, weak_mass_ratio)
    classifier = LogisticRegression(
        C=0.5,
        class_weight=None,
        max_iter=3_000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    if weak_mass_ratio > 0.0:
        labels = pd.concat([gold["review_label"], weak["training_label"]], ignore_index=True).astype(str)
        weights = np.concatenate([gold_weights, weak_weights])
        classifier.fit(combined_matrix, labels, sample_weight=weights)
    else:
        classifier.fit(
            combined_matrix[: len(gold)],
            gold["review_label"].astype(str),
            sample_weight=gold_weights,
        )
    return Pipeline([("features", features), ("classifier", classifier)])


def _fit_model(
    gold: pd.DataFrame,
    weak: pd.DataFrame,
    mode: str,
    weak_mass_ratio: float,
) -> Pipeline:
    gold_input = pd.Series([compose_v3_input(row, mode) for row in gold.to_dict(orient="records")])
    weak_input = pd.Series([compose_v3_input(row, mode) for row in weak.to_dict(orient="records")])
    combined_input = pd.concat([gold_input, weak_input], ignore_index=True)
    features = build_stance_pipeline(c_value=0.5).named_steps["features"]
    combined_matrix = features.fit_transform(combined_input.astype(str))
    return _fit_from_matrices(features, combined_matrix, gold, weak, weak_mass_ratio)


def _aligned_probabilities(model: Pipeline, values: pd.Series) -> np.ndarray:
    raw = model.predict_proba(values.astype(str))
    positions = {label: index for index, label in enumerate(model.named_steps["classifier"].classes_)}
    return np.column_stack([raw[:, positions[label]] for label in LABELS])


def _apply_policy(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    min_probability: float,
    min_margin: float,
) -> tuple[np.ndarray, list[str]]:
    prediction = directional_abstention(
        probabilities,
        LABELS,
        min_probability=min_probability,
        min_margin=min_margin,
    )
    reasons: list[str] = []
    for index, text in enumerate(frame["text_excerpt"]):
        prediction[index], reason = ownership_abstention(text, str(prediction[index]))
        reasons.append(reason)
    return prediction, reasons


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(DATA_DIR / "frozen_gold_split.csv", encoding="utf-8-sig").fillna("")
    weak = pd.read_csv(DATA_DIR / "weak_context_labels_5000.csv", encoding="utf-8-sig").fillna("")
    development = split.loc[split["split"].eq("train")].reset_index(drop=True)
    holdout = split.loc[split["split"].eq("holdout")].reset_index(drop=True)
    truth = development["review_label"].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probability_store = {
        (mode, ratio): np.zeros((len(development), len(LABELS)), dtype=float)
        for mode in MODES
        for ratio in WEAK_MASS_RATIOS
    }

    weak_inputs = {
        mode: pd.Series([compose_v3_input(row, mode) for row in weak.to_dict(orient="records")])
        for mode in MODES
    }
    development_inputs = {
        mode: pd.Series(
            [compose_v3_input(row, mode) for row in development.to_dict(orient="records")]
        )
        for mode in MODES
    }
    for fit_index, validation_index in cv.split(development, truth):
        fold_gold = development.iloc[fit_index].reset_index(drop=True)
        for mode in MODES:
            fold_input = development_inputs[mode].iloc[fit_index].reset_index(drop=True)
            combined_input = pd.concat([fold_input, weak_inputs[mode]], ignore_index=True)
            features = build_stance_pipeline(c_value=0.5).named_steps["features"]
            combined_matrix = features.fit_transform(combined_input.astype(str))
            validation_matrix = features.transform(development_inputs[mode].iloc[validation_index].astype(str))
            for ratio in WEAK_MASS_RATIOS:
                model = _fit_from_matrices(features, combined_matrix, fold_gold, weak, ratio)
                raw = model.named_steps["classifier"].predict_proba(validation_matrix)
                positions = {
                    label: index for index, label in enumerate(model.named_steps["classifier"].classes_)
                }
                probability_store[(mode, ratio)][validation_index] = np.column_stack(
                    [raw[:, positions[label]] for label in LABELS]
                )

    search_rows: list[dict[str, object]] = []
    for (mode, ratio), probabilities in probability_store.items():
        prediction = LABELS[np.argmax(probabilities, axis=1)]
        search_rows.append({"mode": mode, "weak_mass_ratio": ratio, **_metrics(truth, prediction)})
    search = pd.DataFrame(search_rows).sort_values(
        ["macro_f1", "accuracy", "neutral_false_direction_rate"],
        ascending=[False, False, True],
    )
    selected = search.iloc[0]
    selected_key = (str(selected["mode"]), float(selected["weak_mass_ratio"]))
    development_probabilities = probability_store[selected_key]

    temperature_rows: list[dict[str, float]] = []
    for temperature in np.arange(0.60, 2.01, 0.05):
        scaled = temperature_scale(development_probabilities, float(temperature))
        temperature_rows.append(
            {
                "temperature": float(temperature),
                "log_loss": float(log_loss(truth, scaled, labels=LABELS)),
            }
        )
    temperature_table = pd.DataFrame(temperature_rows).sort_values("log_loss")
    temperature = float(temperature_table.iloc[0]["temperature"])
    calibrated_development = temperature_scale(development_probabilities, temperature)

    threshold_rows: list[dict[str, object]] = []
    for min_probability in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for min_margin in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            prediction, _ = _apply_policy(
                development, calibrated_development, min_probability, min_margin
            )
            threshold_rows.append(
                {
                    "min_probability": min_probability,
                    "min_margin": min_margin,
                    **_metrics(truth, prediction),
                }
            )
    threshold_table = pd.DataFrame(threshold_rows).sort_values(
        ["macro_f1", "accuracy", "neutral_false_direction_rate", "min_probability", "min_margin"],
        ascending=[False, False, True, False, False],
    )
    policy = threshold_table.iloc[0]
    min_probability = float(policy["min_probability"])
    min_margin = float(policy["min_margin"])
    development_prediction, development_reasons = _apply_policy(
        development, calibrated_development, min_probability, min_margin
    )

    holdout_model = _fit_model(development, weak, selected_key[0], selected_key[1])
    holdout_input = pd.Series(
        [compose_v3_input(row, selected_key[0]) for row in holdout.to_dict(orient="records")]
    )
    holdout_probabilities = temperature_scale(
        _aligned_probabilities(holdout_model, holdout_input), temperature
    )
    holdout_prediction, holdout_reasons = _apply_policy(
        holdout, holdout_probabilities, min_probability, min_margin
    )

    all_gold = pd.concat([development, holdout], ignore_index=True)
    final_model = _fit_model(all_gold, weak, selected_key[0], selected_key[1])
    artifact = {
        "model": final_model,
        "classes": LABELS.tolist(),
        "model_version": "stance_context_v3",
        "representation_mode": selected_key[0],
        "weak_mass_ratio": selected_key[1],
        "temperature": temperature,
        "min_probability": min_probability,
        "min_margin": min_margin,
        "weighting": "explicit gold-class balance plus separately balanced capped weak mass",
    }
    joblib.dump(artifact, OUTPUT_DIR / "stance_context_v3.joblib")

    evaluation = {
        "status": "trained_shadow",
        "selected_mode": selected_key[0],
        "selected_weak_mass_ratio": selected_key[1],
        "temperature": temperature,
        "min_probability": min_probability,
        "min_margin": min_margin,
        "development_oof": _metrics(truth, development_prediction),
        "engineering_holdout": _metrics(
            holdout["review_label"].astype(str).to_numpy(), holdout_prediction
        ),
        "holdout_warning": "historical engineering holdout; not publication-independent",
        "new_target_holdout_opened": False,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    search.to_csv(OUTPUT_DIR / "representation_search.csv", index=False, encoding="utf-8-sig")
    temperature_table.to_csv(OUTPUT_DIR / "temperature_search.csv", index=False, encoding="utf-8-sig")
    threshold_table.to_csv(OUTPUT_DIR / "policy_search.csv", index=False, encoding="utf-8-sig")
    development_output = development[["audit_id", "review_label", "text_excerpt", "target_type"]].copy()
    development_output["prediction"] = development_prediction
    development_output["ownership_reason"] = development_reasons
    for index, label in enumerate(LABELS):
        development_output[f"probability_{label}"] = calibrated_development[:, index]
    development_output.to_csv(OUTPUT_DIR / "development_oof_predictions.csv", index=False, encoding="utf-8-sig")
    holdout_output = holdout[["audit_id", "review_label", "text_excerpt", "target_type"]].copy()
    holdout_output["prediction"] = holdout_prediction
    holdout_output["ownership_reason"] = holdout_reasons
    for index, label in enumerate(LABELS):
        holdout_output[f"probability_{label}"] = holdout_probabilities[:, index]
    holdout_output.to_csv(OUTPUT_DIR / "engineering_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print()
    print(search.to_string(index=False))


if __name__ == "__main__":
    main()
