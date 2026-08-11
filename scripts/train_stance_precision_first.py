"""Train a conservative through-2022 parliamentary stance classifier.

This is a shadow experiment. It never writes active forecast inputs. Model and
policy selection use development OOF predictions; the historical engineering
holdout is evaluated once after selection. Directional false positives and
sign reversals are lexicographically more costly than abstention to neutral.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import (  # noqa: E402
    PrecisionPolicy,
    apply_precision_policy,
    build_precision_features,
    compose_precision_input,
    neutral_information_features,
    precision_first_metrics,
    stance_adoption_assessment,
)


ALLOWED_ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
DEFAULT_GOLD_FILE = (
    ROOT / "data" / "shadow" / "stance_precision_gold_through2022.csv"
)
DEFAULT_WEAK_FILE = (
    ROOT / "data" / "shadow" / "stance_precision_weak_anchors_through2022.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "assembly_stance" / "precision_first_v1"
MODES = ("current_only", "current_context", "risk_aware_context")
C_VALUES = (0.25, 0.50, 1.00, 2.00)
WEAK_SCALES = (0.00, 0.02, 0.05, 0.10)
WEAK_REASONS = frozenset(
    {
        "anti_corruption_policy_not_attack",
        "implicit_direct_criticism",
        "implicit_direct_support",
        "praise_for_anti_corruption_action",
        "reported_defense",
    }
)
RANDOM_STATE = 20260717


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=4_000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def _fit_heads(
    matrix,
    labels: np.ndarray,
    c_value: float,
    sample_weight: np.ndarray | None = None,
):
    directional = (labels != "neutral").astype(int)
    direction_model = _classifier(c_value)
    direction_model.fit(matrix, directional, sample_weight=sample_weight)
    polarity_mask = labels != "neutral"
    polarity_model = _classifier(c_value)
    polarity_weight = None if sample_weight is None else sample_weight[polarity_mask]
    polarity_model.fit(
        matrix[polarity_mask],
        labels[polarity_mask],
        sample_weight=polarity_weight,
    )
    return direction_model, polarity_model


def _predict_heads(direction_model, polarity_model, matrix) -> tuple[np.ndarray, np.ndarray]:
    direction_positions = {
        int(label): index for index, label in enumerate(direction_model.classes_)
    }
    direction = direction_model.predict_proba(matrix)[:, direction_positions[1]]
    raw_polarity = polarity_model.predict_proba(matrix)
    polarity_positions = {
        str(label): index for index, label in enumerate(polarity_model.classes_)
    }
    polarity = np.column_stack(
        [
            raw_polarity[:, polarity_positions["negative"]],
            raw_polarity[:, polarity_positions["positive"]],
        ]
    )
    return direction, polarity


def _inputs(frame: pd.DataFrame, mode: str) -> pd.Series:
    return pd.Series(
        [compose_precision_input(row, mode) for row in frame.to_dict(orient="records")],
        index=frame.index,
        dtype=str,
    )


def _load_gold(split_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not split_path.exists():
        raise FileNotFoundError(f"missing frozen gold split: {split_path}")
    frame = pd.read_csv(split_path, encoding="utf-8-sig").fillna("")
    observed_elections = set(frame["election_id"].astype(str))
    forbidden = sorted(observed_elections.difference(ALLOWED_ELECTIONS))
    if forbidden:
        raise ValueError(f"gold file contains forbidden elections: {forbidden}")
    frame = frame.copy()
    if frame.empty:
        raise ValueError("no through-2022 gold rows")
    if set(frame["review_label"]) != {"negative", "neutral", "positive"}:
        raise ValueError("gold labels are incomplete")
    development = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    holdout = frame.loc[frame["split"].eq("holdout")].reset_index(drop=True)
    if development.empty or holdout.empty:
        raise ValueError("frozen development/holdout split is incomplete")
    return development, holdout


def _load_weak(weak_path: Path, gold: pd.DataFrame) -> pd.DataFrame:
    if not weak_path.exists():
        raise FileNotFoundError(f"missing weak anchor file: {weak_path}")
    frame = pd.read_csv(weak_path, encoding="utf-8-sig").fillna("")
    observed_elections = set(frame["election_id"].astype(str))
    forbidden = sorted(observed_elections.difference(ALLOWED_ELECTIONS))
    if forbidden:
        raise ValueError(f"weak anchor file contains forbidden elections: {forbidden}")
    observed_reasons = set(frame["weak_reason"].astype(str))
    forbidden_reasons = sorted(observed_reasons.difference(WEAK_REASONS))
    if forbidden_reasons:
        raise ValueError(f"weak anchor file contains unapproved reasons: {forbidden_reasons}")
    if not set(frame["training_label"]).issubset({"negative", "neutral", "positive"}):
        raise ValueError("weak anchor labels are invalid")
    overlap = set(frame["text_sha256"]).intersection(set(gold["text_sha256"]))
    if overlap:
        raise ValueError(f"weak anchors overlap gold text hashes: {len(overlap)}")
    return frame.reset_index(drop=True)


def _groups(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["source_file"].astype(str)
        + "|"
        + frame["meeting_date"].astype(str)
        + "|"
        + frame["committee"].astype(str)
    )


def _oof_probabilities(
    development: pd.DataFrame,
    weak: pd.DataFrame,
) -> dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]]:
    truth = development["review_label"].astype(str).to_numpy()
    groups = _groups(development)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    stores = {
        (mode, c_value, weak_scale): (
            np.zeros(len(development), dtype=float),
            np.zeros((len(development), 2), dtype=float),
        )
        for mode in MODES
        for c_value in C_VALUES
        for weak_scale in WEAK_SCALES
    }
    mode_inputs = {mode: _inputs(development, mode) for mode in MODES}
    weak_inputs = {mode: _inputs(weak, mode) for mode in MODES}
    weak_labels = weak["training_label"].astype(str).to_numpy()
    weak_confidence = weak["weak_confidence"].astype(float).to_numpy()
    for train_index, validation_index in cv.split(development, truth, groups):
        fold_truth = truth[train_index]
        for mode in MODES:
            gold_features = build_precision_features()
            gold_train_matrix = gold_features.fit_transform(
                mode_inputs[mode].iloc[train_index]
            )
            gold_validation_matrix = gold_features.transform(
                mode_inputs[mode].iloc[validation_index]
            )
            for c_value in C_VALUES:
                direction_model, polarity_model = _fit_heads(
                    gold_train_matrix, fold_truth, c_value
                )
                direction, polarity = _predict_heads(
                    direction_model, polarity_model, gold_validation_matrix
                )
                stores[(mode, c_value, 0.0)][0][validation_index] = direction
                stores[(mode, c_value, 0.0)][1][validation_index] = polarity

            augmented_inputs = pd.concat(
                [mode_inputs[mode].iloc[train_index], weak_inputs[mode]],
                ignore_index=True,
            )
            augmented_labels = np.concatenate([fold_truth, weak_labels])
            augmented_features = build_precision_features()
            augmented_train_matrix = augmented_features.fit_transform(augmented_inputs)
            augmented_validation_matrix = augmented_features.transform(
                mode_inputs[mode].iloc[validation_index]
            )
            for c_value in C_VALUES:
                for weak_scale in WEAK_SCALES[1:]:
                    sample_weight = np.concatenate(
                        [
                            np.ones(len(train_index), dtype=float),
                            weak_confidence * float(weak_scale),
                        ]
                    )
                    direction_model, polarity_model = _fit_heads(
                        augmented_train_matrix,
                        augmented_labels,
                        c_value,
                        sample_weight,
                    )
                    direction, polarity = _predict_heads(
                        direction_model,
                        polarity_model,
                        augmented_validation_matrix,
                    )
                    stores[(mode, c_value, weak_scale)][0][validation_index] = direction
                    stores[(mode, c_value, weak_scale)][1][validation_index] = polarity
    return stores


def _select_model_and_policy(
    development: pd.DataFrame,
    stores: dict[tuple[str, float, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[str, float, float, PrecisionPolicy, pd.DataFrame, np.ndarray]:
    truth = development["review_label"].astype(str).to_numpy()
    texts = development["text_excerpt"].astype(str).to_numpy()
    thresholds = tuple(np.round(np.arange(0.50, 0.951, 0.025), 3))
    rows: list[dict[str, object]] = []
    for (mode, c_value, weak_scale), (direction, polarity) in stores.items():
        for direction_threshold in thresholds:
            for polarity_threshold in thresholds:
                for risk_surcharge in (0.0, 0.05, 0.10, 0.15, 0.20):
                    policy = PrecisionPolicy(
                        direction_threshold=float(direction_threshold),
                        polarity_threshold=float(polarity_threshold),
                        risk_surcharge=float(risk_surcharge),
                    )
                    prediction = apply_precision_policy(
                        direction, polarity, texts, policy
                    )
                    metrics = precision_first_metrics(truth, prediction)
                    rows.append(
                        {
                            "mode": mode,
                            "c_value": c_value,
                            "weak_scale": weak_scale,
                            **policy.to_dict(),
                            **metrics,
                            "observed_zero_harmful_errors": metrics["harmful_error_count"] == 0,
                        }
                    )
    table = pd.DataFrame(rows)
    table = table.sort_values(
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
    mode = str(selected["mode"])
    c_value = float(selected["c_value"])
    weak_scale = float(selected["weak_scale"])
    policy = PrecisionPolicy(
        direction_threshold=float(selected["direction_threshold"]),
        polarity_threshold=float(selected["polarity_threshold"]),
        risk_surcharge=float(selected["risk_surcharge"]),
    )
    direction, polarity = stores[(mode, c_value, weak_scale)]
    prediction = apply_precision_policy(
        direction,
        polarity,
        texts,
        policy,
    )
    return mode, c_value, weak_scale, policy, table, prediction


def _fit_artifact(
    frame: pd.DataFrame,
    weak: pd.DataFrame,
    mode: str,
    c_value: float,
    weak_scale: float,
) -> dict[str, object]:
    values = _inputs(frame, mode)
    labels = frame["review_label"].astype(str).to_numpy()
    sample_weight = None
    if weak_scale > 0:
        values = pd.concat([values, _inputs(weak, mode)], ignore_index=True)
        labels = np.concatenate(
            [labels, weak["training_label"].astype(str).to_numpy()]
        )
        sample_weight = np.concatenate(
            [
                np.ones(len(frame), dtype=float),
                weak["weak_confidence"].astype(float).to_numpy() * weak_scale,
            ]
        )
    features = build_precision_features()
    matrix = features.fit_transform(values)
    direction_model, polarity_model = _fit_heads(
        matrix, labels, c_value, sample_weight
    )
    return {
        "features": features,
        "direction_model": direction_model,
        "polarity_model": polarity_model,
        "representation_mode": mode,
        "c_value": c_value,
        "weak_scale": weak_scale,
    }


def _predict_artifact(
    artifact: dict[str, object], frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    values = _inputs(frame, str(artifact["representation_mode"]))
    matrix = artifact["features"].transform(values)
    return _predict_heads(
        artifact["direction_model"], artifact["polarity_model"], matrix
    )


def _prediction_frame(
    frame: pd.DataFrame,
    direction: np.ndarray,
    polarity: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "audit_id",
        "election_id",
        "assembly_daesu",
        "meeting_date",
        "committee",
        "speaker",
        "issue_name",
        "target_type",
        "target_name",
        "text_excerpt",
        "context_before",
        "context_after",
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
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_GOLD_FILE)
    parser.add_argument("--weak-file", type=Path, default=DEFAULT_WEAK_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    split_path = args.gold_file.resolve()
    weak_path = args.weak_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development, holdout = _load_gold(split_path)
    all_gold = pd.concat([development, holdout], ignore_index=True)
    weak = _load_weak(weak_path, all_gold)
    stores = _oof_probabilities(development, weak)
    mode, c_value, weak_scale, policy, policy_search, development_prediction = (
        _select_model_and_policy(development, stores)
    )
    development_direction, development_polarity = stores[(mode, c_value, weak_scale)]
    development_metrics = precision_first_metrics(
        development["review_label"], development_prediction
    )

    development_artifact = _fit_artifact(
        development, weak, mode, c_value, weak_scale
    )
    holdout_direction, holdout_polarity = _predict_artifact(
        development_artifact, holdout
    )
    holdout_prediction = apply_precision_policy(
        holdout_direction,
        holdout_polarity,
        holdout["text_excerpt"].astype(str).to_numpy(),
        policy,
    )
    holdout_metrics = precision_first_metrics(
        holdout["review_label"], holdout_prediction
    )

    final_artifact = _fit_artifact(
        all_gold, weak, mode, c_value, weak_scale
    )
    final_artifact.update(
        {
            "model_version": "stance_precision_first_v1",
            "policy": asdict(policy),
            "approved_weak_reasons": sorted(WEAK_REASONS),
            "allowed_elections": list(ALLOWED_ELECTIONS),
            "active_forecast_integration": False,
        }
    )
    joblib.dump(final_artifact, output_dir / "stance_precision_first_v1.joblib")

    development_output = _prediction_frame(
        development,
        development_direction,
        development_polarity,
        development_prediction,
    )
    holdout_output = _prediction_frame(
        holdout,
        holdout_direction,
        holdout_polarity,
        holdout_prediction,
    )
    development_output.to_csv(
        output_dir / "development_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    holdout_output.to_csv(
        output_dir / "engineering_holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    policy_search.to_csv(
        output_dir / "policy_search.csv", index=False, encoding="utf-8-sig"
    )

    payload = {
        "status": "shadow_not_active",
        "model_version": "stance_precision_first_v1",
        "scope": {
            "allowed_elections": list(ALLOWED_ELECTIONS),
            "post_2022_rows_present": False,
            "vote_outcomes_used": False,
        },
        "data": {
            "source_split": str(split_path.relative_to(ROOT)),
            "source_sha256": _sha256(split_path),
            "weak_anchor_source": str(weak_path.relative_to(ROOT)),
            "weak_anchor_sha256": _sha256(weak_path),
            "weak_anchor_rows": len(weak),
            "development_rows": len(development),
            "engineering_holdout_rows": len(holdout),
            "holdout_warning": "historical engineering holdout; not publication-independent",
        },
        "selection": {
            "objective": "zero observed neutral-to-direction and wrong-direction errors before maximizing correct directional coverage",
            "representation_mode": mode,
            "c_value": c_value,
            "weak_scale": weak_scale,
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
