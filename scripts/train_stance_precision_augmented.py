"""Train a shadow precision classifier with manually reviewed extra gold rows."""

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
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
from scripts.train_stance_precision_first import (  # noqa: E402
    ALLOWED_ELECTIONS,
    RANDOM_STATE,
    _classifier,
    _fit_heads,
    _groups,
    _load_gold,
    _predict_heads,
)


DEFAULT_GOLD = ROOT / "data" / "shadow" / "stance_precision_gold_through2022.csv"
DEFAULT_MANUAL = ROOT / "data" / "shadow" / "stance_manual_gold_expansion_v1.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "assembly_stance" / "precision_augmented_v3"
C_VALUES = (0.50, 1.00, 2.00)
MANUAL_SCALES = (0.75, 1.00, 1.50, 2.00)
RISK_SURCHARGES = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
MODE = "current_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [compose_precision_input(row, MODE) for row in frame.to_dict(orient="records")],
        index=frame.index,
        dtype=str,
    )


def _load_manual(path: Path, all_gold: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing manual expansion: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    required = {
        "election_id",
        "review_label",
        "review_target_correct",
        "text_sha256",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"manual expansion is missing required columns: {missing}")
    forbidden = sorted(set(frame["election_id"]).difference(ALLOWED_ELECTIONS))
    if forbidden:
        raise ValueError(f"manual expansion contains forbidden elections: {forbidden}")
    if set(frame["review_label"]) != {"negative", "neutral", "positive"}:
        raise ValueError("manual expansion must contain all three labels")
    target_correct = frame["review_target_correct"].astype(str).str.strip().str.lower()
    if not target_correct.eq("true").all():
        raise ValueError("manual expansion contains target-invalid rows")
    if frame["text_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").eq(False).any():
        raise ValueError("manual expansion contains invalid text hashes")
    if frame["text_sha256"].duplicated().any():
        raise ValueError("manual expansion contains duplicate text hashes")
    overlap = set(frame["text_sha256"]).intersection(set(all_gold["text_sha256"]))
    if overlap:
        raise ValueError(f"manual expansion overlaps original gold: {len(overlap)}")
    if "training_weight" not in frame:
        frame["training_weight"] = 1.0
    frame["training_weight"] = pd.to_numeric(
        frame["training_weight"], errors="coerce"
    )
    valid_weights = frame["training_weight"].between(0.0, 1.0, inclusive="right")
    if frame["training_weight"].isna().any() or not valid_weights.all():
        raise ValueError("manual expansion contains invalid training weights")
    return frame.reset_index(drop=True)


def _oof(
    development: pd.DataFrame,
    manual: pd.DataFrame,
) -> tuple[
    dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    list[int],
]:
    labels = development["review_label"].astype(str).to_numpy()
    manual_labels = manual["review_label"].astype(str).to_numpy()
    development_groups = _groups(development)
    manual_groups = _groups(manual)
    development_inputs = _inputs(development)
    manual_inputs = _inputs(manual)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    stores = {
        (c_value, manual_scale): (
            np.zeros(len(development), dtype=float),
            np.zeros((len(development), 2), dtype=float),
        )
        for c_value in C_VALUES
        for manual_scale in MANUAL_SCALES
    }
    manual_rows_per_fold: list[int] = []
    for train_index, validation_index in splitter.split(
        development,
        labels,
        development_groups,
    ):
        validation_groups = set(development_groups.iloc[validation_index])
        allowed_manual = ~manual_groups.isin(validation_groups)
        manual_rows_per_fold.append(int(allowed_manual.sum()))
        combined_inputs = pd.concat(
            [development_inputs.iloc[train_index], manual_inputs.loc[allowed_manual]],
            ignore_index=True,
        )
        combined_labels = np.concatenate(
            [labels[train_index], manual_labels[allowed_manual.to_numpy()]]
        )
        features = build_precision_features()
        train_matrix = features.fit_transform(combined_inputs)
        validation_matrix = features.transform(development_inputs.iloc[validation_index])
        gold_count = len(train_index)
        manual_count = int(allowed_manual.sum())
        manual_base_weight = manual.loc[allowed_manual, "training_weight"].to_numpy(
            dtype=float
        )
        for c_value in C_VALUES:
            for manual_scale in MANUAL_SCALES:
                sample_weight = np.concatenate(
                    [
                        np.ones(gold_count, dtype=float),
                        manual_base_weight * manual_scale,
                    ]
                )
                direction_model, polarity_model = _fit_heads(
                    train_matrix,
                    combined_labels,
                    c_value,
                    sample_weight,
                )
                direction, polarity = _predict_heads(
                    direction_model,
                    polarity_model,
                    validation_matrix,
                )
                stores[(c_value, manual_scale)][0][validation_index] = direction
                stores[(c_value, manual_scale)][1][validation_index] = polarity
    return stores, manual_rows_per_fold


def _select(
    development: pd.DataFrame,
    stores: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.Series, PrecisionPolicy, pd.DataFrame, np.ndarray]:
    labels = development["review_label"].astype(str).to_numpy()
    texts = development["text_excerpt"].astype(str).to_numpy()
    thresholds = tuple(np.round(np.arange(0.25, 0.951, 0.025), 3))
    rows: list[dict[str, object]] = []
    for (c_value, manual_scale), (direction, polarity) in stores.items():
        for direction_threshold in thresholds:
            for polarity_threshold in thresholds:
                for surcharge in RISK_SURCHARGES:
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
                            "c_value": c_value,
                            "manual_scale": manual_scale,
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
    direction, polarity = stores[
        (float(selected["c_value"]), float(selected["manual_scale"]))
    ]
    prediction = apply_precision_policy(direction, polarity, texts, policy)
    return selected, policy, table, prediction


def _fit_artifact(
    gold: pd.DataFrame,
    manual: pd.DataFrame,
    selected: pd.Series,
    policy: PrecisionPolicy,
    model_version: str = "stance_precision_augmented_v3",
) -> dict[str, object]:
    values = pd.concat([_inputs(gold), _inputs(manual)], ignore_index=True)
    labels = np.concatenate(
        [
            gold["review_label"].astype(str).to_numpy(),
            manual["review_label"].astype(str).to_numpy(),
        ]
    )
    manual_scale = float(selected["manual_scale"])
    sample_weight = np.concatenate(
        [
            np.ones(len(gold), dtype=float),
            manual["training_weight"].to_numpy(dtype=float) * manual_scale,
        ]
    )
    features = build_precision_features()
    matrix = features.fit_transform(values)
    direction, polarity = _fit_heads(
        matrix,
        labels,
        float(selected["c_value"]),
        sample_weight,
    )
    return {
        "model_version": model_version,
        "representation": MODE,
        "c_value": float(selected["c_value"]),
        "manual_scale": manual_scale,
        "policy": asdict(policy),
        "features": features,
        "direction_model": direction,
        "polarity_model": polarity,
        "active_forecast_integration": False,
    }


def _predict_artifact(
    artifact: dict[str, object],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = artifact["features"].transform(_inputs(frame))
    direction, polarity = _predict_heads(
        artifact["direction_model"],
        artifact["polarity_model"],
        matrix,
    )
    prediction = apply_precision_policy(
        direction,
        polarity,
        frame["text_excerpt"].astype(str).to_numpy(),
        PrecisionPolicy(**artifact["policy"]),
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
    parser.add_argument("--manual-file", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-version", default="stance_precision_augmented_v3")
    args = parser.parse_args()
    gold_path = args.gold_file.resolve()
    manual_path = args.manual_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development, holdout = _load_gold(gold_path)
    all_original_gold = pd.concat([development, holdout], ignore_index=True)
    manual = _load_manual(manual_path, all_original_gold)
    stores, manual_rows_per_fold = _oof(development, manual)
    selected, policy, search, development_prediction = _select(development, stores)
    key = (float(selected["c_value"]), float(selected["manual_scale"]))
    development_direction, development_polarity = stores[key]
    development_metrics = precision_first_metrics(
        development["review_label"], development_prediction
    )

    development_artifact = _fit_artifact(
        development, manual, selected, policy, args.model_version
    )
    holdout_direction, holdout_polarity, holdout_prediction = _predict_artifact(
        development_artifact,
        holdout,
    )
    holdout_metrics = precision_first_metrics(holdout["review_label"], holdout_prediction)

    final_artifact = _fit_artifact(
        all_original_gold, manual, selected, policy, args.model_version
    )
    final_artifact["allowed_elections"] = list(ALLOWED_ELECTIONS)
    joblib.dump(final_artifact, output_dir / f"{args.model_version}.joblib")
    _prediction_frame(
        development,
        development_direction,
        development_polarity,
        development_prediction,
    ).to_csv(
        output_dir / "development_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    _prediction_frame(
        holdout,
        holdout_direction,
        holdout_polarity,
        holdout_prediction,
    ).to_csv(
        output_dir / "engineering_holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    search.to_csv(output_dir / "policy_search.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "shadow_not_active",
        "model_version": args.model_version,
        "scope": {
            "allowed_elections": list(ALLOWED_ELECTIONS),
            "post_2022_rows_present": False,
            "vote_outcomes_used": False,
        },
        "data": {
            "original_gold_source": str(gold_path.relative_to(ROOT)),
            "original_gold_sha256": _sha256(gold_path),
            "manual_gold_source": str(manual_path.relative_to(ROOT)),
            "manual_gold_sha256": _sha256(manual_path),
            "manual_gold_rows": len(manual),
            "manual_label_distribution": {
                str(key): int(value)
                for key, value in manual["review_label"].value_counts().items()
            },
            "manual_target_distribution": {
                str(key): int(value)
                for key, value in manual["target_type"].value_counts().items()
            },
            "manual_rows_per_oof_fold": manual_rows_per_fold,
            "engineering_holdout_warning": "historical engineering holdout; not publication-independent",
        },
        "selection": {
            "objective": "zero observed harmful errors before maximizing correct directional coverage",
            "representation": MODE,
            "c_value": float(selected["c_value"]),
            "manual_scale": float(selected["manual_scale"]),
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
