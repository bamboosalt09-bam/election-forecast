"""Audit context stance classifier weaknesses without changing active inputs."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from election_forecast.stance_target_policy import (  # noqa: E402
    generic_legacy_label,
    target_aware_decision,
)
from scripts.evaluate_raw_stance_shadow import candidate_reference  # noqa: E402
from scripts.train_stance_context_model_5000 import (  # noqa: E402
    LABELS,
    RANDOM_STATE,
    fit_fold_model,
)


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
OUTPUT_DIR = DATA_DIR / "classifier_weakness_audit"
QUOTE_REPORT = re.compile(
    r"[‘’“”\"']|라고|다고|라는|이라며|말했|주장|보도|전했|회신|발표|답변"
)
NEGATION = re.compile(r"아니|않|없|못|말라|금지|아닙")
CONTRAST = re.compile(r"하지만|그러나|그런데|반면|불구하고|다만|오히려")
QUESTION = re.compile(r"[?？]|습니까|나요|아닌가|것 아니|겠습니까")
STANCE_TERM = re.compile(r"지지|찬성|환영|비판|비방|규탄|사퇴|퇴진|실패|잘못|불법|책임")


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - margin, center + margin


def _predict_oof() -> pd.DataFrame:
    split = pd.read_csv(DATA_DIR / "frozen_gold_split.csv", encoding="utf-8-sig").fillna("")
    weak = pd.read_csv(DATA_DIR / "weak_context_labels_5000.csv", encoding="utf-8-sig").fillna("")
    train = split.loc[split["split"].eq("train")].reset_index(drop=True)
    holdout = split.loc[split["split"].eq("holdout")].reset_index(drop=True)
    y = train["review_label"].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.zeros((len(train), len(LABELS)), dtype=float)
    fold_ids = np.zeros(len(train), dtype=int)
    for fold_id, (fit_index, validation_index) in enumerate(cv.split(train["model_input"], y), 1):
        model = fit_fold_model(
            train.iloc[fit_index],
            weak,
            c_value=0.5,
            weak_mass_ratio=0.5,
        )
        fold_probabilities = model.predict_proba(train.iloc[validation_index]["model_input"].astype(str))
        positions = {label: index for index, label in enumerate(model.named_steps["classifier"].classes_)}
        for label_index, label in enumerate(LABELS):
            probabilities[validation_index, label_index] = fold_probabilities[:, positions[label]]
        fold_ids[validation_index] = fold_id
    train_output = train.copy()
    train_output["evaluation_source"] = "five_fold_oof"
    train_output["evaluation_fold"] = fold_ids
    for index, label in enumerate(LABELS):
        train_output[f"probability_{label}"] = probabilities[:, index]

    holdout_predictions = pd.read_csv(
        DATA_DIR / "holdout_predictions.csv", encoding="utf-8-sig"
    ).fillna("")
    holdout_output = holdout.merge(
        holdout_predictions[
            ["audit_id", *[f"probability_{label}" for label in LABELS]]
        ],
        on="audit_id",
        how="left",
        validate="one_to_one",
    )
    holdout_output["evaluation_source"] = "frozen_holdout"
    holdout_output["evaluation_fold"] = 0
    output = pd.concat([train_output, holdout_output], ignore_index=True)
    probability_columns = [f"probability_{label}" for label in LABELS]
    probability_values = output[probability_columns].to_numpy(float)
    output["raw_prediction"] = np.asarray(LABELS)[np.argmax(probability_values, axis=1)]
    output["max_probability"] = probability_values.max(axis=1)
    sorted_probabilities = np.sort(probability_values, axis=1)
    output["probability_margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    output["legacy_prediction"] = output["stance_label"].map(generic_legacy_label)
    output["rule_stance_label"] = output["stance_label"]
    output["rule_stance_polarity"] = output["stance_polarity"]
    output["rule_stance_confidence"] = output["stance_confidence"]

    v2_labels: list[str] = []
    for row in output.to_dict(orient="records"):
        decision = target_aware_decision(
            row,
            model_label=str(row["raw_prediction"]),
            model_probability=float(row["max_probability"]),
            model_margin=float(row["probability_margin"]),
            allow_high_confidence_model_override=False,
        )
        v2_labels.append(decision.label)
    output["target_v2_prediction"] = v2_labels
    output["raw_correct"] = output["raw_prediction"].eq(output["review_label"]).astype(int)
    output["legacy_correct"] = output["legacy_prediction"].eq(output["review_label"]).astype(int)
    output["target_v2_correct"] = output["target_v2_prediction"].eq(output["review_label"]).astype(int)
    return output


def _metric_record(frame: pd.DataFrame, method: str, prediction_column: str, group: str, value: str) -> dict[str, object]:
    truth = frame["review_label"].astype(str)
    pred = frame[prediction_column].astype(str)
    successes = int(truth.eq(pred).sum())
    low, high = _wilson(successes, len(frame))
    return {
        "method": method,
        "group": group,
        "value": value,
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(truth, pred)),
        "accuracy_ci95_low": low,
        "accuracy_ci95_high": high,
        "macro_f1": float(f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(truth, pred, labels=LABELS)),
    }


def _subgroup_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["context_availability"] = np.select(
        [
            working["context_before"].astype(str).ne("") & working["context_after"].astype(str).ne(""),
            working["context_before"].astype(str).ne("") | working["context_after"].astype(str).ne(""),
        ],
        ["both", "one_side"],
        default="none",
    )
    working["quote_report_risk"] = working["text_excerpt"].astype(str).str.contains(QUOTE_REPORT).map(
        {True: "yes", False: "no"}
    )
    working["negation_risk"] = working["text_excerpt"].astype(str).str.contains(NEGATION).map(
        {True: "yes", False: "no"}
    )
    working["contrast_risk"] = working["text_excerpt"].astype(str).str.contains(CONTRAST).map(
        {True: "yes", False: "no"}
    )
    working["question_risk"] = working["text_excerpt"].astype(str).str.contains(QUESTION).map(
        {True: "yes", False: "no"}
    )
    working["explicit_stance_term"] = working["text_excerpt"].astype(str).str.contains(STANCE_TERM).map(
        {True: "yes", False: "no"}
    )
    lengths = working["text_excerpt"].astype(str).str.len()
    working["length_quartile"] = pd.qcut(lengths.rank(method="first"), 4, labels=["Q1_short", "Q2", "Q3", "Q4_long"])
    methods = {
        "legacy": "legacy_prediction",
        "raw_context": "raw_prediction",
        "target_v2": "target_v2_prediction",
    }
    group_columns = [
        "evaluation_source",
        "review_label",
        "target_type",
        "context_availability",
        "quote_report_risk",
        "negation_risk",
        "contrast_risk",
        "question_risk",
        "explicit_stance_term",
        "length_quartile",
        "election_id",
    ]
    records: list[dict[str, object]] = []
    for method, prediction_column in methods.items():
        records.append(_metric_record(working, method, prediction_column, "overall", "all"))
        for column in group_columns:
            for value, group in working.groupby(column, observed=True, sort=True):
                records.append(_metric_record(group, method, prediction_column, column, str(value)))
    return pd.DataFrame(records)


def _calibration(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001]
    labels = ["<0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", ">=0.90"]
    working = frame.copy()
    working["confidence_bin"] = pd.cut(
        working["max_probability"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    table = (
        working.groupby("confidence_bin", observed=True)
        .agg(
            rows=("raw_correct", "size"),
            mean_confidence=("max_probability", "mean"),
            accuracy=("raw_correct", "mean"),
            mean_margin=("probability_margin", "mean"),
        )
        .reset_index()
    )
    table["calibration_gap"] = table["mean_confidence"] - table["accuracy"]
    ece = float(
        ((table["rows"] / len(working)) * table["calibration_gap"].abs()).sum()
    )
    probabilities = working[[f"probability_{label}" for label in LABELS]].to_numpy(float)
    truth_indices = working["review_label"].map({label: index for index, label in enumerate(LABELS)}).to_numpy(int)
    one_hot = np.eye(len(LABELS))[truth_indices]
    summary = {
        "rows": int(len(working)),
        "top_label_ece": ece,
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "multiclass_log_loss": float(log_loss(working["review_label"], probabilities, labels=LABELS)),
        "mean_confidence_correct": float(working.loc[working["raw_correct"].eq(1), "max_probability"].mean()),
        "mean_confidence_wrong": float(working.loc[working["raw_correct"].eq(0), "max_probability"].mean()),
        "high_confidence_wrong_rows_ge_080": int(
            (working["raw_correct"].eq(0) & working["max_probability"].ge(0.80)).sum()
        ),
    }
    return table, summary


def _candidate_coverage() -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / "context_predictions_5000.csv", encoding="utf-8-sig").fillna("")
    candidates = candidate_reference()
    maps = {
        "person": {
            (str(row.election_id), str(row.candidate_name)): str(row.slot)
            for row in candidates.itertuples(index=False)
        },
        "party": {
            (str(row.election_id), str(row.party_name)): str(row.slot)
            for row in candidates.itertuples(index=False)
        },
    }
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        target_type = str(row.target_type)
        slot = maps.get(target_type, {}).get((str(row.election_id), str(row.target_name)), "")
        if not slot:
            continue
        rows.append(
            {
                "election_id": str(row.election_id),
                "slot": slot,
                "target_type": target_type,
                "model_label": str(row.context_model_label),
            }
        )
    mapped = pd.DataFrame(rows)
    coverage = (
        mapped.groupby(["election_id", "slot", "target_type", "model_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for label in LABELS:
        if label not in coverage:
            coverage[label] = 0
    coverage["rows"] = coverage[LABELS].sum(axis=1)
    return coverage


def _downstream_summary() -> pd.DataFrame:
    paths = {
        "legacy_and_context": DATA_DIR / "forecast_protocols" / "protocol_national_summary.csv",
        "hard_3class": DATA_DIR / "hard_label_forecast_protocols" / "protocol_national_summary.csv",
        "target_v2": DATA_DIR / "target_aware_v2_protocols" / "protocol_national_summary.csv",
    }
    pieces: list[pd.DataFrame] = []
    for source, path in paths.items():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.insert(0, "source", source)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True).drop_duplicates(
        ["variant", "protocol", "election_id"], keep="last"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = _predict_oof()
    subgroup = _subgroup_metrics(predictions)
    calibration, calibration_summary = _calibration(predictions)
    coverage = _candidate_coverage()
    downstream = _downstream_summary()
    legacy_only = int((predictions["legacy_correct"].eq(1) & predictions["raw_correct"].eq(0)).sum())
    raw_only = int((predictions["legacy_correct"].eq(0) & predictions["raw_correct"].eq(1)).sum())
    mapped_rows = int(coverage["rows"].sum())
    audit_summary = {
        "cross_fitted_rows": int(len(predictions)),
        "raw_context_accuracy": float(predictions["raw_correct"].mean()),
        "legacy_accuracy": float(predictions["legacy_correct"].mean()),
        "raw_only_correct": raw_only,
        "legacy_only_correct": legacy_only,
        "mcnemar_exact_two_sided_p": float(
            binomtest(legacy_only, n=legacy_only + raw_only, p=0.5).pvalue
        ),
        "neutral_false_direction_rows": int(
            (
                predictions["review_label"].eq("neutral")
                & ~predictions["raw_prediction"].eq("neutral")
            ).sum()
        ),
        "neutral_gold_rows": int(predictions["review_label"].eq("neutral").sum()),
        "current_candidate_or_party_rows_in_5000": mapped_rows,
        "current_target_coverage_share": mapped_rows / 5_000.0,
        "target_gold_rows": int(predictions["target_type"].isin(["person", "party"]).sum()),
        "active_engine_changed": False,
    }
    predictions.to_csv(OUTPUT_DIR / "cross_fitted_predictions_273.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(OUTPUT_DIR / "subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(OUTPUT_DIR / "calibration_bins.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUTPUT_DIR / "candidate_target_coverage.csv", index=False, encoding="utf-8-sig")
    downstream.to_csv(OUTPUT_DIR / "downstream_national_metrics.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "calibration_summary.json").write_text(
        json.dumps(calibration_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(subgroup.loc[subgroup["group"].isin(["overall", "review_label", "target_type", "quote_report_risk", "context_availability"])].to_string(index=False))
    print()
    print(json.dumps(calibration_summary, ensure_ascii=False, indent=2))
    print()
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
