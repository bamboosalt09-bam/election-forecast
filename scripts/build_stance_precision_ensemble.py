"""Build a shadow precision ensemble from TF-IDF and embedding predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import (  # noqa: E402
    ConsensusPolicy,
    combine_precision_children,
    precision_first_metrics,
    stance_adoption_assessment,
)


DEFAULT_FIRST = ROOT / "outputs" / "assembly_stance" / "precision_first_v1"
DEFAULT_SECOND = ROOT / "outputs" / "assembly_stance" / "precision_embedding_v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "assembly_stance" / "precision_ensemble_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pair(first: Path, second: Path, filename: str) -> pd.DataFrame:
    first_frame = pd.read_csv(first / filename, encoding="utf-8-sig")
    second_frame = pd.read_csv(second / filename, encoding="utf-8-sig")
    columns = [
        "audit_id",
        "direction_probability",
        "polarity_probability_negative",
        "polarity_probability_positive",
        "precision_prediction",
    ]
    return first_frame.merge(
        second_frame[columns],
        on="audit_id",
        how="inner",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    first = np.column_stack(
        [
            frame["polarity_probability_negative_first"],
            frame["polarity_probability_positive_first"],
        ]
    )
    second = np.column_stack(
        [
            frame["polarity_probability_negative_second"],
            frame["polarity_probability_positive_second"],
        ]
    )
    return first, second


def _apply(frame: pd.DataFrame, policy: ConsensusPolicy) -> tuple[np.ndarray, np.ndarray]:
    first_polarity, second_polarity = _arrays(frame)
    return combine_precision_children(
        frame["precision_prediction_first"],
        frame["precision_prediction_second"],
        frame["direction_probability_first"].to_numpy(),
        frame["direction_probability_second"].to_numpy(),
        first_polarity,
        second_polarity,
        frame["text_excerpt"].astype(str).to_numpy(),
        policy,
    )


def _select(development: pd.DataFrame) -> tuple[ConsensusPolicy, pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    direction_thresholds = np.round(np.arange(0.30, 0.901, 0.025), 3)
    polarity_thresholds = np.round(np.arange(0.30, 0.901, 0.025), 3)
    sign_margins = np.round(np.arange(0.0, 0.501, 0.05), 3)
    for direction_threshold in direction_thresholds:
        for polarity_threshold in polarity_thresholds:
            for sign_margin in sign_margins:
                policy = ConsensusPolicy(
                    float(direction_threshold),
                    float(polarity_threshold),
                    float(sign_margin),
                )
                prediction, source = _apply(development, policy)
                metrics = precision_first_metrics(development["review_label"], prediction)
                key = (
                    policy.direction_threshold,
                    policy.polarity_threshold,
                    policy.sign_margin,
                )
                predictions[key] = (prediction, source)
                rows.append(
                    {
                        **policy.to_dict(),
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
            "sign_margin",
        ],
        ascending=[False, True, False, False, False, False, False],
    ).reset_index(drop=True)
    selected = table.iloc[0]
    policy = ConsensusPolicy(
        float(selected["direction_threshold"]),
        float(selected["polarity_threshold"]),
        float(selected["sign_margin"]),
    )
    prediction, source = predictions[
        (policy.direction_threshold, policy.polarity_threshold, policy.sign_margin)
    ]
    return policy, table, prediction, source


def _output(frame: pd.DataFrame, prediction: np.ndarray, source: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    out["ensemble_prediction"] = prediction
    out["ensemble_source"] = source
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dir", type=Path, default=DEFAULT_FIRST)
    parser.add_argument("--second-dir", type=Path, default=DEFAULT_SECOND)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-version", default="stance_precision_ensemble_v2")
    args = parser.parse_args()
    first = args.first_dir.resolve()
    second = args.second_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development = _load_pair(first, second, "development_oof_predictions.csv")
    holdout = _load_pair(first, second, "engineering_holdout_predictions.csv")
    policy, search, development_prediction, development_source = _select(development)
    holdout_prediction, holdout_source = _apply(holdout, policy)
    development_metrics = precision_first_metrics(
        development["review_label"], development_prediction
    )
    holdout_metrics = precision_first_metrics(holdout["review_label"], holdout_prediction)

    _output(development, development_prediction, development_source).to_csv(
        output_dir / "development_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    _output(holdout, holdout_prediction, holdout_source).to_csv(
        output_dir / "engineering_holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    search.to_csv(output_dir / "consensus_search.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "shadow_not_active",
        "model_version": str(args.model_version),
        "scope": {
            "post_2022_rows_present": False,
            "vote_outcomes_used": False,
        },
        "children": {
            "first_metrics_sha256": _sha256(first / "metrics.json"),
            "second_metrics_sha256": _sha256(second / "metrics.json"),
        },
        "selection": {
            "objective": "zero observed harmful errors before maximizing directional coverage",
            "policy": policy.to_dict(),
            "consensus_risk_policy": "hard_abstention",
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
