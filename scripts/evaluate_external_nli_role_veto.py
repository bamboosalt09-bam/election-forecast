"""Use external Korean NLI role probabilities only as a stance veto.

The external model may abstain an existing directional prediction, but it may
never create or reverse one.  This isolates the useful target/owner signal from
the low-coverage polarity behavior observed in the standalone V26-S cascade.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from election_forecast.stance_precision import precision_first_metrics
from scripts.evaluate_external_nli_cascade import (
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    _classifier,
    _groups,
    _positive_probability,
    extract_nli_features,
    load_audits,
)


MODEL_VERSION = "stance_external_kornli_task_specific_veto_v27s"
SOURCE_COLUMNS = {
    1: "ensemble_prediction",
    2: "target_aware_prediction",
    3: "context_consensus_prediction",
    4: "context_prediction",
    5: "ambiguity_gated_prediction",
    6: "ambiguity_gated_prediction",
    7: "v15_prediction",
    8: "v16_prediction",
    9: "v17_prediction",
    10: "v18_prediction",
    11: "v19_prediction",
    12: "v20_prediction",
    13: "v21_prediction",
    14: "v22_prediction",
    15: "v23s_prediction",
    16: "v24s_prediction",
    17: "v25s_prediction",
}


@dataclass(frozen=True)
class VetoPolicy:
    target_threshold: float
    owner_threshold: float


def source_predictions(frame: pd.DataFrame) -> np.ndarray:
    prediction = np.full(len(frame), "neutral", dtype="<U8")
    for version, column in SOURCE_COLUMNS.items():
        matched = frame["audit_version"].eq(version)
        if not matched.any():
            continue
        if column not in frame.columns:
            raise ValueError(f"audit v{version} is missing {column}")
        values = frame.loc[matched, column].astype(str).to_numpy()
        if not np.isin(values, ["negative", "neutral", "positive"]).all():
            raise ValueError(f"audit v{version} has invalid source predictions")
        prediction[np.flatnonzero(matched.to_numpy())] = values
    return prediction


def apply_role_veto(
    source_prediction: np.ndarray,
    target_probability: np.ndarray,
    owner_probability: np.ndarray,
    policy: VetoPolicy,
) -> np.ndarray:
    output = np.asarray(source_prediction, dtype=str).astype("<U8")
    accepted = (
        (np.asarray(target_probability) >= policy.target_threshold)
        & (np.asarray(owner_probability) >= policy.owner_threshold)
    )
    output[~accepted] = "neutral"
    return output


def task_role_matrices(probabilities: np.ndarray, frame: pd.DataFrame):
    """Keep polarity evidence out of target and ownership decisions."""

    target_types = pd.get_dummies(
        pd.Categorical(frame["target_type"], categories=["person", "party", "government"]),
        dtype=float,
    ).to_numpy()
    target_matrix = np.hstack([probabilities[:, 0, :], target_types])
    owner_matrix = probabilities[:, 1:3, :].reshape(len(frame), -1)
    return target_matrix.astype(float), owner_matrix.astype(float)


def role_oof(
    frame: pd.DataFrame,
    target_matrix: np.ndarray,
    owner_matrix: np.ndarray,
    c_value: float,
):
    from sklearn.model_selection import StratifiedGroupKFold

    target = frame["target_truth"].astype(int).to_numpy()
    owner = frame["owner_truth"].astype(int).to_numpy()
    target_probability = np.zeros(len(frame), dtype=float)
    owner_probability = np.zeros(len(frame), dtype=float)
    groups = _groups(frame)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1729)
    for train, validation in splitter.split(target_matrix, frame["review_label"], groups):
        target_model = _classifier(c_value).fit(target_matrix[train], target[train])
        owner_model = _classifier(c_value).fit(owner_matrix[train], owner[train])
        target_probability[validation] = _positive_probability(
            target_model, target_matrix[validation]
        )
        owner_probability[validation] = _positive_probability(
            owner_model, owner_matrix[validation]
        )
    return target_probability, owner_probability


def select_policy(
    frame: pd.DataFrame,
    target_matrix: np.ndarray,
    owner_matrix: np.ndarray,
):
    source = source_predictions(frame)
    truth = frame["review_label"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    stores: dict[tuple[float, float, float], np.ndarray] = {}
    for c_value in (0.10, 0.50, 1.00, 2.00, 5.00):
        target_probability, owner_probability = role_oof(
            frame, target_matrix, owner_matrix, c_value
        )
        for target_threshold in np.round(np.arange(0.20, 0.951, 0.05), 2):
            for owner_threshold in np.round(np.arange(0.20, 0.951, 0.05), 2):
                policy = VetoPolicy(float(target_threshold), float(owner_threshold))
                prediction = apply_role_veto(
                    source, target_probability, owner_probability, policy
                )
                metrics = precision_first_metrics(truth, prediction)
                key = (c_value, policy.target_threshold, policy.owner_threshold)
                stores[key] = prediction
                rows.append({"c_value": c_value, **asdict(policy), **metrics})
    search = pd.DataFrame(rows)
    search["eligible_emissions"] = search["predicted_directional_rows"].ge(59)
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
    policy = VetoPolicy(
        float(selected.target_threshold), float(selected.owner_threshold)
    )
    key = (float(selected.c_value), policy.target_threshold, policy.owner_threshold)
    return float(selected.c_value), policy, stores[key], search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=ROOT / "data" / "shadow")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "outputs" / "assembly_stance" / "stance_external_kornli_role_cascade_v26s")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "assembly_stance" / MODEL_VERSION)
    parser.add_argument("--train-through", type=int, default=15)
    parser.add_argument("--evaluate-through", type=int, default=17)
    args = parser.parse_args()

    frame = load_audits(range(1, args.evaluate_through + 1), args.audit_root.resolve())
    development = frame.loc[frame["audit_version"].le(args.train_through)].reset_index(drop=True)
    holdout = frame.loc[frame["audit_version"].gt(args.train_through)].reset_index(drop=True)
    if holdout["text_sha256"].isin(set(development["text_sha256"])).any():
        raise ValueError("pseudo-holdout overlaps development texts")

    probabilities = extract_nli_features(
        frame,
        args.feature_dir.resolve() / "external_nli_probabilities.npz",
        batch_size=16,
        max_length=160,
        row_chunk_size=16,
        local_files_only=True,
    )
    target_matrix, owner_matrix = task_role_matrices(probabilities, frame)
    development_target = target_matrix[: len(development)]
    holdout_target = target_matrix[len(development) :]
    development_owner = owner_matrix[: len(development)]
    holdout_owner = owner_matrix[len(development) :]
    c_value, policy, development_prediction, search = select_policy(
        development, development_target, development_owner
    )

    target_model = _classifier(c_value).fit(
        development_target, development["target_truth"].astype(int)
    )
    owner_model = _classifier(c_value).fit(
        development_owner, development["owner_truth"].astype(int)
    )
    target_probability = _positive_probability(target_model, holdout_target)
    owner_probability = _positive_probability(owner_model, holdout_owner)
    holdout_source = source_predictions(holdout)
    holdout_prediction = apply_role_veto(
        holdout_source, target_probability, owner_probability, policy
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(output_dir / "policy_search.csv", index=False, encoding="utf-8-sig")
    output = holdout.copy()
    output["source_prediction"] = holdout_source
    output["target_probability"] = target_probability
    output["owner_probability"] = owner_probability
    output["veto_prediction"] = holdout_prediction
    output.to_csv(output_dir / "pseudo_holdout_predictions.csv", index=False, encoding="utf-8-sig")

    payload = {
        "status": "shadow_not_active",
        "model_version": MODEL_VERSION,
        "external_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
        },
        "method": "task_specific_external_target_owner_veto_only",
        "feature_separation": {
            "target_head": ["direct_target_nli", "target_type"],
            "owner_head": ["speaker_owned_nli", "reported_external_nli"],
            "polarity_features_for_role_heads": False,
        },
        "selection": {"c_value": c_value, "policy": asdict(policy)},
        "scope": {
            "development_rows": len(development),
            "pseudo_holdout_rows": len(holdout),
            "vote_outcomes_used": False,
            "post_2022_rows_present": False,
            "active_forecast_changed": False,
        },
        "development_source_baseline": precision_first_metrics(
            development["review_label"], source_predictions(development)
        ),
        "development_grouped_oof_veto": precision_first_metrics(
            development["review_label"], development_prediction
        ),
        "pseudo_holdout_source_baseline": precision_first_metrics(
            holdout["review_label"], holdout_source
        ),
        "pseudo_holdout_veto": precision_first_metrics(
            holdout["review_label"], holdout_prediction
        ),
        "pseudo_holdout_by_audit": {
            f"v{version}": {
                "source": precision_first_metrics(
                    output.loc[output["audit_version"].eq(version), "review_label"],
                    output.loc[output["audit_version"].eq(version), "source_prediction"],
                ),
                "veto": precision_first_metrics(
                    output.loc[output["audit_version"].eq(version), "review_label"],
                    output.loc[output["audit_version"].eq(version), "veto_prediction"],
                ),
            }
            for version in sorted(output["audit_version"].unique())
        },
        "adoption": {
            "promoted": False,
            "reason": "pseudo-holdout only; fresh independent audit required",
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
