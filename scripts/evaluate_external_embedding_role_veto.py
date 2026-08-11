"""Evaluate Korean NLI sentence embeddings as a grouped role-veto model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from election_forecast.stance_precision import (  # noqa: E402
    compose_embedding_input,
    precision_first_metrics,
)
from scripts.evaluate_external_nli_cascade import (  # noqa: E402
    _classifier,
    _groups,
    _positive_probability,
    extract_nli_features,
    feature_matrix,
    load_audits,
)
from scripts.evaluate_external_nli_role_veto import (  # noqa: E402
    VetoPolicy,
    apply_role_veto,
    source_predictions,
)
from scripts.train_stance_context_encoder import encode_texts  # noqa: E402


MODEL_VERSION = "stance_external_embedding_role_veto_v28s"
ENCODER = {
    "backend": "sentence_transformer",
    "model_id": "jhgan/ko-sroberta-nli",
    "revision": "c4e15f24df2aceadfc931e2a57094726b2409861",
    "model_version": MODEL_VERSION,
}


def embedding_features(
    frame: pd.DataFrame,
    cache_path: Path,
    *,
    batch_size: int,
) -> np.ndarray:
    expected_hash = "\n".join(frame["text_sha256"].astype(str))
    import hashlib

    expected_hash = hashlib.sha256(expected_hash.encode("ascii")).hexdigest()
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            str(cached["row_hash"].item()) != expected_hash
            or str(cached["encoder_revision"].item()) != ENCODER["revision"]
        ):
            raise ValueError(f"invalid embedding cache: {cache_path}")
        return cached["embeddings"]
    texts = [
        compose_embedding_input(row, "risk_aware_context")
        for row in frame.to_dict(orient="records")
    ]
    matrix = encode_texts(
        texts,
        ENCODER,
        batch_size=batch_size,
        local_files_only=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            embeddings=matrix,
            row_hash=np.asarray(expected_hash),
            encoder_revision=np.asarray(ENCODER["revision"]),
        )
    temporary.replace(cache_path)
    return matrix


def role_oof(frame: pd.DataFrame, matrix: np.ndarray, c_value: float):
    from sklearn.model_selection import StratifiedGroupKFold

    target = frame["target_truth"].astype(int).to_numpy()
    owner = frame["owner_truth"].astype(int).to_numpy()
    target_probability = np.zeros(len(frame), dtype=float)
    owner_probability = np.zeros(len(frame), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1729)
    groups = _groups(frame)
    for train, validation in splitter.split(matrix, frame["review_label"], groups):
        target_model = _classifier(c_value).fit(matrix[train], target[train])
        owner_model = _classifier(c_value).fit(matrix[train], owner[train])
        target_probability[validation] = _positive_probability(
            target_model, matrix[validation]
        )
        owner_probability[validation] = _positive_probability(
            owner_model, matrix[validation]
        )
    return target_probability, owner_probability


def select_policy(frame: pd.DataFrame, matrix: np.ndarray, min_emissions: int):
    source = source_predictions(frame)
    truth = frame["review_label"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[float, float, float], np.ndarray] = {}
    for c_value in (0.01, 0.05, 0.10, 0.50, 1.00, 2.00):
        target_probability, owner_probability = role_oof(frame, matrix, c_value)
        for target_threshold in np.round(np.arange(0.20, 0.951, 0.05), 2):
            for owner_threshold in np.round(np.arange(0.20, 0.951, 0.05), 2):
                policy = VetoPolicy(float(target_threshold), float(owner_threshold))
                prediction = apply_role_veto(
                    source, target_probability, owner_probability, policy
                )
                metrics = precision_first_metrics(truth, prediction)
                key = (c_value, policy.target_threshold, policy.owner_threshold)
                predictions[key] = prediction
                rows.append({"c_value": c_value, **asdict(policy), **metrics})
    search = pd.DataFrame(rows)
    search["eligible_emissions"] = search["predicted_directional_rows"].ge(
        min_emissions
    )
    search = search.sort_values(
        [
            "eligible_emissions",
            "harmful_error_upper_95",
            "harmful_error_rate_among_emitted",
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
    return float(selected.c_value), policy, predictions[key], search


def evaluate_feature_set(
    name: str,
    frame: pd.DataFrame,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    matrix: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    development_matrix = matrix[: len(development)]
    holdout_matrix = matrix[len(development) :]
    min_development_emissions = int(
        np.ceil(59 * len(development) / max(len(holdout), 1))
    )
    c_value, policy, development_prediction, search = select_policy(
        development, development_matrix, min_development_emissions
    )
    target_model = _classifier(c_value).fit(
        development_matrix, development["target_truth"].astype(int)
    )
    owner_model = _classifier(c_value).fit(
        development_matrix, development["owner_truth"].astype(int)
    )
    target_probability = _positive_probability(target_model, holdout_matrix)
    owner_probability = _positive_probability(owner_model, holdout_matrix)
    source = source_predictions(holdout)
    prediction = apply_role_veto(source, target_probability, owner_probability, policy)
    search.to_csv(output_dir / f"policy_search_{name}.csv", index=False, encoding="utf-8-sig")
    output = holdout[[
        "audit_version",
        "text_sha256",
        "review_label",
        "audit_target_correct",
        "audit_quotation_owner",
        "text_excerpt",
    ]].copy()
    output["source_prediction"] = source
    output["target_probability"] = target_probability
    output["owner_probability"] = owner_probability
    output["veto_prediction"] = prediction
    output.to_csv(
        output_dir / f"pseudo_holdout_{name}.csv", index=False, encoding="utf-8-sig"
    )
    return {
        "c_value": c_value,
        "policy": asdict(policy),
        "min_development_emissions": min_development_emissions,
        "development_grouped_oof": precision_first_metrics(
            development["review_label"], development_prediction
        ),
        "pseudo_holdout": precision_first_metrics(holdout["review_label"], prediction),
        "pseudo_holdout_by_audit": {
            f"v{version}": precision_first_metrics(
                output.loc[output["audit_version"].eq(version), "review_label"],
                output.loc[output["audit_version"].eq(version), "veto_prediction"],
            )
            for version in sorted(output["audit_version"].unique())
        },
    }


def evaluate_dual_target_consensus(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    embeddings: np.ndarray,
    nli: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    """Require embedding role confidence plus an orthogonal NLI target gate."""

    n_development = len(development)
    development_embedding = embeddings[:n_development]
    holdout_embedding = embeddings[n_development:]
    development_nli = nli[:n_development]
    holdout_nli = nli[n_development:]
    embed_target, embed_owner = role_oof(development, development_embedding, 2.0)
    nli_target, _ = role_oof(development, development_nli, 0.1)
    source = source_predictions(development)
    truth = development["review_label"].astype(str).to_numpy()
    min_emissions = int(np.ceil(59 * len(development) / max(len(holdout), 1)))
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[float, float, float], np.ndarray] = {}
    thresholds = np.round(np.arange(0.20, 0.951, 0.05), 2)
    for embed_target_threshold in thresholds:
        for embed_owner_threshold in thresholds:
            for nli_target_threshold in thresholds:
                accepted = (
                    (embed_target >= embed_target_threshold)
                    & (embed_owner >= embed_owner_threshold)
                    & (nli_target >= nli_target_threshold)
                )
                prediction = source.copy()
                prediction[~accepted] = "neutral"
                metrics = precision_first_metrics(truth, prediction)
                key = (
                    float(embed_target_threshold),
                    float(embed_owner_threshold),
                    float(nli_target_threshold),
                )
                predictions[key] = prediction
                rows.append(
                    {
                        "embed_target_threshold": embed_target_threshold,
                        "embed_owner_threshold": embed_owner_threshold,
                        "nli_target_threshold": nli_target_threshold,
                        **metrics,
                    }
                )
    search = pd.DataFrame(rows)
    search["eligible_emissions"] = search["predicted_directional_rows"].ge(
        min_emissions
    )
    search = search.sort_values(
        [
            "eligible_emissions",
            "harmful_error_upper_95",
            "harmful_error_rate_among_emitted",
            "correct_direction_count",
            "predicted_directional_rows",
        ],
        ascending=[False, True, True, False, False],
    ).reset_index(drop=True)
    selected = search.iloc[0]
    key = (
        float(selected.embed_target_threshold),
        float(selected.embed_owner_threshold),
        float(selected.nli_target_threshold),
    )
    development_prediction = predictions[key]
    search.to_csv(
        output_dir / "policy_search_dual_target_consensus.csv",
        index=False,
        encoding="utf-8-sig",
    )

    embed_target_model = _classifier(2.0).fit(
        development_embedding, development["target_truth"].astype(int)
    )
    embed_owner_model = _classifier(2.0).fit(
        development_embedding, development["owner_truth"].astype(int)
    )
    nli_target_model = _classifier(0.1).fit(
        development_nli, development["target_truth"].astype(int)
    )
    holdout_embed_target = _positive_probability(
        embed_target_model, holdout_embedding
    )
    holdout_embed_owner = _positive_probability(embed_owner_model, holdout_embedding)
    holdout_nli_target = _positive_probability(nli_target_model, holdout_nli)
    holdout_source = source_predictions(holdout)
    accepted = (
        (holdout_embed_target >= key[0])
        & (holdout_embed_owner >= key[1])
        & (holdout_nli_target >= key[2])
    )
    holdout_prediction = holdout_source.copy()
    holdout_prediction[~accepted] = "neutral"
    output = holdout[[
        "audit_version",
        "text_sha256",
        "review_label",
        "audit_quotation_owner",
        "text_excerpt",
    ]].copy()
    output["source_prediction"] = holdout_source
    output["embed_target_probability"] = holdout_embed_target
    output["embed_owner_probability"] = holdout_embed_owner
    output["nli_target_probability"] = holdout_nli_target
    output["veto_prediction"] = holdout_prediction
    output.to_csv(
        output_dir / "pseudo_holdout_dual_target_consensus.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "fixed_role_c_values": {"embedding": 2.0, "nli_target": 0.1},
        "policy": {
            "embed_target_threshold": key[0],
            "embed_owner_threshold": key[1],
            "nli_target_threshold": key[2],
        },
        "min_development_emissions": min_emissions,
        "development_grouped_oof": precision_first_metrics(
            development["review_label"], development_prediction
        ),
        "pseudo_holdout": precision_first_metrics(
            holdout["review_label"], holdout_prediction
        ),
        "pseudo_holdout_by_audit": {
            f"v{version}": precision_first_metrics(
                output.loc[output["audit_version"].eq(version), "review_label"],
                output.loc[output["audit_version"].eq(version), "veto_prediction"],
            )
            for version in sorted(output["audit_version"].unique())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=ROOT / "data" / "shadow")
    parser.add_argument("--nli-feature-dir", type=Path, default=ROOT / "outputs" / "assembly_stance" / "stance_external_kornli_role_cascade_v26s")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "assembly_stance" / MODEL_VERSION)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    frame = load_audits(range(1, 18), args.audit_root.resolve())
    development = frame.loc[frame["audit_version"].le(15)].reset_index(drop=True)
    holdout = frame.loc[frame["audit_version"].gt(15)].reset_index(drop=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = embedding_features(
        frame, output_dir / "ko_sroberta_nli_embeddings.npz", batch_size=args.batch_size
    )
    nli_probabilities = extract_nli_features(
        frame,
        args.nli_feature_dir.resolve() / "external_nli_probabilities.npz",
        batch_size=16,
        max_length=160,
        row_chunk_size=16,
        local_files_only=True,
    )
    nli = feature_matrix(nli_probabilities, frame)
    feature_sets = {
        "embedding_only": embeddings,
        "embedding_plus_nli": np.hstack([embeddings, nli]),
    }
    results = {
        name: evaluate_feature_set(
            name, frame, development, holdout, matrix, output_dir
        )
        for name, matrix in feature_sets.items()
    }
    results["dual_target_consensus"] = evaluate_dual_target_consensus(
        development, holdout, embeddings, nli, output_dir
    )
    payload = {
        "status": "shadow_not_active",
        "model_version": MODEL_VERSION,
        "encoder": ENCODER,
        "method": "external_korean_nli_embedding_role_veto",
        "scope": {
            "development_rows": len(development),
            "pseudo_holdout_rows": len(holdout),
            "vote_outcomes_used": False,
            "post_2022_rows_present": False,
            "active_forecast_changed": False,
        },
        "source_baseline": precision_first_metrics(
            holdout["review_label"], source_predictions(holdout)
        ),
        "feature_ablation": results,
        "adoption": {
            "promoted": False,
            "reason": (
                "v16-v17 were inspected during consensus design and are now "
                "development diagnostics; fresh independent v18 audit required"
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
