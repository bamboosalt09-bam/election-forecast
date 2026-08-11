"""Freeze and apply the V29-S embedding role veto to untouched candidates."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from election_forecast.stance_precision import precision_first_metrics  # noqa: E402
from scripts.evaluate_external_embedding_role_veto import (  # noqa: E402
    ENCODER,
    embedding_features,
    select_policy,
)
from scripts.evaluate_external_nli_cascade import (  # noqa: E402
    _classifier,
    _positive_probability,
    load_audits,
)
from scripts.evaluate_external_nli_role_veto import (  # noqa: E402
    apply_role_veto,
    source_predictions,
)


MODEL_VERSION = "stance_external_embedding_role_veto_v29s"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / MODEL_VERSION
LOCK_PATH = ROOT / "data" / "shadow" / "stance_locked_audit_v18.csv"
SOURCE_FILES = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_lexical_role_v24s"
    / "confirmatory_40000_base"
    / "context_predictions.csv",
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_semantic_role_v25s"
    / "confirmatory_40000_base"
    / "context_predictions.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def untouched_candidates(audited_hashes: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in SOURCE_FILES:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False).fillna("")
        frame = frame.loc[
            frame["context_prediction"].astype(str).ne("neutral")
            & ~frame["text_sha256"].astype(str).isin(audited_hashes)
        ].copy()
        frame["candidate_source"] = str(path.relative_to(ROOT))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("text_sha256", keep="first").reset_index(drop=True)
    if combined["text_sha256"].astype(str).isin(audited_hashes).any():
        raise ValueError("candidate pool overlaps prior audits")
    return combined


def main() -> None:
    if LOCK_PATH.exists():
        raise FileExistsError(LOCK_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    development = load_audits(range(1, 18), ROOT / "data" / "shadow")
    candidates = untouched_candidates(set(development["text_sha256"].astype(str)))
    combined = pd.concat([development, candidates], ignore_index=True, sort=False).fillna("")
    embeddings = embedding_features(
        combined,
        OUTPUT_DIR / "ko_sroberta_nli_embeddings.npz",
        batch_size=32,
    )
    development_matrix = embeddings[: len(development)]
    candidate_matrix = embeddings[len(development) :]
    minimum_development_emissions = int(
        np.ceil(59 * len(development) / max(len(candidates), 1))
    )
    c_value, policy, oof_prediction, search = select_policy(
        development,
        development_matrix,
        minimum_development_emissions,
    )
    search.to_csv(OUTPUT_DIR / "policy_search.csv", index=False, encoding="utf-8-sig")

    target_model = _classifier(c_value).fit(
        development_matrix, development["target_truth"].astype(int)
    )
    owner_model = _classifier(c_value).fit(
        development_matrix, development["owner_truth"].astype(int)
    )
    target_probability = _positive_probability(target_model, candidate_matrix)
    owner_probability = _positive_probability(owner_model, candidate_matrix)
    source = candidates["context_prediction"].astype(str).to_numpy()
    prediction = apply_role_veto(source, target_probability, owner_probability, policy)
    output = candidates.copy()
    output["source_prediction"] = source
    output["target_probability"] = target_probability
    output["owner_probability"] = owner_probability
    output["v29_prediction"] = prediction
    output.to_csv(OUTPUT_DIR / "candidate_predictions.csv", index=False, encoding="utf-8-sig")

    directional = output.loc[output["v29_prediction"].ne("neutral")].copy()
    directional = directional.sort_values(
        ["target_probability", "owner_probability", "text_sha256"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    if "schema_version" in directional.columns:
        directional = directional.rename(
            columns={"schema_version": "source_schema_version"}
        )
    directional.insert(0, "audit_id", [f"stance_v18_{index:03d}" for index in range(1, len(directional) + 1)])
    directional.insert(1, "schema_version", MODEL_VERSION)
    directional["selection_rank"] = np.arange(1, len(directional) + 1)
    directional["audit_locked_label"] = ""
    directional["audit_target_correct"] = ""
    directional["audit_quotation_owner"] = ""
    directional["audit_notes"] = ""
    directional.to_csv(LOCK_PATH, index=False, encoding="utf-8-sig")

    artifact = {
        "model_version": MODEL_VERSION,
        "encoder": ENCODER,
        "c_value": c_value,
        "policy": asdict(policy),
        "target_model": target_model,
        "owner_model": owner_model,
        "active_forecast_integration": False,
        "vote_outcomes_used": False,
        "post_2022_rows_present": False,
    }
    joblib.dump(artifact, OUTPUT_DIR / f"{MODEL_VERSION}.joblib")
    script_path = Path(__file__).resolve()
    payload = {
        "status": "locked_before_review",
        "model_version": MODEL_VERSION,
        "script_sha256": sha256_file(script_path),
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in SOURCE_FILES
        ],
        "prior_audit_rows": len(development),
        "candidate_rows": len(candidates),
        "minimum_development_emissions": minimum_development_emissions,
        "selection": {"c_value": c_value, "policy": asdict(policy)},
        "development_grouped_oof": precision_first_metrics(
            development["review_label"], oof_prediction
        ),
        "locked_directional_rows": len(directional),
        "lock_path": str(LOCK_PATH),
        "lock_sha256": sha256_file(LOCK_PATH),
        "active_forecast_changed": False,
        "vote_outcomes_used": False,
        "post_2022_rows_present": False,
    }
    (OUTPUT_DIR / "lock_state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
