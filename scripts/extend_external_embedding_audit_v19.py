"""Extend the unopened V29-S audit with additional untouched base emissions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.evaluate_external_embedding_role_veto import embedding_features  # noqa: E402
from scripts.evaluate_external_nli_cascade import _positive_probability, load_audits  # noqa: E402
from scripts.evaluate_external_nli_role_veto import apply_role_veto, VetoPolicy  # noqa: E402


MODEL_VERSION = "stance_external_embedding_role_veto_v29s"
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / MODEL_VERSION / "extension_v19"
ARTIFACT_PATH = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / MODEL_VERSION
    / f"{MODEL_VERSION}.joblib"
)
LOCK_V18 = ROOT / "data" / "shadow" / "stance_locked_audit_v18.csv"
LOCK_PATH = ROOT / "data" / "shadow" / "stance_locked_audit_v19.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "outputs" / "assembly_stance").rglob(
            "context_predictions.csv"
        )
        if MODEL_VERSION not in str(path)
    )


def candidates(excluded: set[str], paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False).fillna("")
        if "context_prediction" not in frame.columns:
            continue
        frame = frame.loc[
            frame["context_prediction"].astype(str).ne("neutral")
            & ~frame["text_sha256"].astype(str).isin(excluded)
        ].copy()
        frame["candidate_source"] = str(path.relative_to(ROOT))
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    return output.drop_duplicates("text_sha256", keep="first").reset_index(drop=True)


def main() -> None:
    if LOCK_PATH.exists():
        raise FileExistsError(LOCK_PATH)
    artifact = joblib.load(ARTIFACT_PATH)
    if artifact.get("active_forecast_integration") is not False:
        raise ValueError("artifact is not shadow-only")
    prior = load_audits(range(1, 18), ROOT / "data" / "shadow")
    v18 = pd.read_csv(LOCK_V18, encoding="utf-8-sig").fillna("")
    excluded = set(prior["text_sha256"].astype(str)) | set(v18["text_sha256"].astype(str))
    paths = source_files()
    frame = candidates(excluded, paths)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = embedding_features(
        frame,
        OUTPUT_DIR / "ko_sroberta_nli_embeddings.npz",
        batch_size=32,
    )
    target_probability = _positive_probability(artifact["target_model"], matrix)
    owner_probability = _positive_probability(artifact["owner_model"], matrix)
    policy = VetoPolicy(**artifact["policy"])
    source = frame["context_prediction"].astype(str).to_numpy()
    prediction = apply_role_veto(source, target_probability, owner_probability, policy)
    output = frame.copy()
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
        directional = directional.rename(columns={"schema_version": "source_schema_version"})
    directional.insert(0, "audit_id", [f"stance_v19_{index:03d}" for index in range(1, len(directional) + 1)])
    directional.insert(1, "schema_version", MODEL_VERSION)
    directional["selection_rank"] = np.arange(1, len(directional) + 1)
    directional["audit_locked_label"] = ""
    directional["audit_target_correct"] = ""
    directional["audit_quotation_owner"] = ""
    directional["audit_notes"] = ""
    directional.to_csv(LOCK_PATH, index=False, encoding="utf-8-sig")

    payload = {
        "status": "locked_before_review",
        "model_version": MODEL_VERSION,
        "artifact_path": str(ARTIFACT_PATH),
        "artifact_sha256": sha256_file(ARTIFACT_PATH),
        "prior_lock_v18_sha256": sha256_file(LOCK_V18),
        "source_manifest": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "candidate_rows": len(frame),
        "locked_directional_rows": len(directional),
        "combined_unopened_v18_v19_rows": len(v18) + len(directional),
        "lock_path": str(LOCK_PATH),
        "lock_sha256": sha256_file(LOCK_PATH),
        "policy_retrained_after_v18": False,
        "v18_labels_read": False,
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
