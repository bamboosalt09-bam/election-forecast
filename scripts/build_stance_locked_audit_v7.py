"""Lock every previously unaudited V15 directional emission before review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V15_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_ownership_v15"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v7.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v7.lock.json"
SOURCES = sorted(V15_DIR.glob("application*/context_predictions_v15.csv"))
MINIMUM_EMISSIONS = 59


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V7 audit or lock already exists; refusing to overwrite")
    frames = [pd.read_csv(path, encoding="utf-8-sig").fillna("") for path in SOURCES]
    combined = pd.concat(frames, ignore_index=True)
    prior_hashes: set[str] = set()
    for version in range(1, 7):
        prior = pd.read_csv(
            ROOT / "data" / "shadow" / f"stance_locked_audit_v{version}.csv",
            encoding="utf-8-sig",
            usecols=["text_sha256"],
        )
        prior_hashes.update(prior["text_sha256"].astype(str))
    audit = combined.loc[
        combined["v15_prediction"].ne("neutral")
        & ~combined["text_sha256"].astype(str).isin(prior_hashes)
    ].copy()
    audit = audit.sort_values("text_sha256").reset_index(drop=True)
    if len(audit) < MINIMUM_EMISSIONS:
        raise RuntimeError(
            f"need at least {MINIMUM_EMISSIONS} fresh emissions, found {len(audit)}"
        )
    if audit["text_sha256"].duplicated().any():
        raise RuntimeError("V7 audit contains duplicate text hashes")
    audit.insert(0, "audit_id", [f"stance_v7_{index:03d}" for index in range(1, len(audit) + 1)])
    for column in (
        "audit_locked_label",
        "audit_target_correct",
        "audit_quotation_owner",
        "audit_notes",
    ):
        audit[column] = ""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    state = {
        "status": "locked_before_review",
        "rows": int(len(audit)),
        "selection": "all V15 directional emissions absent from locked audits v1-v6",
        "minimum_precommitted_directional_emissions": MINIMUM_EMISSIONS,
        "model_version": "stance_context_ownership_v15",
        "rule_frozen_before_review": True,
        "source_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for path in SOURCES
        ],
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256_before_review": _sha256(OUTPUT),
        "election_counts": audit["election_id"].value_counts().sort_index().to_dict(),
        "target_counts": audit["target_type"].value_counts().to_dict(),
        "prediction_counts": audit["v15_prediction"].value_counts().to_dict(),
    }
    LOCK.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
