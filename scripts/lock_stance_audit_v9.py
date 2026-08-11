"""Lock a confirmatory V17 audit before any manual review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_speaker_scope_v17"
    / "supplement_5000_v17"
    / "context_predictions_v17.csv"
)
PRIOR_AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_b.csv"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v9.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v9.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V9 audit already exists; refusing to overwrite")
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig").fillna("")
    prior = pd.read_csv(PRIOR_AUDIT, encoding="utf-8-sig").fillna("")
    used_hashes = set(prior["text_sha256"].astype(str))
    candidates = frame.loc[
        frame["v17_prediction"].ne("neutral")
        & ~frame["text_sha256"].astype(str).isin(used_hashes)
    ].sort_values("text_sha256")
    if len(candidates) < 59:
        raise RuntimeError(
            f"V9 requires 59 unseen V17 emissions; only {len(candidates)} available"
        )
    audit = candidates.head(59).copy().reset_index(drop=True)
    audit.insert(0, "audit_id", [f"stance_v9_{index:03d}" for index in range(1, 60)])
    for column in (
        "audit_locked_label",
        "audit_target_correct",
        "audit_quotation_owner",
        "audit_notes",
    ):
        audit[column] = ""
    audit.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    state = {
        "status": "locked_before_review",
        "model_version": "stance_context_speaker_scope_v17",
        "rows": 59,
        "selection": "lexicographically first 59 unseen V17 directional text hashes",
        "rule_frozen_before_review": True,
        "prior_audit_hashes_excluded": len(used_hashes),
        "unseen_directional_population": int(len(candidates)),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256_before_review": _sha256(OUTPUT),
    }
    LOCK.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
