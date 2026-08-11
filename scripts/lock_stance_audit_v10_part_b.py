"""Lock nine supplemental V18 emissions to complete audit V10."""

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
    / "stance_context_speaker_scope_v18"
    / "supplement_5000_v18"
    / "context_predictions_v18.csv"
)
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v10_part_b.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v10_part_b.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V10-B audit already exists; refusing to overwrite")
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig").fillna("")
    candidates = frame.loc[frame["v18_prediction"].ne("neutral")].sort_values("text_sha256")
    if len(candidates) < 9:
        raise RuntimeError(f"V10-B requires 9 V18 emissions; only {len(candidates)} available")
    audit = candidates.head(9).copy().reset_index(drop=True)
    audit.insert(0, "audit_id", [f"stance_v10b_{index:03d}" for index in range(1, 10)])
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
        "model_version": "stance_context_assertion_v18",
        "rows": 9,
        "selection": "lexicographically first 9 V18 directional text hashes from fresh supplement D",
        "rule_frozen_before_review": True,
        "directional_population": int(len(candidates)),
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
