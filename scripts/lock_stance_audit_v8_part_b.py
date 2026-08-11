"""Lock the hash-first V16 supplement emission before review."""

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
    / "stance_context_attribution_v16"
    / "supplement_5000_v16"
    / "context_predictions_v16.csv"
)
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_b.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_b.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V8-B audit already exists; refusing to overwrite")
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig").fillna("")
    candidates = frame.loc[frame["v16_prediction"].ne("neutral")].sort_values("text_sha256")
    if candidates.empty:
        raise RuntimeError("V8 supplement has no V16 directional emission")
    audit = candidates.head(1).copy().reset_index(drop=True)
    audit.insert(0, "audit_id", ["stance_v8b_001"])
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
        "model_version": "stance_context_attribution_v16",
        "rows": 1,
        "selection": "lexicographically first V16 directional text hash from fresh supplement B",
        "rule_frozen_before_review": True,
        "supplement_directional_population": int(len(candidates)),
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
