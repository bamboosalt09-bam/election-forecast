"""Lock all 58 fresh V16 emissions before any review."""

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
    / "fresh_5000_v16"
    / "context_predictions_v16.csv"
)
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_a.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_a.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V8-A audit already exists; refusing to overwrite")
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig").fillna("")
    audit = frame.loc[frame["v16_prediction"].ne("neutral")].copy()
    audit = audit.sort_values("text_sha256").reset_index(drop=True)
    if len(audit) != 58 or audit["text_sha256"].duplicated().any():
        raise RuntimeError(f"expected 58 unique V16 emissions, found {len(audit)}")
    audit.insert(0, "audit_id", [f"stance_v8a_{index:03d}" for index in range(1, 59)])
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
        "rows": int(len(audit)),
        "selection": "all V16 directional emissions from fresh confirmatory corpus A",
        "rule_frozen_before_review": True,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256_before_review": _sha256(OUTPUT),
        "prediction_counts": audit["v16_prediction"].value_counts().to_dict(),
        "election_counts": audit["election_id"].value_counts().sort_index().to_dict(),
    }
    LOCK.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
