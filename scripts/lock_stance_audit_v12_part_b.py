"""Lock all 36 V20 supplement-F emissions for a 73-row audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "assembly_stance" / "stance_context_strict_owner_v20" / "supplement_f_v20" / "context_predictions_v20.csv"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v12_part_b.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v12_part_b.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V12-B already exists; refusing to overwrite")
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig").fillna("")
    candidates = frame.loc[frame["v20_prediction"].ne("neutral")].sort_values("text_sha256")
    if len(candidates) != 36:
        raise RuntimeError(f"V12-B expects exactly 36 emissions; found {len(candidates)}")
    audit = candidates.copy().reset_index(drop=True)
    audit.insert(0, "audit_id", [f"stance_v12b_{i:03d}" for i in range(1, 37)])
    for column in ("audit_locked_label", "audit_target_correct", "audit_quotation_owner", "audit_notes"):
        audit[column] = ""
    audit.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    state = {
        "status": "locked_before_review",
        "model_version": "stance_context_strict_owner_v20",
        "rows": 36,
        "selection": "all V20 directional hashes from fresh supplement F",
        "rule_frozen_before_review": True,
        "combined_v12_rows": 73,
        "zero_error_upper_95_if_all_correct": 0.040206794253165846,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256_before_review": _sha256(OUTPUT),
    }
    LOCK.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
