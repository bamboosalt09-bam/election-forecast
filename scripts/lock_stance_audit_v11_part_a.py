"""Lock unseen V19 emissions remaining in supplement D."""

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
    / "stance_context_owner_resolution_v19"
    / "supplement_d_v19"
    / "context_predictions_v19.csv"
)
PRIOR = ROOT / "data" / "shadow" / "stance_locked_audit_v10_part_b.csv"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v11_part_a.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v11_part_a.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V11-A audit already exists; refusing to overwrite")
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig").fillna("")
    prior = pd.read_csv(PRIOR, encoding="utf-8-sig").fillna("")
    used = set(prior["text_sha256"].astype(str))
    candidates = frame.loc[
        frame["v19_prediction"].ne("neutral")
        & ~frame["text_sha256"].astype(str).isin(used)
    ].sort_values("text_sha256")
    if not 1 <= len(candidates) < 59:
        raise RuntimeError(f"V11-A expects 1-58 unseen emissions; found {len(candidates)}")
    audit = candidates.copy().reset_index(drop=True)
    audit.insert(
        0,
        "audit_id",
        [f"stance_v11a_{index:03d}" for index in range(1, len(audit) + 1)],
    )
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
        "model_version": "stance_context_owner_resolution_v19",
        "rows": int(len(audit)),
        "selection": "all unseen V19 directional hashes remaining in supplement D",
        "rule_frozen_before_review": True,
        "excluded_prior_hashes": len(used),
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
