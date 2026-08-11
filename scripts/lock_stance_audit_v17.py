"""Lock independent V25-S emissions before content review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "assembly_stance" / "stance_context_semantic_role_v25s" / "confirmatory_40000_v25s" / "context_predictions_v25s.csv"
SAMPLE_STATE = ROOT / "data" / "shadow" / "stance_context_v25s_confirmatory_40000" / "state.json"
CODE = ROOT / "src" / "election_forecast" / "stance_context_v25s.py"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v17.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v17.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V17 audit already exists")
    code_hash = _sha256(CODE)
    sample_state = json.loads(SAMPLE_STATE.read_text(encoding="utf-8"))
    if sample_state["frozen_v25s_sha256"] != code_hash:
        raise RuntimeError("V25-S code changed after confirmatory sampling")
    frame = pd.read_csv(SOURCE, encoding="utf-8-sig", low_memory=False).fillna("")
    audit = frame.loc[frame["v25s_prediction"].ne("neutral")].copy()
    audit = audit.sort_values("text_sha256").reset_index(drop=True)
    audit.insert(0, "audit_id", [f"stance_v17_{index:03d}" for index in range(1, len(audit) + 1)])
    if len(audit) < 59 or audit["text_sha256"].duplicated().any():
        raise RuntimeError("V17 requires at least 59 unique independent emissions")
    audit.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    state = {
        "status": "independent_v25s_audit_locked_before_review",
        "rows": len(audit),
        "prediction_counts": audit["v25s_prediction"].value_counts().to_dict(),
        "source_sha256": _sha256(SOURCE),
        "sample_state_sha256": _sha256(SAMPLE_STATE),
        "frozen_v25s_sha256": code_hash,
        "output_sha256_before_review": _sha256(OUTPUT),
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "active_forecast_changed": False,
    }
    LOCK.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
