"""Lock combined independent V22 emissions before content review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "outputs" / "assembly_stance" / "stance_context_grammatical_target_v22" / "confirmatory_10000_v22" / "context_predictions_v22.csv",
    ROOT / "outputs" / "assembly_stance" / "stance_context_grammatical_target_v22" / "holdout_5000_v22" / "context_predictions_v22.csv",
]
STATES = [
    ROOT / "data" / "shadow" / "stance_context_v22_confirmatory_10000" / "state.json",
    ROOT / "data" / "shadow" / "stance_context_v22_broad_holdout_5000" / "stance_context_v22_broad_holdout_5000.state.json",
]
V22_CODE = ROOT / "src" / "election_forecast" / "stance_context_v22.py"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v14.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v14.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V14 audit already exists")
    code_hash = _sha256(V22_CODE)
    for state_path in STATES:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["frozen_v22_sha256"] != code_hash:
            raise RuntimeError(f"V22 code changed after sampling: {state_path}")
    frames = [pd.read_csv(path, encoding="utf-8-sig", low_memory=False).fillna("") for path in SOURCES]
    combined = pd.concat(frames, ignore_index=True)
    audit = combined.loc[combined["v22_prediction"].ne("neutral")].copy()
    audit = audit.sort_values("text_sha256").reset_index(drop=True)
    audit.insert(0, "audit_id", [f"stance_v14_{index:03d}" for index in range(1, len(audit) + 1)])
    if len(audit) < 59 or audit["text_sha256"].duplicated().any():
        raise RuntimeError("V14 requires at least 59 unique emissions")
    audit.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    state = {
        "status": "independent_v22_audit_locked_before_review",
        "rows": len(audit),
        "prediction_counts": audit["v22_prediction"].value_counts().to_dict(),
        "source_sha256": [_sha256(path) for path in SOURCES],
        "frozen_v22_sha256": code_hash,
        "output_sha256_before_review": _sha256(OUTPUT),
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "active_forecast_changed": False,
    }
    LOCK.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
