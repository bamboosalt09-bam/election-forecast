"""Lock combined independent V23-S emissions before content review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "outputs" / "assembly_stance" / "stance_context_pragmatic_role_v23s" / "confirmatory_10000_v23s" / "context_predictions_v23s.csv",
    ROOT / "outputs" / "assembly_stance" / "stance_context_pragmatic_role_v23s" / "holdout_5000_v23s" / "context_predictions_v23s.csv",
    ROOT / "outputs" / "assembly_stance" / "stance_context_pragmatic_role_v23s" / "holdout_final_5000_v23s" / "context_predictions_v23s.csv",
    ROOT / "outputs" / "assembly_stance" / "stance_context_pragmatic_role_v23s" / "targeted_holdout_5000_v23s" / "context_predictions_v23s.csv",
]
STATES = [
    ROOT / "data" / "shadow" / "stance_context_v23s_confirmatory_10000" / "state.json",
    ROOT / "data" / "shadow" / "stance_context_v23s_broad_holdout_5000" / "stance_context_v23s_broad_holdout_5000.state.json",
    ROOT / "data" / "shadow" / "stance_context_v23s_broad_holdout_final_5000" / "stance_context_v23s_broad_holdout_final_5000.state.json",
    ROOT / "data" / "shadow" / "stance_context_v23s_targeted_holdout_5000" / "state.json",
]
V23S_CODE = ROOT / "src" / "election_forecast" / "stance_context_v23s.py"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v15.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v15.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise FileExistsError("V15 audit already exists")
    code_hash = _sha256(V23S_CODE)
    for state_path in STATES:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["frozen_v23s_sha256"] != code_hash:
            raise RuntimeError(f"V23-S code changed after sampling: {state_path}")

    frames = [
        pd.read_csv(path, encoding="utf-8-sig", low_memory=False).fillna("")
        for path in SOURCES
    ]
    combined = pd.concat(frames, ignore_index=True)
    audit = combined.loc[combined["v23s_prediction"].ne("neutral")].copy()
    audit = audit.sort_values("text_sha256").reset_index(drop=True)
    audit.insert(0, "audit_id", [f"stance_v15_{index:03d}" for index in range(1, len(audit) + 1)])
    if len(audit) < 59 or audit["text_sha256"].duplicated().any():
        raise RuntimeError("V15 requires at least 59 unique independent emissions")
    audit.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    state = {
        "status": "independent_v23s_audit_locked_before_review",
        "rows": len(audit),
        "prediction_counts": audit["v23s_prediction"].value_counts().to_dict(),
        "source_sha256": [_sha256(path) for path in SOURCES],
        "sample_state_sha256": [_sha256(path) for path in STATES],
        "frozen_v23s_sha256": code_hash,
        "output_sha256_before_review": _sha256(OUTPUT),
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "active_forecast_changed": False,
    }
    LOCK.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
