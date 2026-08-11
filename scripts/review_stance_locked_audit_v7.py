"""Write the manual V7 labels after verifying the pre-review lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v7.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v7.lock.json"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v7_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v7_labels.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v7_005": (
        "neutral",
        "true",
        "reported_external",
        "reports public and international support rather than owning the support",
    ),
    "stance_v7_007": (
        "neutral",
        "false",
        "historical_other_government",
        "the 2012 row concerns the prior Roh administration, not the assigned current government",
    ),
    "stance_v7_013": (
        "neutral",
        "true",
        "reported_external",
        "reports the opposition's criticism without adopting it",
    ),
    "stance_v7_018": (
        "neutral",
        "true",
        "reported_external_rebutted",
        "reports an opposition accusation that the following context rebuts",
    ),
    "stance_v7_022": (
        "neutral",
        "false",
        "foreign_government",
        "government alias resolves to the United States administration in the current sentence",
    ),
    "stance_v7_030": (
        "neutral",
        "true",
        "target_self_position",
        "the party states its own position rather than receiving an external evaluation",
    ),
    "stance_v7_052": (
        "neutral",
        "false",
        "historical_other_government",
        "the 2002 row concerns the prior Kim Young-sam administration",
    ),
}


def main() -> None:
    if LABELS.exists() or LABEL_LOCK.exists():
        raise FileExistsError("V7 labels already exist; refusing to overwrite")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    current_hash = _sha256(AUDIT)
    if current_hash != lock["output_sha256_before_review"]:
        raise RuntimeError("V7 audit changed after it was locked")
    frame = pd.read_csv(AUDIT, encoding="utf-8-sig").fillna("")
    records: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        if row.audit_id in OVERRIDES:
            label, target_correct, owner, notes = OVERRIDES[row.audit_id]
        else:
            label = str(row.v15_prediction)
            target_correct = "true"
            owner = "speaker"
            notes = "direct speaker-owned evaluation"
        records.append(
            {
                "text_sha256": str(row.text_sha256),
                "audit_locked_label": label,
                "audit_target_correct": target_correct,
                "audit_quotation_owner": owner,
                "audit_notes": notes,
            }
        )
    output = pd.DataFrame(records)
    if len(output) != int(lock["rows"]) or output["text_sha256"].duplicated().any():
        raise RuntimeError("V7 label cardinality does not match the locked audit")
    output.to_csv(LABELS, index=False, encoding="utf-8-sig")
    state = {
        "status": "review_complete_rule_remained_frozen",
        "rows": int(len(output)),
        "audit_sha256": current_hash,
        "labels_sha256": _sha256(LABELS),
        "label_counts": output["audit_locked_label"].value_counts().to_dict(),
        "target_correct_counts": output["audit_target_correct"].value_counts().to_dict(),
        "ownership_counts": output["audit_quotation_owner"].value_counts().to_dict(),
    }
    LABEL_LOCK.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
