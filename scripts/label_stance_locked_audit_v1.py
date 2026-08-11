"""Write the completed manual labels for the already locked v1 audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v1.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v1.lock.json"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v1_labels.csv"

LABELS = (
    "neutral", "neutral", "negative", "positive", "neutral",
    "neutral", "negative", "positive", "negative", "positive",
    "positive", "negative", "neutral", "neutral", "negative",
    "positive", "neutral", "neutral", "negative", "positive",
    "negative", "positive", "negative", "neutral", "negative",
    "negative", "neutral", "negative", "positive", "neutral",
    "neutral", "negative", "positive", "neutral", "neutral",
    "neutral", "negative", "negative", "neutral", "neutral",
    "negative", "neutral", "neutral", "neutral", "positive",
    "positive", "negative", "negative", "neutral", "neutral",
    "negative", "positive", "neutral", "positive", "neutral",
    "negative", "positive", "negative", "negative", "neutral",
    "positive", "neutral", "negative", "negative", "negative",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "neutral", "positive", "negative", "negative", "neutral",
    "positive", "neutral", "neutral", "neutral", "neutral",
)

TARGET_CORRECT = (
    False, False, True, True, False,
    False, True, True, True, True,
    True, True, False, True, True,
    True, False, True, True, True,
    True, True, True, False, True,
    True, False, True, True, True,
    False, True, True, False, True,
    False, True, True, False, False,
    True, True, False, True, True,
    True, True, True, False, True,
    True, True, False, True, False,
    True, True, True, True, False,
    True, True, True, True, True,
    True, False, False, True, True,
    False, True, True, True, True,
    True, True, True, True, False,
)

REPORTED = {
    32,
    33,
    44,
    49,
    61,
    69,
    76,
}

NOTES = {
    0: "welcomes the nominee's adjustment, not the presidential office",
    1: "criticism is directed at the housing corporation",
    3: "speaker-owned defense of the administration's fiscal context",
    5: "party name identifies the speaker rather than the evaluated object",
    10: "speaker rejects the accusation against the candidate",
    12: "party affiliation identifies the speaker",
    15: "positive self-report of anti-corruption administration",
    20: "question contains a speaker-owned demand for apology",
    23: "party members are only the source of prior questions",
    26: "fragment does not evaluate the extracted government target",
    30: "support is for a policy and attributed to a foreign administration",
    32: "quoted attack is explicitly rejected by the speaker",
    33: "foreign-government support is reported, not speaker-owned",
    35: "requesting government support is not support for government",
    38: "policy advocacy without an evaluated government target",
    40: "speaker is skeptical of the candidate's no-problem claim",
    41: "party support is reported while criticism targets the minister",
    44: "speaker rebuts an allegation against the presidential office",
    49: "external favorable statement is reported",
    52: "rhetorical criticism targets a minister, not the president",
    54: "evaluation concerns a statute rather than government",
    59: "party name is the speaker subject",
    61: "external praise is reported",
    64: "wish follows explicit criticism of low pledge implementation",
    66: "party self-description is not external treatment",
    67: "procedural bill list only mentions the person as a co-sponsor",
    68: "support is for an action, not an evaluation of the party",
    69: "union condemnation of a party release is reported",
    70: "party self-description is not external treatment",
    73: "speaker criticizes the party for calling reform a political attack",
    76: "alleged partisan support is an incomplete reported clause",
    79: "party affiliation and appointment context, not party treatment",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"audit labels already exist: {OUTPUT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if _sha256(AUDIT) != lock["output_sha256_before_review"]:
        raise RuntimeError("locked audit changed before labels were written")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig").fillna("")
    if len(audit) != len(LABELS) or len(audit) != len(TARGET_CORRECT):
        raise RuntimeError("manual label count does not match locked audit")
    labels = pd.DataFrame(
        {
            "text_sha256": audit["text_sha256"].astype(str),
            "audit_locked_label": LABELS,
            "audit_target_correct": TARGET_CORRECT,
            "audit_quotation_owner": [
                "reported" if index in REPORTED else "speaker"
                for index in range(len(audit))
            ],
            "audit_notes": [NOTES.get(index, "direct contextual judgment") for index in range(len(audit))],
        }
    )
    labels.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(
        {
            "rows": len(labels),
            "labels": labels["audit_locked_label"].value_counts().to_dict(),
            "target_correct": labels["audit_target_correct"].value_counts().to_dict(),
            "output": str(OUTPUT),
            "sha256": _sha256(OUTPUT),
        }
    )


if __name__ == "__main__":
    main()
