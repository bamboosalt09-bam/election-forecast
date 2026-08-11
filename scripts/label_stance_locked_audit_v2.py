"""Write manual labels for the already locked v8 audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v2.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v2.lock.json"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v2_labels.csv"

LABELS = (
    "negative", "positive", "negative", "negative", "negative", "neutral",
    "neutral", "negative", "negative", "negative", "negative", "neutral",
    "neutral", "neutral", "neutral", "negative", "negative", "negative",
    "negative", "negative", "negative", "neutral", "negative", "negative",
    "neutral", "neutral", "neutral", "neutral", "negative", "negative",
    "neutral", "negative", "neutral", "neutral", "neutral", "positive",
    "positive", "negative", "negative", "negative", "negative", "negative",
    "neutral", "negative", "negative", "negative", "negative", "neutral",
    "neutral", "negative", "negative", "negative", "negative", "negative",
    "positive", "neutral", "negative", "negative", "positive", "neutral",
    "negative", "negative", "negative", "negative", "neutral", "negative",
    "neutral", "negative", "neutral", "negative", "negative", "positive",
    "negative", "negative", "neutral", "negative", "negative", "neutral",
)

TARGET_CORRECT = (
    True, True, True, True, True, False,
    False, True, True, True, True, True,
    True, False, True, True, True, True,
    True, True, True, True, True, True,
    False, False, False, True, True, True,
    False, True, True, False, False, True,
    True, True, True, True, True, True,
    False, True, True, True, True, False,
    True, True, True, True, True, True,
    True, False, True, True, True, False,
    True, True, True, True, False, True,
    True, True, False, True, True, True,
    True, True, False, True, True, False,
)

REPORTED = {27}

NOTES = {
    1: "government-side coordination is evaluated as having no problem",
    5: "criticism targets private asset managers, not government",
    6: "generic political criticism without the extracted government target",
    11: "redistribution request without a signed government evaluation",
    12: "speaker explicitly says the issue is not unique to the current government",
    13: "demand targets another official, not the president",
    14: "speaker explicitly denies saying government was wrong",
    21: "conditional warning rather than a current signed presidential evaluation",
    24: "general discussion of public-official corruption",
    25: "criticism targets a prosecutorial argument",
    26: "program description without government treatment",
    27: "another candidate's apology is reported in a question",
    30: "general military-corruption discussion",
    32: "party members' prior request is described neutrally",
    33: "public demand for integrity without government treatment",
    34: "advice to a minister without signed government treatment",
    35: "speaker supports carrying out the government policy",
    36: "speaker defends the government-parliament agreement as proper",
    42: "criticism targets the ruling party committee, not government",
    47: "praise concerns a statutory amendment",
    48: "request for victim assistance, not criticism of government",
    54: "speaker favorably evaluates the person's sponsored bill",
    55: "reform demand targets named institutions rather than government",
    58: "speaker defends the central government's financing position",
    59: "the person is a quoted participant in floor disorder",
    64: "generic discussion of public office",
    66: "cautious bill review, not a signed party evaluation",
    68: "conditional corporation-duty statement, not government treatment",
    71: "speaker states there was no issue raised against government",
    74: "anti-corruption agency advocacy without government treatment",
    77: "criticism targets the rights commission rather than government",
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
        raise RuntimeError("locked v2 audit changed before labeling")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig").fillna("")
    if len(audit) != len(LABELS) or len(audit) != len(TARGET_CORRECT):
        raise RuntimeError("manual label count does not match locked v2 audit")
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
