"""Write manual labels for the v13 follow-up audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v6.csv"
OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v6_labels.csv"
NEUTRAL = {
    "4a39ec0113": (True, "reported_external", "reports an aide's statement and article content"),
    "0b22bc3fb8": (False, "target_self_report", "states the government's own expectation"),
    "734b78cf3e": (True, "reported_external", "reports someone else's slander of Lee"),
    "3d926cc1e5": (True, "reported_external", "reports Netherlands support for Korean government efforts"),
    "023b740809": (True, "quotation_unknown", "quotation ownership is not established in current-only context"),
    "6193c2280e": (False, "target_self_report", "reports what government authorities say about education"),
}


def main() -> None:
    frame = pd.read_csv(AUDIT, encoding="utf-8-sig").fillna("")
    rows = []
    for _, row in frame.iterrows():
        value = str(row["text_sha256"])
        matches = [prefix for prefix in NEUTRAL if value.startswith(prefix)]
        if matches:
            target_correct, owner, note = NEUTRAL[matches[0]]
            label = "neutral"
        else:
            label = str(row["ambiguity_gated_prediction"])
            target_correct = True
            owner = "speaker"
            note = "explicit direct target evaluation"
        rows.append(
            {
                "text_sha256": value,
                "audit_locked_label": label,
                "audit_target_correct": target_correct,
                "audit_quotation_owner": owner,
                "audit_notes": note,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
