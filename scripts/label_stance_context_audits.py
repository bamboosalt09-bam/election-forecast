"""Write the manual labels for the locked v3/v4 context-model audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "7443140bce": ("negative", True, "speaker", "current-regime corruption criticism across context"),
    "ca262c8766": ("negative", True, "speaker", "direct government policy-failure criticism"),
    "5a9054ae6a": ("negative", True, "speaker", "direct comparison criticizing the Roh government"),
    "79918357d7": ("negative", True, "speaker", "explicit first-person foreign-policy criticism"),
    "3e898bd88f": ("neutral", True, "reported_external", "reports opposition and media criticism"),
    "05cd292619": ("neutral", True, "reported_external", "reports Czech support for Korean policy"),
    "208a64bb1c": ("negative", True, "speaker", "direct party-regime criticism"),
    "b37ad66e83": ("negative", True, "speaker", "direct government economic-policy criticism"),
    "6c90cd7f5e": ("negative", True, "speaker", "direct criticism of the Moon government"),
    "79b523c28f": ("neutral", False, "speaker", "education-sector corruption is not a government stance target"),
    "5f77639a31": ("negative", True, "speaker", "criticizes economic agencies despite a narrow disclaimer"),
    "37a172f034": ("negative", True, "speaker", "explicit government forecast failure"),
    "bda97f4d3f": ("negative", True, "speaker", "explicit government job-policy failure"),
    "491b735048": ("negative", True, "speaker", "explicit government policy failure"),
    "b3f23842d8": ("negative", True, "speaker", "context gives speaker's negative causal assessment"),
    "6b5406edb5": ("neutral", True, "reported_external", "describes an opposition framing rather than adopting it"),
    "3cc5657924": ("negative", True, "speaker", "explicit government trust criticism"),
    "47527e8091": ("positive", True, "speaker", "speaker endorses the presidential summit"),
    "bd2387f310": ("negative", True, "speaker", "direct incompetence criticism"),
    "456e3453e8": ("neutral", True, "reported_external", "reports a survey response"),
    "c37c7744ea": ("neutral", True, "reported_external", "reports experts' assessment"),
    "52043d9905": ("negative", True, "speaker", "direct government policy criticism"),
    "e45f75d7c7": ("neutral", False, "speaker", "states the government's stance toward a UN resolution"),
    "d42648bb0f": ("negative", True, "speaker", "explicit first-person government criticism"),
    "7529357c02": ("negative", True, "speaker", "direct government policy criticism"),
    "b58636b802": ("negative", True, "speaker", "direct Lee-government policy criticism"),
    "0ba3e128f6": ("negative", True, "speaker", "direct housing-policy criticism"),
    "e04bca050b": ("negative", True, "speaker", "direct government housing-policy criticism"),
    "2a521be8be": ("negative", True, "speaker", "speaker draws a negative causal conclusion"),
    "75982f1a6e": ("negative", True, "speaker", "direct criticism of education-policy omissions"),
    "1ab38d5629": ("negative", True, "speaker", "warns that unrealistic government policy fails"),
    "1d5cb61db8": ("neutral", True, "speaker_question", "asks whether prior governments are responsible"),
    "52384916f4": ("neutral", False, "speaker", "lists harms of a practice, not a stance toward government"),
    "96e97a8913": ("neutral", False, "speaker", "broad institutional corruption without the assigned target"),
    "0b7fc188a1": ("negative", True, "speaker", "explicit government policy-failure criticism"),
    "29dffd8b11": ("neutral", True, "reported_rebutted", "quotes and then rejects a government-failure claim"),
    "762d908403": ("negative", True, "speaker", "direct Park-government criticism"),
    "7d0a39bff6": ("neutral", False, "speaker", "assigned Park target is contrasted with criticism of Moon"),
    "018b304252": ("negative", True, "speaker", "context criticizes current-regime corruption"),
    "a4f4efe8bf": ("neutral", True, "speaker_question", "question does not assert the failure"),
}


def label_audit(audit_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(audit_path, encoding="utf-8-sig").fillna("")
    rows: list[dict[str, object]] = []
    for value in frame["text_sha256"].astype(str):
        matches = [prefix for prefix in LABELS if value.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"expected one manual label for {value}, got {matches}")
        label, target_correct, owner, notes = LABELS[matches[0]]
        rows.append(
            {
                "text_sha256": value,
                "audit_locked_label": label,
                "audit_target_correct": target_correct,
                "audit_quotation_owner": owner,
                "audit_notes": notes,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    label_audit(args.audit.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
