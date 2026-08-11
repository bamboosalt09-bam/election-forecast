"""Build target-aware training rows from contrastive and rejected v1 audit data."""

from __future__ import annotations

import hashlib
import sys
from itertools import cycle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_stance_contrastive_expansion import ELECTIONS, ISSUES, TARGETS  # noqa: E402


BASE = ROOT / "data" / "shadow" / "stance_contrastive_expansion_v2.csv"
AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v1.csv"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v1_labels.csv"
OUTPUT = ROOT / "data" / "shadow" / "stance_target_aware_expansion_v3.csv"
INVALID_TEMPLATES = (
    "우리 {target}은 {issue} 정책을 책임 있게 추진하겠습니다.",
    "{target} 소속 김 의원이 {issue} 법안을 발의했습니다.",
    "언론은 {target}이 {issue} 정책을 지지했다고 보도했습니다.",
    "'{target}을 지지한다'는 표현이 {issue} 자료에 적혀 있습니다.",
    "{target}께서는 {issue} 방침을 밝혀 주시기 바랍니다.",
    "{target}이 {issue} 법안의 공동발의자 명단에 포함됐습니다.",
    "다른 후보를 비판한 {target}의 {issue} 발언이 보도됐습니다.",
    "{target}과 관련한 {issue} 자료가 위원회에 제출됐습니다.",
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _invalid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    elections = cycle(ELECTIONS)
    index = 0
    for target_type, target_name, target_alias in TARGETS:
        for issue in ISSUES:
            for template in INVALID_TEMPLATES:
                text = template.format(target=target_name, issue=issue)
                rows.append(
                    {
                        "audit_id": f"SYN-INVALID-{index:05d}",
                        "election_id": next(elections),
                        "meeting_date": "1998-01-01",
                        "committee": "synthetic_target_invalid",
                        "speaker": "synthetic_speaker",
                        "issue_name": issue,
                        "target_type": target_type,
                        "target_name": target_name,
                        "target_alias": target_alias,
                        "text_excerpt": text,
                        "context_before": "",
                        "context_after": "",
                        "source_file": "synthetic_target_invalid_v3",
                        "source_row_id": str(index),
                        "text_sha256": _hash(text),
                        "review_label": "neutral",
                        "review_target_correct": "false",
                        "review_basis": "non-activatable target or reported ownership contrast",
                        "training_origin": "synthetic_target_invalid",
                        "training_weight": 0.05,
                    }
                )
                index += 1
    return rows


def _audit_rows() -> pd.DataFrame:
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig").fillna("").drop(
        columns=[
            "audit_locked_label",
            "audit_target_correct",
            "audit_quotation_owner",
            "audit_notes",
        ]
    )
    labels = pd.read_csv(LABELS, encoding="utf-8-sig").fillna("")
    frame = audit.merge(labels, on="text_sha256", validate="one_to_one")
    target_correct = frame["audit_target_correct"].astype(str).str.lower().eq("true")
    frame["review_label"] = frame["audit_locked_label"].where(
        target_correct, "neutral"
    )
    frame["review_target_correct"] = target_correct
    frame["review_basis"] = frame["audit_notes"]
    frame["training_origin"] = "rejected_v7_locked_audit"
    frame["training_weight"] = 1.0
    return frame


def main() -> None:
    base = pd.read_csv(BASE, encoding="utf-8-sig").fillna("")
    invalid = pd.DataFrame(_invalid_rows())
    audit = _audit_rows()
    columns = list(dict.fromkeys([*base.columns, *invalid.columns, *audit.columns]))
    combined = pd.concat(
        [
            base.reindex(columns=columns, fill_value=""),
            invalid.reindex(columns=columns, fill_value=""),
            audit.reindex(columns=columns, fill_value=""),
        ],
        ignore_index=True,
    )
    if combined["text_sha256"].duplicated().any():
        raise RuntimeError("target-aware expansion contains duplicate text hashes")
    combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(
        {
            "rows": len(combined),
            "origins": combined["training_origin"].value_counts().to_dict(),
            "labels": combined["review_label"].value_counts().to_dict(),
            "weighted_mass": combined.groupby("training_origin")["training_weight"]
            .sum()
            .to_dict(),
            "output": str(OUTPUT),
        }
    )


if __name__ == "__main__":
    main()
