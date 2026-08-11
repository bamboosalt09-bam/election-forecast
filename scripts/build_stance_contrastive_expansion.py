"""Build deterministic low-weight Korean stance contrast pairs.

Synthetic rows teach attribution and speech-act contrasts. They are never
treated as audit evidence and receive a much smaller training weight than the
manually reviewed parliamentary rows.
"""

from __future__ import annotations

import hashlib
from itertools import cycle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "data" / "shadow" / "stance_manual_gold_expansion_v1.csv"
OUTPUT = ROOT / "data" / "shadow" / "stance_contrastive_expansion_v2.csv"
ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
TARGETS = (
    ("person", "김민준", "김 후보"),
    ("person", "이서준", "이 후보"),
    ("person", "박지훈", "박 후보"),
    ("person", "최유진", "최 후보"),
    ("person", "정도현", "정 후보"),
    ("person", "한수빈", "한 후보"),
    ("party", "민주개혁당", "개혁당"),
    ("party", "국민보수당", "보수당"),
    ("party", "미래정의당", "정의당"),
    ("party", "새로운중도당", "중도당"),
    ("government", "정부", "행정부"),
    ("government", "청와대", "대통령실"),
)
ISSUES = (
    "경제",
    "주거",
    "복지",
    "외교",
    "안보",
    "교육",
    "노동",
    "연금",
    "의료",
    "부정부패 방지",
    "지역균형",
    "정치개혁",
)
POSITIVE = (
    "저는 {target}의 {issue} 정책을 적극 지지합니다.",
    "{target}이 추진한 {issue} 개혁에 찬성합니다.",
    "{target}의 {issue} 분야 성과를 높이 평가합니다.",
    "우리 당은 {target}의 {issue} 제안을 환영하며 계속 협력하겠습니다.",
)
NEGATIVE = (
    "{target}의 {issue} 정책은 명백한 실패이며 책임을 져야 합니다.",
    "저는 {target}의 {issue} 대응이 잘못되었다고 강하게 비판합니다.",
    "{target}은 {issue} 문제를 해결할 능력이 없었고 국민에게 사과해야 합니다.",
    "우리 당은 {target}의 {issue} 방침에 반대하며 즉시 철회할 것을 요구합니다.",
)
NEUTRAL = (
    "야당은 {target}의 {issue} 정책이 실패했다고 주장했지만 사실 여부는 확인되지 않았습니다.",
    "{target}의 {issue} 정책을 지지하느냐는 질문이 제기되었습니다.",
    "자료에는 {target}의 {issue} 제안을 지지한다는 표현이 인용되어 있습니다.",
    "언론은 한 의원이 {target}의 {issue} 대응을 비판했다고 보도했습니다.",
    "{target}의 {issue} 방침에 대한 찬성 또는 반대 여부를 밝혀 주십시오.",
    "일부 의원은 {target}의 {issue} 정책을 지지했고 다른 의원은 강하게 비판했습니다.",
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    elections = cycle(ELECTIONS)
    index = 0
    for target_type, target_name, target_alias in TARGETS:
        for issue in ISSUES:
            for label, templates in (
                ("positive", POSITIVE),
                ("negative", NEGATIVE),
                ("neutral", NEUTRAL),
            ):
                for template in templates:
                    text = template.format(target=target_name, issue=issue)
                    text_hash = _hash(text)
                    rows.append(
                        {
                            "audit_id": f"SYN-{index:05d}",
                            "election_id": next(elections),
                            "meeting_date": "1998-01-01",
                            "committee": "synthetic_contrastive",
                            "speaker": "synthetic_speaker",
                            "issue_name": issue,
                            "target_type": target_type,
                            "target_name": target_name,
                            "target_alias": target_alias,
                            "text_excerpt": text,
                            "context_before": "",
                            "context_after": "",
                            "source_file": "synthetic_contrastive_v2",
                            "source_row_id": str(index),
                            "text_sha256": text_hash,
                            "review_label": label,
                            "review_target_correct": "true",
                            "review_basis": "deterministic contrast template",
                            "training_origin": "synthetic_contrastive",
                            "training_weight": 0.05,
                        }
                    )
                    index += 1
    return rows


def main() -> None:
    manual = pd.read_csv(MANUAL, encoding="utf-8-sig").fillna("")
    manual["training_origin"] = "manual_parliamentary"
    manual["training_weight"] = 1.0
    synthetic = pd.DataFrame(_synthetic_rows())
    columns = list(dict.fromkeys([*manual.columns, *synthetic.columns]))
    combined = pd.concat(
        [manual.reindex(columns=columns, fill_value=""), synthetic.reindex(columns=columns, fill_value="")],
        ignore_index=True,
    )
    if combined["text_sha256"].duplicated().any():
        raise RuntimeError("contrastive expansion contains duplicate text hashes")
    combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(
        {
            "output": str(OUTPUT),
            "rows": len(combined),
            "manual_rows": len(manual),
            "synthetic_rows": len(synthetic),
            "labels": combined["review_label"].value_counts().to_dict(),
            "weighted_synthetic_mass": float(synthetic["training_weight"].sum()),
        }
    )


if __name__ == "__main__":
    main()
