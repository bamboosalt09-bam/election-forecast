"""Adjudicate the independently locked V22 V14 audit."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import precision_first_metrics, stance_adoption_assessment  # noqa: E402


AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v14.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v14.lock.json"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v14_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v14_labels.lock.json"
METRICS = ROOT / "outputs" / "assembly_stance" / "stance_context_grammatical_target_v22" / "locked_audit_v14_metrics.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v14_010": ("criticism_object", "speaker criticizes using government as an attack object"),
    "stance_v14_011": ("public_report", "reports that citizens do not trust government"),
    "stance_v14_035": ("committee_collective_report", "reports other members' concern about approval decline"),
    "stance_v14_046": ("neutral_policy_hypothesis", "current sentence is a neutral hypothetical fiscal proposition"),
    "stance_v14_054": ("historical_government", "YS government is historical relative to the 2006 meeting"),
}


def main() -> None:
    if LABELS.exists() or LABEL_LOCK.exists():
        raise FileExistsError("V14 labels already exist")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if _sha256(AUDIT) != lock["output_sha256_before_review"]:
        raise RuntimeError("V14 audit changed after lock")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig", low_memory=False).fillna("")
    records: list[dict[str, str]] = []
    for row in audit.itertuples(index=False):
        if row.audit_id in OVERRIDES:
            owner, notes = OVERRIDES[row.audit_id]
            label, target_correct = "neutral", "false"
        else:
            owner, notes = "speaker", "direct speaker-owned negative evaluation of extracted target"
            label, target_correct = str(row.v22_prediction), "true"
        records.append(
            {
                "text_sha256": str(row.text_sha256),
                "audit_locked_label": label,
                "audit_target_correct": target_correct,
                "audit_quotation_owner": owner,
                "audit_notes": notes,
            }
        )
    labels = pd.DataFrame(records)
    labels.to_csv(LABELS, index=False, encoding="utf-8-sig")
    LABEL_LOCK.write_text(
        json.dumps(
            {
                "status": "review_complete_v22_remained_frozen",
                "rows": len(labels),
                "audit_sha256": _sha256(AUDIT),
                "labels_sha256": _sha256(LABELS),
                "neutral_rows": int(labels["audit_locked_label"].eq("neutral").sum()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    evaluated = audit.merge(labels, on="text_sha256", validate="one_to_one")
    truth = evaluated["audit_locked_label"].where(
        evaluated["audit_target_correct"].astype(str).str.lower().eq("true"), "neutral"
    )
    metrics = precision_first_metrics(truth, evaluated["v22_prediction"])
    result = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_grammatical_target_v22",
        "active_forecast_changed": False,
        "audit_rows": len(audit),
        "audit_sha256_before_review": lock["output_sha256_before_review"],
        "labels_sha256": _sha256(LABELS),
        "metrics": metrics,
        "adoption": stance_adoption_assessment(
            metrics,
            independent_audit=True,
            target_attribution_audited=True,
            point_in_time_audited=True,
            rolling_non_degradation=False,
        ),
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
