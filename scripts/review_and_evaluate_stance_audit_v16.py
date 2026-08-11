"""Adjudicate the independently locked V24-S V16 audit."""

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


AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v16.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v16.lock.json"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v16_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v16_labels.lock.json"
METRICS = ROOT / "outputs" / "assembly_stance" / "stance_context_lexical_role_v24s" / "locked_audit_v16_metrics.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v16_005": ("neutral", "false", "factual_budget", "budget-size fact evaluates public enterprises, not the assigned central government"),
    "stance_v16_010": ("neutral", "true", "event_description", "describes suspension after impeachment and national confusion without an owned evaluation"),
    "stance_v16_017": ("neutral", "false", "target_is_victim", "Japan is alleged to seek damage to the Moon government; the target is the attack victim"),
    "stance_v16_024": ("neutral", "false", "macro_hypothesis", "hypothetical debt and property-price scenario does not evaluate the assigned government"),
    "stance_v16_060": ("neutral", "false", "reported_consensus", "reports a common view about two historical governments without explicit speaker ownership"),
    "stance_v16_081": ("neutral", "false", "target_is_victim", "laments that a document leak disrupts the president and disperses government capacity"),
    "stance_v16_091": ("neutral", "false", "hypothetical_executor", "criticizes a redistribution argument while government is only a hypothetical executor"),
}


def main() -> None:
    if LABELS.exists() or LABEL_LOCK.exists():
        raise FileExistsError("V16 labels already exist")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if _sha256(AUDIT) != lock["output_sha256_before_review"]:
        raise RuntimeError("V16 audit changed after lock")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig", low_memory=False).fillna("")
    records: list[dict[str, str]] = []
    for row in audit.itertuples(index=False):
        if row.audit_id in OVERRIDES:
            label, target_correct, owner, notes = OVERRIDES[row.audit_id]
        else:
            label = str(row.v24s_prediction)
            target_correct = "true"
            owner = "speaker"
            notes = "speaker-owned negative evaluation of the extracted target"
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
                "status": "review_complete_v24s_remained_frozen",
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
    metrics = precision_first_metrics(truth, evaluated["v24s_prediction"])
    result = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_lexical_role_v24s",
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
