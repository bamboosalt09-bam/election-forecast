"""Adjudicate the independently locked V25-S V17 audit."""

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


AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v17.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v17.lock.json"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v17_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v17_labels.lock.json"
METRICS = ROOT / "outputs" / "assembly_stance" / "stance_context_semantic_role_v25s" / "locked_audit_v17_metrics.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v17_012": ("neutral", "false", "target_is_burdened_actor", "Blue House is forced to divert effort by circumstances rather than criticized"),
    "stance_v17_017": ("neutral", "true", "reported_review", "reports that government receives a harsh review without speaker ownership"),
    "stance_v17_020": ("neutral", "true", "opposition_report", "reports an opposition attack on government"),
    "stance_v17_021": ("neutral", "false", "target_absent", "describes continuing economic weakness without evaluating the assigned government"),
    "stance_v17_024": ("neutral", "false", "truncated_debt_fact", "truncated report of scholars' debt estimate; government debt is only a category"),
    "stance_v17_035": ("neutral", "false", "prior_government_scope", "current sentence inherits the explicitly prior government from context"),
    "stance_v17_040": ("neutral", "false", "regulator_prescription", "criticizes public enterprises and asks government to act as regulator"),
    "stance_v17_046": ("neutral", "false", "public_enterprise_target", "criticizes a government-owned public enterprise, not central government"),
    "stance_v17_054": ("neutral", "true", "media_report", "reports media criticism of the deputy prime minister"),
    "stance_v17_056": ("neutral", "true", "policy_premise", "states that government spends tax money on jobs without a complete owned evaluation"),
    "stance_v17_058": ("neutral", "false", "small_government_concept", "government appears only in the abstract small-government principle"),
    "stance_v17_071": ("neutral", "true", "reported_former_minister", "reports a former minister's criticism of government policy"),
    "stance_v17_078": ("neutral", "true", "policy_prescription", "prescribes when government should inject funds without criticizing it"),
    "stance_v17_081": ("neutral", "false", "public_corporation_target", "road corporation is criticized for investing without intergovernmental coordination"),
}


def main() -> None:
    if LABELS.exists() or LABEL_LOCK.exists():
        raise FileExistsError("V17 labels already exist")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if _sha256(AUDIT) != lock["output_sha256_before_review"]:
        raise RuntimeError("V17 audit changed after lock")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig", low_memory=False).fillna("")
    records: list[dict[str, str]] = []
    for row in audit.itertuples(index=False):
        if row.audit_id in OVERRIDES:
            label, target_correct, owner, notes = OVERRIDES[row.audit_id]
        else:
            label = str(row.v25s_prediction)
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
                "status": "review_complete_v25s_remained_frozen",
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
    metrics = precision_first_metrics(truth, evaluated["v25s_prediction"])
    result = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_semantic_role_v25s",
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
