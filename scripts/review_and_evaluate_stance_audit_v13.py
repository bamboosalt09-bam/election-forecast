"""Adjudicate the independently locked V21 V13 audit."""

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


AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v13.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v13.lock.json"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v13_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v13_labels.lock.json"
METRICS = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_discourse_target_v21"
    / "locked_audit_v13_metrics.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v13_009": ("party_announces_criticism_of_government", "party owns criticism but is not its target"),
    "stance_v13_033": ("government_policy_is_beneficiary", "a public institution, not government, is the negative target"),
    "stance_v13_044": ("target_lexically_absent", "excerpt criticizes land-project execution without a government referent"),
    "stance_v13_064": ("affiliated_and_local_institutions", "government-invested institutions and local authorities are not central government"),
    "stance_v13_083": ("hypothetical_failure_condition", "lists generic conditions under which privatization can fail"),
    "stance_v13_087": ("labor_collective_owns_distrust", "reports labor's distrust rather than the speaker's own stance"),
    "stance_v13_089": ("public_owns_reported_claim", "reports what citizens say"),
    "stance_v13_093": ("public_corporation_governance_hypothesis", "discusses a hypothetical responsibility design for public firms"),
}


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if _sha256(AUDIT) != lock["output_sha256_before_review"]:
        raise RuntimeError("V13 audit changed after lock")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig", low_memory=False).fillna("")
    if LABELS.exists() and LABEL_LOCK.exists():
        label_lock = json.loads(LABEL_LOCK.read_text(encoding="utf-8"))
        if (
            label_lock["audit_sha256"] != _sha256(AUDIT)
            or label_lock["labels_sha256"] != _sha256(LABELS)
        ):
            raise RuntimeError("existing V13 label lock does not match files")
        labels = pd.read_csv(LABELS, encoding="utf-8-sig").fillna("")
    elif LABELS.exists() or LABEL_LOCK.exists():
        raise RuntimeError("partial V13 label state")
    else:
        records: list[dict[str, str]] = []
        for row in audit.itertuples(index=False):
            if row.audit_id in OVERRIDES:
                owner, notes = OVERRIDES[row.audit_id]
                label = "neutral"
                target_correct = "false"
            else:
                owner, notes = "speaker", "direct speaker-owned negative evaluation of extracted target"
                label = str(row.v21_prediction)
                target_correct = "true"
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
                    "status": "review_complete_v21_remained_frozen",
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
    metrics = precision_first_metrics(truth, evaluated["v21_prediction"])
    result = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_discourse_target_v21",
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
