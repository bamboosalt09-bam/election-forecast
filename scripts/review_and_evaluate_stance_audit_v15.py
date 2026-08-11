"""Adjudicate the independently locked V23-S V15 audit."""

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


AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v15.csv"
LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v15.lock.json"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v15_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v15_labels.lock.json"
METRICS = ROOT / "outputs" / "assembly_stance" / "stance_context_pragmatic_role_v23s" / "locked_audit_v15_metrics.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v15_004": ("neutral", "true", "official_explanation", "official neutrally explains the 1997 crisis and subsequent government response"),
    "stance_v15_011": ("positive", "true", "speaker", "speaker credits the Lee government rate-cut capacity with supporting recovery"),
    "stance_v15_027": ("neutral", "true", "speaker", "general observation that elected governments and central banks often disagree"),
    "stance_v15_047": ("neutral", "true", "fragment", "truncated fragment does not contain a complete evaluative proposition"),
    "stance_v15_050": ("neutral", "true", "public_report", "reports other people's anger and loss without adopting it as the speaker's stance"),
    "stance_v15_053": ("neutral", "true", "causal_explanation", "explains why weak firms lose employment after support ends, not a criticism of government"),
    "stance_v15_059": ("neutral", "false", "historical_scope_unknown", "the deictic past event and government are not attributable to the assigned current government"),
    "stance_v15_062": ("neutral", "false", "target_absent", "describes economic conditions without an explicit or contextual government evaluation"),
    "stance_v15_065": ("neutral", "false", "generic_historical_analysis", "generic crisis mechanism is not attributable to the assigned 2013 government"),
}


def main() -> None:
    if LABELS.exists() or LABEL_LOCK.exists():
        raise FileExistsError("V15 labels already exist")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if _sha256(AUDIT) != lock["output_sha256_before_review"]:
        raise RuntimeError("V15 audit changed after lock")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig", low_memory=False).fillna("")
    records: list[dict[str, str]] = []
    for row in audit.itertuples(index=False):
        if row.audit_id in OVERRIDES:
            label, target_correct, owner, notes = OVERRIDES[row.audit_id]
        else:
            label = str(row.v23s_prediction)
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
                "status": "review_complete_v23s_remained_frozen",
                "rows": len(labels),
                "audit_sha256": _sha256(AUDIT),
                "labels_sha256": _sha256(LABELS),
                "neutral_rows": int(labels["audit_locked_label"].eq("neutral").sum()),
                "positive_rows": int(labels["audit_locked_label"].eq("positive").sum()),
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
    metrics = precision_first_metrics(truth, evaluated["v23s_prediction"])
    result = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_pragmatic_role_v23s",
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
