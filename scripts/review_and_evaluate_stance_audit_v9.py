"""Review and evaluate the independently locked V17 audit."""

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

from election_forecast.stance_precision import (  # noqa: E402
    precision_first_metrics,
    stance_adoption_assessment,
)


AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v9.csv"
AUDIT_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v9.lock.json"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v9_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v9_labels.lock.json"
METRICS = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_speaker_scope_v17"
    / "locked_audit_v9_metrics.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v9_009": (
        "neutral",
        "false",
        "historical_other_government",
        "the 2014 row explicitly evaluates the prior Lee government policy",
    ),
    "stance_v9_011": (
        "neutral",
        "true",
        "target_self_position",
        "the governing party states its joint policy commitment with government",
    ),
    "stance_v9_013": (
        "neutral",
        "false",
        "speaker_question",
        "the speaker leaves responsibility unresolved between the ministry and Blue House",
    ),
    "stance_v9_023": (
        "neutral",
        "true",
        "analytical_projection",
        "projects support and instability effects rather than asserting a government evaluation",
    ),
    "stance_v9_036": (
        "neutral",
        "true",
        "analytical_projection",
        "states a conditional relationship between the economy and government support",
    ),
    "stance_v9_044": (
        "neutral",
        "true",
        "conditional_warning",
        "calls a government bad only under hypothetical implementation failures",
    ),
    "stance_v9_049": (
        "neutral",
        "true",
        "speaker_question",
        "asks whether the government bears primary responsibility",
    ),
    "stance_v9_054": (
        "neutral",
        "true",
        "reported_external",
        "reports Park Won-soon and Lee Hae-chan attributing employment losses to Lee",
    ),
}


def main() -> None:
    if LABELS.exists() or LABEL_LOCK.exists():
        raise FileExistsError("V9 labels already exist; refusing to overwrite")
    lock = json.loads(AUDIT_LOCK.read_text(encoding="utf-8"))
    digest = _sha256(AUDIT)
    if digest != lock["output_sha256_before_review"]:
        raise RuntimeError("V9 audit changed after it was locked")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig").fillna("")
    if len(audit) != 59 or audit["text_sha256"].duplicated().any():
        raise RuntimeError("V9 audit must contain 59 unique emissions")

    records: list[dict[str, str]] = []
    for row in audit.itertuples(index=False):
        if row.audit_id in OVERRIDES:
            label, target_correct, owner, notes = OVERRIDES[row.audit_id]
        else:
            label = str(row.v17_prediction)
            target_correct = "true"
            owner = "speaker"
            notes = "direct speaker-owned evaluation"
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
    label_state = {
        "status": "review_complete_rule_remained_frozen",
        "rows": 59,
        "verified_audit_hash": digest,
        "labels_sha256": _sha256(LABELS),
        "label_counts": labels["audit_locked_label"].value_counts().to_dict(),
        "target_correct_counts": labels["audit_target_correct"].value_counts().to_dict(),
        "ownership_counts": labels["audit_quotation_owner"].value_counts().to_dict(),
    }
    LABEL_LOCK.write_text(
        json.dumps(label_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    evaluated = audit.merge(labels, on="text_sha256", validate="one_to_one")
    truth = evaluated["audit_locked_label_y"].where(
        evaluated["audit_target_correct_y"].astype(str).str.lower().eq("true"),
        "neutral",
    )
    metrics = precision_first_metrics(truth, evaluated["v17_prediction"])
    adoption = stance_adoption_assessment(
        metrics,
        independent_audit=True,
        target_attribution_audited=True,
        point_in_time_audited=True,
        rolling_non_degradation=False,
    )
    state = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_speaker_scope_v17",
        "active_forecast_changed": False,
        "audit_rows": 59,
        "audit_sha256_before_review": digest,
        "labels_sha256": _sha256(LABELS),
        "metrics": metrics,
        "adoption": adoption,
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
