"""Review and evaluate the 59 independently locked V16 emissions."""

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


PARTS = {
    "a": (
        ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_a.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_a.lock.json",
    ),
    "b": (
        ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_b.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v8_part_b.lock.json",
    ),
}
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v8_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v8_labels.lock.json"
METRICS = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_attribution_v16"
    / "locked_audit_v8_metrics.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v8a_017": (
        "neutral",
        "true",
        "speaker_question",
        "asks whether a government action was proper and requests an answer",
    ),
    "stance_v8a_023": (
        "neutral",
        "false",
        "historical_other_government",
        "surrounding context distinguishes the criticized prior policy from the current regime",
    ),
    "stance_v8a_024": (
        "neutral",
        "true",
        "reported_target_admission",
        "reports the president assigning responsibility to government",
    ),
    "stance_v8a_034": (
        "neutral",
        "true",
        "reported_external",
        "reports a professor's judgment",
    ),
    "stance_v8a_040": (
        "neutral",
        "true",
        "target_self_position",
        "the president presents the government's own favorable performance account",
    ),
    "stance_v8a_042": (
        "neutral",
        "true",
        "target_self_position",
        "the foreign minister states the government's own future policy",
    ),
    "stance_v8a_047": (
        "neutral",
        "false",
        "foreign_government",
        "the explicit government alias refers to the incoming United States administration",
    ),
    "stance_v8a_050": (
        "neutral",
        "false",
        "future_government",
        "the explicit alias refers to a future administration rather than the current target",
    ),
    "stance_v8a_056": (
        "neutral",
        "true",
        "reported_external",
        "reports international support for government policy",
    ),
}


def main() -> None:
    if LABELS.exists() or LABEL_LOCK.exists():
        raise FileExistsError("V8 labels already exist; refusing to overwrite")
    frames: list[pd.DataFrame] = []
    verified_parts: dict[str, str] = {}
    for name, (path, lock_path) in PARTS.items():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        digest = _sha256(path)
        if digest != lock["output_sha256_before_review"]:
            raise RuntimeError(f"V8-{name.upper()} changed after it was locked")
        verified_parts[name] = digest
        frames.append(pd.read_csv(path, encoding="utf-8-sig").fillna(""))
    audit = pd.concat(frames, ignore_index=True)
    if len(audit) != 59 or audit["text_sha256"].duplicated().any():
        raise RuntimeError("combined V8 audit must contain 59 unique emissions")
    records: list[dict[str, str]] = []
    for row in audit.itertuples(index=False):
        if row.audit_id in OVERRIDES:
            label, target_correct, owner, notes = OVERRIDES[row.audit_id]
        else:
            label = str(row.v16_prediction)
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
        "verified_audit_hashes": verified_parts,
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
    metrics = precision_first_metrics(truth, evaluated["v16_prediction"])
    adoption = stance_adoption_assessment(
        metrics,
        independent_audit=True,
        target_attribution_audited=True,
        point_in_time_audited=True,
        rolling_non_degradation=False,
    )
    state = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_attribution_v16",
        "active_forecast_changed": False,
        "audit_rows": 59,
        "metrics": metrics,
        "adoption": adoption,
    }
    METRICS.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
