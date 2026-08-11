"""Review and evaluate the independently locked V20 audit."""

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


PARTS = {
    "a": (ROOT / "data" / "shadow" / "stance_locked_audit_v12_part_a.csv", ROOT / "data" / "shadow" / "stance_locked_audit_v12_part_a.lock.json"),
    "b": (ROOT / "data" / "shadow" / "stance_locked_audit_v12_part_b.csv", ROOT / "data" / "shadow" / "stance_locked_audit_v12_part_b.lock.json"),
}
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v12_labels.csv"
LABEL_LOCK = ROOT / "data" / "shadow" / "stance_locked_audit_v12_labels.lock.json"
METRICS = ROOT / "outputs" / "assembly_stance" / "stance_context_strict_owner_v20" / "locked_audit_v12_metrics.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OVERRIDES = {
    "stance_v12a_004": (
        "neutral",
        "false",
        "historical_other_government",
        "the demonstrative refers to the preceding Lee-Park governments before context switches to Moon",
    ),
    "stance_v12b_001": (
        "neutral",
        "true",
        "conditional_warning",
        "describes failure only if inflation is not controlled",
    ),
    "stance_v12b_008": (
        "neutral",
        "true",
        "reported_hearing_opinion",
        "a committee staff report summarizes opposing confirmation opinions",
    ),
    "stance_v12b_025": (
        "neutral",
        "true",
        "target_self_report",
        "reports President Moon's own judgment about hiring corruption",
    ),
}


def main() -> None:
    if LABELS.exists() or LABEL_LOCK.exists():
        raise FileExistsError("V12 labels already exist; refusing to overwrite")
    frames: list[pd.DataFrame] = []
    verified: dict[str, str] = {}
    for name, (path, lock_path) in PARTS.items():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        digest = _sha256(path)
        if digest != lock["output_sha256_before_review"]:
            raise RuntimeError(f"V12-{name.upper()} changed after lock")
        verified[name] = digest
        frames.append(pd.read_csv(path, encoding="utf-8-sig").fillna(""))
    audit = pd.concat(frames, ignore_index=True)
    if len(audit) != 73 or audit["text_sha256"].duplicated().any():
        raise RuntimeError("combined V12 audit must contain 73 unique emissions")
    records: list[dict[str, str]] = []
    for row in audit.itertuples(index=False):
        if row.audit_id in OVERRIDES:
            label, target_correct, owner, notes = OVERRIDES[row.audit_id]
        else:
            label, target_correct, owner, notes = str(row.v20_prediction), "true", "speaker", "direct speaker-owned evaluation"
        records.append({
            "text_sha256": str(row.text_sha256),
            "audit_locked_label": label,
            "audit_target_correct": target_correct,
            "audit_quotation_owner": owner,
            "audit_notes": notes,
        })
    labels = pd.DataFrame(records)
    labels.to_csv(LABELS, index=False, encoding="utf-8-sig")
    label_state = {
        "status": "review_complete_rule_remained_frozen",
        "rows": 73,
        "verified_audit_hashes": verified,
        "labels_sha256": _sha256(LABELS),
        "label_counts": labels["audit_locked_label"].value_counts().to_dict(),
        "target_correct_counts": labels["audit_target_correct"].value_counts().to_dict(),
        "ownership_counts": labels["audit_quotation_owner"].value_counts().to_dict(),
    }
    LABEL_LOCK.write_text(json.dumps(label_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluated = audit.merge(labels, on="text_sha256", validate="one_to_one")
    truth = evaluated["audit_locked_label_y"].where(
        evaluated["audit_target_correct_y"].astype(str).str.lower().eq("true"), "neutral"
    )
    metrics = precision_first_metrics(truth, evaluated["v20_prediction"])
    result = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_strict_owner_v20",
        "active_forecast_changed": False,
        "audit_rows": 73,
        "audit_sha256_before_review": verified,
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
