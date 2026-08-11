"""Record and apply a transparent full-text correction to V13 adjudication."""

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
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v13_labels.csv"
CORRECTION = ROOT / "data" / "shadow" / "stance_locked_audit_v13_adjudication_correction.csv"
CORRECTION_LOCK = CORRECTION.with_suffix(".lock.json")
METRICS = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_discourse_target_v21"
    / "locked_audit_v13_metrics_corrected.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if CORRECTION.exists() or CORRECTION_LOCK.exists():
        raise FileExistsError("V13 correction already exists")
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig", low_memory=False).fillna("")
    labels = pd.read_csv(LABELS, encoding="utf-8-sig").fillna("")
    row = audit.loc[audit["audit_id"].eq("stance_v13_044")]
    if len(row) != 1 or "정부정책 불신을 초래" not in str(row.iloc[0]["text_excerpt"]):
        raise RuntimeError("full-text evidence for V13-044 correction is absent")
    correction = pd.DataFrame(
        [
            {
                "audit_id": "stance_v13_044",
                "text_sha256": str(row.iloc[0]["text_sha256"]),
                "old_label": "neutral",
                "new_label": "negative",
                "old_target_correct": "false",
                "new_target_correct": "true",
                "reason": "initial 220-character display truncated the concluding direct claim that the problems cause distrust in government policy",
            }
        ]
    )
    correction.to_csv(CORRECTION, index=False, encoding="utf-8-sig")
    CORRECTION_LOCK.write_text(
        json.dumps(
            {
                "status": "adjudication_correction_locked",
                "audit_sha256": _sha256(AUDIT),
                "original_labels_sha256": _sha256(LABELS),
                "correction_sha256": _sha256(CORRECTION),
                "rows": 1,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    corrected = labels.merge(correction[["text_sha256", "new_label", "new_target_correct"]], on="text_sha256", how="left")
    corrected["audit_locked_label"] = corrected["new_label"].where(
        corrected["new_label"].notna(), corrected["audit_locked_label"]
    )
    corrected["audit_target_correct"] = corrected["new_target_correct"].where(
        corrected["new_target_correct"].notna(), corrected["audit_target_correct"]
    )
    evaluated = audit.merge(corrected, on="text_sha256", validate="one_to_one")
    truth = evaluated["audit_locked_label"].where(
        evaluated["audit_target_correct"].astype(str).str.lower().eq("true"), "neutral"
    )
    metrics = precision_first_metrics(truth, evaluated["v21_prediction"])
    result = {
        "status": "independent_locked_audit_complete_with_adjudication_correction",
        "model_version": "stance_context_discourse_target_v21",
        "active_forecast_changed": False,
        "audit_rows": len(audit),
        "original_labels_sha256": _sha256(LABELS),
        "correction_sha256": _sha256(CORRECTION),
        "metrics": metrics,
        "adoption": stance_adoption_assessment(
            metrics,
            independent_audit=True,
            target_attribution_audited=True,
            point_in_time_audited=True,
            rolling_non_degradation=False,
        ),
    }
    METRICS.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
