"""Evaluate frozen V15 predictions on the independently locked V7 audit."""

from __future__ import annotations

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


AUDIT = ROOT / "data" / "shadow" / "stance_locked_audit_v7.csv"
LABELS = ROOT / "data" / "shadow" / "stance_locked_audit_v7_labels.csv"
OUTPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_ownership_v15"
    / "locked_audit_v7_metrics.json"
)


def main() -> None:
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig").fillna("")
    labels = pd.read_csv(LABELS, encoding="utf-8-sig").fillna("")
    evaluated = audit.merge(labels, on="text_sha256", validate="one_to_one")
    truth = evaluated["audit_locked_label_y"].where(
        evaluated["audit_target_correct_y"].astype(str).str.lower().eq("true"),
        "neutral",
    )
    metrics = precision_first_metrics(truth, evaluated["v15_prediction"])
    adoption = stance_adoption_assessment(
        metrics,
        independent_audit=True,
        target_attribution_audited=True,
        point_in_time_audited=True,
        rolling_non_degradation=False,
    )
    state = {
        "status": "independent_locked_audit_complete",
        "model_version": "stance_context_ownership_v15",
        "active_forecast_changed": False,
        "audit_rows": int(len(evaluated)),
        "metrics": metrics,
        "adoption": adoption,
    }
    OUTPUT.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
