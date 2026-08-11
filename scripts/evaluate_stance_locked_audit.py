"""Evaluate a frozen stance prediction against a separate audit-label file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import precision_first_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = pd.read_csv(args.audit.resolve(), encoding="utf-8-sig").fillna("")
    audit = audit.drop(
        columns=[
            "audit_locked_label",
            "audit_target_correct",
            "audit_quotation_owner",
            "audit_notes",
        ],
        errors="ignore",
    )
    labels = pd.read_csv(args.labels.resolve(), encoding="utf-8-sig").fillna("")
    frame = audit.merge(labels, on="text_sha256", validate="one_to_one")
    target_correct = (
        frame["audit_target_correct"].astype(str).str.lower().eq("true")
    )
    truth = frame["audit_locked_label"].where(target_correct, "neutral")
    metrics = precision_first_metrics(truth, frame[args.prediction_column])
    payload = {
        "status": "locked_audit_evaluated",
        "active_forecast_changed": False,
        "audit_rows": len(frame),
        "target_incorrect_rows": int((~target_correct).sum()),
        "reported_or_question_rows": int(
            frame["audit_quotation_owner"].astype(str).ne("speaker").sum()
        ),
        "prediction_column": args.prediction_column,
        "metrics": metrics,
    }
    if args.output:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
