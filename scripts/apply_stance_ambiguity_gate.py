"""Apply hard ambiguity abstention to a shadow stance prediction file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_precision import (  # noqa: E402
    apply_ambiguity_abstention,
    precision_first_metrics,
)
from scripts.apply_stance_precision_ensemble import validate_shadow_corpus  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_ko_nli_context_v10"
    / "application_5000"
    / "context_predictions_5000.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_nli_ambiguity_v14"
    / "application_5000"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prediction-column", default="context_prediction")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input.resolve(), encoding="utf-8-sig").fillna("")
    validate_shadow_corpus(frame)
    prediction, reasons = apply_ambiguity_abstention(
        frame.to_dict(orient="records"), frame[args.prediction_column]
    )
    output = frame.copy()
    output["ambiguity_abstention_reasons"] = reasons
    output["ambiguity_gated_prediction"] = prediction
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ambiguity_gated_predictions_5000.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    directional = output.loc[output["ambiguity_gated_prediction"].ne("neutral")]
    state: dict[str, object] = {
        "status": "shadow_ambiguity_gate_complete",
        "model_version": "stance_nli_ambiguity_v14",
        "active_forecast_changed": False,
        "selection_note": "v4-v6 audits informed these rules; v14 has no untouched confirmatory audit",
        "rows": len(output),
        "directional_before": int(frame[args.prediction_column].ne("neutral").sum()),
        "directional_after": len(directional),
        "abstention_reason_counts": (
            output.loc[output["ambiguity_abstention_reasons"].ne(""), "ambiguity_abstention_reasons"]
            .str.get_dummies(sep="|")
            .sum()
            .sort_values(ascending=False)
            .to_dict()
        ),
        "output": str(output_path),
    }
    if args.audit and args.labels:
        audit = pd.read_csv(args.audit.resolve(), encoding="utf-8-sig").fillna("")
        labels = pd.read_csv(args.labels.resolve(), encoding="utf-8-sig").fillna("")
        evaluated = audit[["text_sha256"]].merge(
            output[["text_sha256", "ambiguity_gated_prediction"]],
            on="text_sha256",
            validate="one_to_one",
        ).merge(labels, on="text_sha256", validate="one_to_one")
        target_correct = (
            evaluated["audit_target_correct"].astype(str).str.lower().eq("true")
        )
        truth = evaluated["audit_locked_label"].where(target_correct, "neutral")
        state["development_v4_diagnostic"] = precision_first_metrics(
            truth, evaluated["ambiguity_gated_prediction"]
        )
    (output_dir / "application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
