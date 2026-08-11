"""Apply frozen V19 owner-resolution gates to a fresh NLI shadow corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from election_forecast.stance_context_v19 import (  # noqa: E402
    apply_contextual_owner_resolution_gate_v19,
)
from scripts.apply_stance_precision_ensemble import validate_shadow_corpus  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "context_predictions_v19.csv"
    if output_path.exists():
        raise FileExistsError(f"V19 prediction output already exists: {output_path}")

    frame = pd.read_csv(source, encoding="utf-8-sig").fillna("")
    validate_shadow_corpus(frame)
    prediction, reasons = apply_contextual_owner_resolution_gate_v19(
        frame.to_dict(orient="records"), frame["context_prediction"]
    )
    output = frame.copy()
    output["v19_prediction"] = prediction
    output["v19_abstention_reasons"] = reasons
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    state = {
        "status": "frozen_v19_shadow_application_complete",
        "model_version": "stance_context_owner_resolution_v19",
        "active_forecast_changed": False,
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "rows": int(len(output)),
        "base_directional": int(output["context_prediction"].ne("neutral").sum()),
        "v19_directional": int(output["v19_prediction"].ne("neutral").sum()),
        "prediction_counts": output["v19_prediction"].value_counts().to_dict(),
        "input_sha256": _sha256(source),
        "output_sha256": _sha256(output_path),
        "output": str(output_path),
    }
    (output_dir / "application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
