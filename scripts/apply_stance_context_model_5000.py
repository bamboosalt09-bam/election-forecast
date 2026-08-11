"""Apply the trained context model to its frozen 5,000-sentence corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from election_forecast.stance_context_model import compose_context_input  # noqa: E402
from election_forecast.stance_text_model import label_to_polarity  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"


def main() -> None:
    frame = pd.read_csv(DATA_DIR / "stance_context_5000.csv", encoding="utf-8-sig").fillna("")
    artifact = joblib.load(DATA_DIR / "stance_context_5000_v1.joblib")
    model_inputs = pd.Series(
        [compose_context_input(row) for row in frame.to_dict(orient="records")]
    )
    probabilities = artifact["model"].predict_proba(model_inputs.astype(str))
    classes = np.asarray(artifact["classes"])
    best_indices = np.argmax(probabilities, axis=1)
    predictions = classes[best_indices]
    sorted_probabilities = np.sort(probabilities, axis=1)

    output = frame.copy()
    output["context_model_label"] = predictions
    output["context_model_polarity"] = [label_to_polarity(label) for label in predictions]
    output["context_model_probability"] = probabilities[np.arange(len(output)), best_indices]
    output["context_model_probability_margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    for index, label in enumerate(classes):
        output[f"context_model_probability_{label}"] = probabilities[:, index]
    output["context_model_changed_from_rule"] = (
        output["context_model_polarity"].astype(int)
        != pd.to_numeric(output["rule_stance_polarity"], errors="coerce").fillna(0).astype(int)
    ).astype(int)
    output.to_csv(DATA_DIR / "context_predictions_5000.csv", index=False, encoding="utf-8-sig")

    cross = pd.crosstab(
        pd.to_numeric(output["rule_stance_polarity"], errors="coerce").fillna(0).astype(int),
        output["context_model_label"],
    )
    state = {
        "status": "complete",
        "rows": int(len(output)),
        "prediction_counts": output["context_model_label"].value_counts().to_dict(),
        "changed_from_rule": int(output["context_model_changed_from_rule"].sum()),
        "mean_max_probability": float(output["context_model_probability"].mean()),
        "mean_probability_margin": float(output["context_model_probability_margin"].mean()),
        "rule_polarity_by_model_label": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in cross.iterrows()
        },
        "model_version": artifact["model_version"],
        "weak_mass_ratio": artifact["weak_mass_ratio"],
    }
    (DATA_DIR / "application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
