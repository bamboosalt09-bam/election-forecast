"""Run a fresh 5,000-row context inference with intensity and information labels."""

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
from election_forecast.stance_intensity import neutral_information, stance_intensity  # noqa: E402
from election_forecast.stance_text_model import label_to_polarity  # noqa: E402


DATA_DIR = ROOT / "outputs" / "assembly_stance" / "stance_context_model_5000"
OUTPUT = DATA_DIR / "context_intensity_predictions_5000.csv"


def main() -> None:
    frame = pd.read_csv(DATA_DIR / "stance_context_5000.csv", encoding="utf-8-sig").fillna("")
    artifact = joblib.load(DATA_DIR / "stance_context_5000_v1.joblib")
    inputs = pd.Series([compose_context_input(row) for row in frame.to_dict(orient="records")])
    probabilities = artifact["model"].predict_proba(inputs.astype(str))
    classes = np.asarray(artifact["classes"])
    class_index = {str(label): index for index, label in enumerate(classes)}
    required = {"negative", "neutral", "positive"}
    if set(class_index) != required:
        raise RuntimeError(f"unexpected model classes: {classes.tolist()}")

    best_indices = np.argmax(probabilities, axis=1)
    predictions = classes[best_indices]
    output = frame.copy()
    output["context_model_label"] = predictions
    output["context_model_polarity"] = [label_to_polarity(str(label)) for label in predictions]
    for label in ("negative", "neutral", "positive"):
        output[f"context_model_probability_{label}"] = probabilities[:, class_index[label]]
    output["context_model_probability"] = probabilities[np.arange(len(output)), best_indices]
    sorted_probabilities = np.sort(probabilities, axis=1)
    output["context_model_probability_margin"] = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )

    intensity_rows: list[dict[str, object]] = []
    for row, probability in zip(output.to_dict(orient="records"), probabilities, strict=True):
        intensity = stance_intensity(
            probability[class_index["negative"]],
            probability[class_index["neutral"]],
            probability[class_index["positive"]],
            row.get("text_excerpt", ""),
        )
        information = neutral_information(
            probability[class_index["neutral"]],
            row.get("text_excerpt", ""),
            issue_name=row.get("issue_name", ""),
            context_before=row.get("context_before", ""),
            context_after=row.get("context_after", ""),
        )
        intensity_rows.append(
            {
                "positive_strength_score": intensity.positive_strength,
                "negative_strength_score": intensity.negative_strength,
                "positive_strength_label": intensity.positive_label,
                "negative_strength_label": intensity.negative_label,
                "directional_score": intensity.directional_score,
                "directional_strength": intensity.directional_strength,
                "emphasis_score": intensity.emphasis_score,
                "information_content_score": information.content_score,
                "neutral_information_score": information.neutral_information_score,
                "neutral_information_label": information.label,
                "information_analysis_flag": information.analysis_flag,
                "information_impact_flag": information.impact_flag,
                "information_evidence_flag": information.evidence_flag,
                "information_procedural_flag": information.procedural_flag,
            }
        )
    output = pd.concat([output, pd.DataFrame(intensity_rows)], axis=1)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    state = {
        "status": "complete",
        "rows": int(len(output)),
        "unique_text_hashes": int(output["text_sha256"].nunique()),
        "model_version": artifact["model_version"],
        "context_label_counts": output["context_model_label"].value_counts().to_dict(),
        "positive_strength_labels": output["positive_strength_label"].value_counts().to_dict(),
        "negative_strength_labels": output["negative_strength_label"].value_counts().to_dict(),
        "neutral_information_labels": output["neutral_information_label"].value_counts().to_dict(),
        "mean_positive_strength": float(output["positive_strength_score"].mean()),
        "mean_negative_strength": float(output["negative_strength_score"].mean()),
        "mean_neutral_information": float(output["neutral_information_score"].mean()),
        "note": (
            "Strength is posterior contrast plus capped emphasis; neutral information is a "
            "transparent weak label and never creates vote direction."
        ),
    }
    (DATA_DIR / "intensity_application_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
