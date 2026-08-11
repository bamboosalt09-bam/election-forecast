"""Apply the quantitative stance correction model to review-batch CSV files."""

from __future__ import annotations

import argparse
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

from election_forecast.stance_text_model import apply_rule_correction, label_to_polarity  # noqa: E402


DEFAULT_MODEL = (
    ROOT / "outputs" / "assembly_stance" / "stance_text_model_v1" / "stance_text_model_v1.joblib"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "assembly_stance" / "stance_text_model_v1" / "corrected_pilots"
ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")


def apply_frame(frame: pd.DataFrame, artifact: dict[str, object]) -> pd.DataFrame:
    required = {
        "text_excerpt",
        "rule_stance_label",
        "rule_stance_polarity",
        "rule_stance_confidence",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    model = artifact["model"]
    classes = np.asarray(artifact["classes"])
    texts = frame["text_excerpt"].fillna("").astype(str).to_numpy()
    probabilities = model.predict_proba(texts)
    model_labels = classes[np.argmax(probabilities, axis=1)]
    sorted_probabilities = np.sort(probabilities, axis=1)
    max_probability = sorted_probabilities[:, -1]
    probability_margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    legacy_labels = (
        pd.to_numeric(frame["rule_stance_polarity"], errors="coerce")
        .fillna(0)
        .astype(int)
        .map({-1: "negative", 0: "neutral", 1: "positive"})
        .to_numpy()
    )
    corrected = apply_rule_correction(
        probabilities,
        classes,
        legacy_labels,
        min_override_probability=float(artifact["min_override_probability"]),
        min_probability_margin=float(artifact["min_probability_margin"]),
        allow_neutral_source=bool(artifact.get("allow_neutral_source", False)),
    )
    override = corrected != legacy_labels

    out = frame.copy()
    out["legacy_rule_stance_label"] = out["rule_stance_label"]
    out["legacy_rule_stance_polarity"] = out["rule_stance_polarity"]
    out["legacy_rule_stance_confidence"] = out["rule_stance_confidence"]
    out["quant_model_argmax_label"] = model_labels
    out["quant_model_max_probability"] = max_probability
    out["quant_model_probability_margin"] = probability_margin
    for index, label in enumerate(classes):
        out[f"quant_model_probability_{label}"] = probabilities[:, index]
    out["quant_model_overridden"] = override.astype(int)
    out["quant_model_corrected_label"] = corrected
    out["quant_model_corrected_polarity"] = [label_to_polarity(label) for label in corrected]

    negative = override & (corrected == "negative")
    positive = override & (corrected == "positive")
    neutral = override & (corrected == "neutral")
    out.loc[negative, "rule_stance_label"] = "attack"
    out.loc[positive, "rule_stance_label"] = "endorse"
    out.loc[neutral, "rule_stance_label"] = "neutral"
    out.loc[override, "rule_stance_polarity"] = out.loc[
        override, "quant_model_corrected_polarity"
    ].astype(int)
    # Existing feature builders require >=0.60 for directional evidence. This
    # is an effective aggregation weight, while the raw model probability is
    # retained separately and must not be described as calibrated confidence.
    directional_override = negative | positive
    out.loc[directional_override, "rule_stance_confidence"] = (
        0.60
        + 0.40
        * (
            out.loc[directional_override, "quant_model_max_probability"]
            - float(artifact["min_override_probability"])
        )
        / (1.0 - float(artifact["min_override_probability"]))
    ).clip(0.60, 1.0)
    out.loc[neutral, "rule_stance_confidence"] = 0.20
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for election_id in ELECTIONS:
        source = (
            ROOT
            / "outputs"
            / "assembly_stance"
            / f"pilot_{election_id}_{args.sample_size}"
            / "review_batch.csv"
        )
        frame = pd.read_csv(source, encoding="utf-8-sig")
        corrected = apply_frame(frame, artifact)
        destination = args.output_dir / f"{election_id}.csv"
        corrected.to_csv(destination, index=False, encoding="utf-8-sig")
        summaries.append(
            {
                "election_id": election_id,
                "rows": int(len(corrected)),
                "overrides": int(corrected["quant_model_overridden"].sum()),
                "directional_before": int(
                    pd.to_numeric(corrected["legacy_rule_stance_polarity"], errors="coerce")
                    .fillna(0)
                    .ne(0)
                    .sum()
                ),
                "directional_after": int(
                    pd.to_numeric(corrected["rule_stance_polarity"], errors="coerce")
                    .fillna(0)
                    .ne(0)
                    .sum()
                ),
                "source": str(source),
                "output": str(destination),
            }
        )
    (args.output_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "model_version": artifact["model_version"],
                "policy": artifact["policy"],
                "model_metadata_columns": artifact["metadata_columns"],
                "sample_size_per_election": args.sample_size,
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
