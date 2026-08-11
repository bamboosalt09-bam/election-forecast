"""Summarize V20 label prevalence on the broad and prior fresh corpora."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEW = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_strict_owner_v20"
    / "broad_analysis_10000_v20"
    / "context_predictions_v20.csv"
)
PRIOR = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_context_strict_owner_v20"
)
OUTPUT = PRIOR / "broad_analysis_10000_v20"


def distribution(frame: pd.DataFrame) -> dict[str, object]:
    counts = frame["v20_prediction"].value_counts().to_dict()
    rows = len(frame)
    return {
        "rows": rows,
        "positive": int(counts.get("positive", 0)),
        "negative": int(counts.get("negative", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "positive_rate": float(counts.get("positive", 0) / max(rows, 1)),
        "negative_rate": float(counts.get("negative", 0) / max(rows, 1)),
        "neutral_rate": float(counts.get("neutral", 0) / max(rows, 1)),
        "directional_rate": float(
            (counts.get("positive", 0) + counts.get("negative", 0)) / max(rows, 1)
        ),
        "ratio_positive_negative_neutral": (
            f"{counts.get('positive', 0)}:{counts.get('negative', 0)}:"
            f"{counts.get('neutral', 0)}"
        ),
    }


def grouped(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    counts = (
        frame.groupby([dimension, "v20_prediction"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for label in ("positive", "negative", "neutral"):
        if label not in counts:
            counts[label] = 0
    counts = counts[["positive", "negative", "neutral"]].reset_index()
    counts["rows"] = counts[["positive", "negative", "neutral"]].sum(axis=1)
    counts["directional"] = counts["positive"] + counts["negative"]
    counts["directional_rate"] = counts["directional"] / counts["rows"]
    counts.insert(0, "dimension", dimension)
    counts = counts.rename(columns={dimension: "value"})
    return counts


def main() -> None:
    new = pd.read_csv(NEW, encoding="utf-8-sig", low_memory=False).fillna("")
    prior_e = pd.read_csv(
        PRIOR / "supplement_e_v20" / "context_predictions_v20.csv",
        encoding="utf-8-sig",
        low_memory=False,
    ).fillna("")
    prior_f = pd.read_csv(
        PRIOR / "supplement_f_v20" / "context_predictions_v20.csv",
        encoding="utf-8-sig",
        low_memory=False,
    ).fillna("")
    prior = pd.concat([prior_e, prior_f], ignore_index=True)
    combined = pd.concat([prior, new], ignore_index=True)
    representative = new.loc[new["sample_component"].eq("representative")]
    supplement = new.loc[~new["sample_component"].eq("representative")]

    metrics = {
        "model_version": "stance_context_strict_owner_v20",
        "active_forecast_changed": False,
        "post_2022_rows_present": False,
        "vote_outcomes_used": False,
        "prior_fresh_10000": distribution(prior),
        "new_broad_10000": distribution(new),
        "new_representative_5000": distribution(representative),
        "new_coverage_supplement_5000": distribution(supplement),
        "combined_20000": distribution(combined),
        "base_new_broad_10000": {
            "rows": int(len(new)),
            "prediction_counts": new["context_prediction"].value_counts().to_dict(),
            "directional_rate": float(new["context_prediction"].ne("neutral").mean()),
        },
    }
    (OUTPUT / "broad_v20_distribution.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tables = pd.concat(
        [grouped(new, dimension) for dimension in ("election_id", "assembly_daesu", "target_type", "issue_name")],
        ignore_index=True,
    )
    tables.to_csv(OUTPUT / "broad_v20_group_distribution.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
