"""Freeze a target-bearing stance audit before reading its labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "precision_contrastive_ensemble_v7"
    / "application_5000"
    / "stance_precision_predictions_5000.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "shadow" / "stance_locked_audit_v1.csv"
TARGET_TYPES = {"person", "party", "government"}
SEED = "stance-locked-audit-v1-before-review"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(text_hash: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{text_hash}".encode("utf-8")).hexdigest()


def _excluded_hashes() -> set[str]:
    paths = [
        ROOT / "data" / "shadow" / "stance_precision_gold_through2022.csv",
        ROOT / "data" / "shadow" / "stance_manual_gold_expansion_v1.csv",
        ROOT
        / "outputs"
        / "assembly_stance"
        / "precision_augmented_ensemble_v4"
        / "application_5000"
        / "directional_audit_sample.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v1.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v2.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v3.csv",
        ROOT / "data" / "shadow" / "stance_locked_audit_v4.csv",
    ]
    values: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["text_sha256"])
        values.update(frame["text_sha256"].dropna().astype(str))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--prediction-column", default="ensemble_prediction")
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()
    predictions_path = args.predictions.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"locked audit already exists: {output_path}")

    frame = pd.read_csv(predictions_path, encoding="utf-8-sig").fillna("")
    eligible = frame.loc[
        frame["target_type"].isin(TARGET_TYPES)
        & frame[args.prediction_column].ne("neutral")
        & ~frame["text_sha256"].astype(str).isin(_excluded_hashes())
    ].copy()
    eligible["audit_rank"] = eligible["text_sha256"].astype(str).map(
        lambda value: _rank(value, args.seed)
    )
    eligible = eligible.sort_values("audit_rank")
    selected = (
        eligible.groupby(
            ["election_id", "target_type", args.prediction_column],
            group_keys=False,
        )
        .head(3)
        .sort_values("audit_rank")
    )
    if len(selected) < args.rows:
        remaining = eligible.loc[~eligible.index.isin(selected.index)]
        selected = pd.concat(
            [selected, remaining.head(args.rows - len(selected))], ignore_index=False
        )
    selected = selected.sort_values("audit_rank").head(args.rows).copy()
    selected["audit_locked_label"] = ""
    selected["audit_target_correct"] = ""
    selected["audit_quotation_owner"] = ""
    selected["audit_notes"] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False, encoding="utf-8-sig")
    state = {
        "status": "locked_before_review",
        "rows": int(len(selected)),
        "seed": args.seed,
        "source": str(predictions_path),
        "source_sha256": _sha256(predictions_path),
        "output": str(output_path),
        "output_sha256_before_review": _sha256(output_path),
        "election_counts": selected["election_id"].value_counts().sort_index().to_dict(),
        "target_counts": selected["target_type"].value_counts().to_dict(),
        "prediction_column": args.prediction_column,
        "prediction_counts": selected[args.prediction_column].value_counts().to_dict(),
    }
    state_path = output_path.with_suffix(".lock.json")
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
