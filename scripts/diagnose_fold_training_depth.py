"""Report how much training each fold had, beside what it cost.

Strict chronological nesting means the first scored election is predicted from
the warmup alone. That is correct - it is the point of the design - but it also
means the folds are not comparable to each other, and an equal-election macro
averages a fold fitted on one election with a fold fitted on five.

This is read-only. It changes no model and proposes no change; it reports a
property of the evaluation design that the headline figures hide.

Two things it separates that are easy to conflate:

    regional shape   does the model get the pattern across regions right
    national level   does it get the balance between candidates right

A fold can be good at one and bad at the other, and the panel's worst fold is
worst at only one of them.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.active_model_pointer import active_output_dir

ACTIVE_DIR = active_output_dir()
OUTPUT_DIR = ROOT / "outputs" / "fold_training_depth"


def fold_depth(active_dir: Path) -> pd.DataFrame:
    """Training elections and design condition per fold, from the fold audit."""

    audit = pd.read_csv(active_dir / "fold_audit.csv", encoding="utf-8-sig")
    columns = ["target_election", "training_elections"]
    if "raw_max_predictor_vif" in audit.columns:
        columns.append("raw_max_predictor_vif")
    if "predictor_count" in audit.columns:
        columns.append("predictor_count")
    depth = audit[columns].drop_duplicates().copy()
    depth["training_election_count"] = (
        depth["training_elections"].astype(str).str.count(r"\|") + 1
    )
    return depth.sort_values("training_election_count").reset_index(drop=True)


def fold_cost(active_dir: Path) -> pd.DataFrame:
    """Regional shape and national level error per fold."""

    frame = pd.read_csv(
        active_dir / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    column = (
        "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"
    )
    rows: list[dict[str, object]] = []
    for election, group in frame.groupby("election_id"):
        actual = group.groupby(column).apply(
            lambda part: np.average(part.actual, weights=part.contest_votes)
        )
        predicted = group.groupby(column).apply(
            lambda part: np.average(part.layer_pred, weights=part.contest_votes)
        )
        rows.append(
            {
                "target_election": str(election),
                "regional_weighted_mae_pp": float(
                    np.average(
                        (group.layer_pred - group.actual).abs(),
                        weights=group.contest_votes,
                    )
                    * 100
                ),
                "national_mae_pp": float((predicted - actual).abs().mean() * 100),
                "winner_correct": bool(predicted.idxmax() == actual.idxmax()),
            }
        )
    return pd.DataFrame(rows)


def report(active_dir: Path = ACTIVE_DIR) -> pd.DataFrame:
    merged = fold_depth(active_dir).merge(fold_cost(active_dir), on="target_election")
    total = merged["national_mae_pp"].sum()
    merged["share_of_national_macro"] = (
        merged["national_mae_pp"] / total if total else np.nan
    )
    return merged.sort_values("training_election_count").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-dir", type=Path, default=ACTIVE_DIR)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    merged = report(args.active_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_DIR / "fold_training_depth.csv", index=False, encoding="utf-8-sig")

    display = merged[
        [
            "target_election",
            "training_election_count",
            "raw_max_predictor_vif",
            "regional_weighted_mae_pp",
            "national_mae_pp",
            "winner_correct",
            "share_of_national_macro",
        ]
    ]
    print(display.round(4).to_string(index=False))
    print()

    shallowest = merged.iloc[0]
    print(
        f"{shallowest['target_election']} trains on"
        f" {int(shallowest['training_election_count'])} election(s) and carries"
        f" {shallowest['share_of_national_macro']:.0%} of the national macro."
    )
    if len(merged) > 2:
        correlation = merged["training_election_count"].corr(merged["national_mae_pp"])
        print(
            f"training depth against national MAE: r = {correlation:.3f} over"
            f" {len(merged)} folds - a pattern at this size, not evidence."
        )
    print()
    print(
        "The shallowest fold is a property of strict chronological nesting, not a"
        " defect to tune away. It does mean an equal-election macro averages folds"
        " that had very different amounts of training."
    )


if __name__ == "__main__":
    main()
