"""Isolated ablation and order permutation of the three structural postprocesses.

V26 applies `strong_incumbent_veto`, `third_candidate_lineage_ceiling` and
`weak_same_lane_refusal` in that order. The three are not commutative: each
renormalises the contest, so an earlier layer changes the recipient weights the
next one redistributes at. On the 2025 target that compounds - the veto widens
the two-major gap from 7.80 to 17.48 points, and the ceiling then splits the
third candidate's recovered mass at the already-widened ratio, adding a further
3.18. Isolated on/off tests cannot see that, because the interaction lives in
the ordering rather than in any single layer.

This harness measures both at once. The expensive nested run happens once: the
three layers are pure transforms applied after it, so every cell reuses one
captured pre-postprocess frame.

    16 cells = 1 empty + 3 singletons + 3 pairs x 2 orders + 6 triple orders

Scope: a development comparison over the through-2022 sample. Each cell is
point-in-time safe, but the same five outcomes measure every cell, so the table
is not an untouched holdout. No 2025 outcome is read.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

OUTPUT_DIR = ROOT / "outputs" / "postprocess_ablation"
BASE_CACHE = OUTPUT_DIR / "base_predictions.csv"
LAYER_ORDER = ("veto", "ceiling", "refusal")


def _layers() -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    from presidential_issue_engine import strong_incumbent_veto
    from presidential_issue_engine import third_candidate_lineage_constraint
    from presidential_issue_engine import weak_same_lane_refusal

    return {
        "veto": lambda frame: strong_incumbent_veto.apply_strong_incumbent_veto(frame)[0],
        "ceiling": lambda frame: third_candidate_lineage_constraint.apply_lineage_ceiling(
            frame
        )[0],
        "refusal": lambda frame: weak_same_lane_refusal.apply_weak_same_lane_refusal(
            frame
        )[0],
    }


def capture_base(*, refresh: bool = False) -> pd.DataFrame:
    """Return the predictions as they enter the first postprocess layer.

    The frame is captured by intercepting the first layer rather than by
    reconstructing it, so it is exactly what V26 feeds the stack.
    """

    if BASE_CACHE.exists() and not refresh:
        return pd.read_csv(BASE_CACHE, encoding="utf-8-sig", low_memory=False)

    from presidential_issue_engine import strong_incumbent_veto
    from scripts import run_active_presidential_model_v26 as v26

    captured: list[pd.DataFrame] = []
    original = strong_incumbent_veto.apply_strong_incumbent_veto

    def capturing(frame, *args, **kwargs):
        captured.append(frame.copy())
        return original(frame, *args, **kwargs)

    strong_incumbent_veto.apply_strong_incumbent_veto = capturing
    try:
        v26.run(output_dir=OUTPUT_DIR / "_capture_run")
    finally:
        strong_incumbent_veto.apply_strong_incumbent_veto = original

    if not captured:
        raise RuntimeError("the postprocess stack did not run; nothing was captured")
    base = captured[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.to_csv(BASE_CACHE, index=False, encoding="utf-8-sig")
    return base


def apply_sequence(base: pd.DataFrame, sequence: Sequence[str]) -> pd.DataFrame:
    layers = _layers()
    frame = base.copy()
    for name in sequence:
        frame = layers[name](frame)
    return frame


def _metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
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
        weighted_regional = float(
            np.average(
                (group.layer_pred - group.actual).abs(), weights=group.contest_votes
            )
            * 100
        )
        rows.append(
            {
                "cell": label,
                "election_id": str(election),
                "regional_weighted_mae_pp": weighted_regional,
                "national_mae_pp": float((predicted - actual).abs().mean() * 100),
                "winner_correct": bool(predicted.idxmax() == actual.idxmax()),
            }
        )
    return pd.DataFrame(rows)


def cells() -> list[tuple[str, tuple[str, ...]]]:
    """Every on/off subset, and every ordering of the subsets with more than one."""

    out: list[tuple[str, tuple[str, ...]]] = [("none", ())]
    for size in (1, 2, 3):
        for subset in itertools.combinations(LAYER_ORDER, size):
            for order in itertools.permutations(subset):
                label = ">".join(order)
                if size == 1:
                    label = order[0]
                out.append((label, order))
    return out


def run(*, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = capture_base(refresh=refresh)
    frames = [_metrics(apply_sequence(base, order), label) for label, order in cells()]
    by_election = pd.concat(frames, ignore_index=True)
    summary = (
        by_election.groupby("cell", sort=False)
        .agg(
            regional_weighted_macro_pp=("regional_weighted_mae_pp", "mean"),
            national_macro_pp=("national_mae_pp", "mean"),
            winners_correct=("winner_correct", "sum"),
        )
        .reset_index()
    )
    shipped = summary.loc[summary["cell"].eq(">".join(LAYER_ORDER))]
    if not shipped.empty:
        reference = float(shipped["national_macro_pp"].iloc[0])
        summary["national_vs_shipped_pp"] = summary["national_macro_pp"] - reference
    return summary, by_election


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-run the nested model instead of reusing the cached base frame",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    summary, by_election = run(refresh=args.refresh)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    print(summary.sort_values("national_macro_pp").to_string(index=False))


if __name__ == "__main__":
    main()
