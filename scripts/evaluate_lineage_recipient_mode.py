"""Compare live and reference recipient weights for the third-candidate ceiling.

The ceiling takes mass off a self-founded third candidate and hands it to the
two majors. It currently splits at the prediction as it stands when the ceiling
runs, so whatever an earlier postprocess did to the two majors decides the
split. On the 2025 target that compounds: the strong incumbent veto widens the
two-major gap from 7.80 to 17.48 points, and the ceiling then redistributes at
that widened ratio, adding a further 3.18.

``reference`` mode splits at a column no postprocess writes to.

Read the scored comparison with its limit in view: the ceiling binds for
권영길 2002 alone, so the panel carries **one** observation of this rule, and
2002 is also the only scored election where two layers meet. The 2025 figures
below are a mechanism demonstration, not evidence - no outcome is read for them.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from presidential_issue_engine import third_candidate_lineage_constraint as ceiling  # noqa: E402
from scripts import evaluate_postprocess_ablation as ablation  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "lineage_recipient_mode"
PROSPECTIVE = ROOT / "outputs" / "prospective_pres_2025_v25" / "prediction_stage_audit.csv"
SHIPPED_ORDER = ablation.LAYER_ORDER


def _apply(base: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Run the shipped stack with the ceiling in the given recipient mode."""

    from presidential_issue_engine import strong_incumbent_veto
    from presidential_issue_engine import weak_same_lane_refusal

    frame = strong_incumbent_veto.apply_strong_incumbent_veto(base.copy())[0]
    frame = ceiling.apply_lineage_ceiling(frame, recipient_weight_mode=mode)[0]
    return weak_same_lane_refusal.apply_weak_same_lane_refusal(frame)[0]


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
        rows.append(
            {
                "mode": label,
                "election_id": str(election),
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


def target_demonstration() -> pd.DataFrame | None:
    """Show the 2025 two-major gap under both modes. No outcome is read."""

    if not PROSPECTIVE.exists():
        return None
    audit = pd.read_csv(PROSPECTIVE, encoding="utf-8-sig", low_memory=False)
    needed = {"slot", "contest_votes", "v24_post_strong_veto_pred", "anchored_pred"}
    if not needed.issubset(audit.columns):
        return None

    def weighted(column: str) -> pd.Series:
        values = pd.to_numeric(audit[column], errors="coerce")
        return audit.assign(_v=values).groupby("slot").apply(
            lambda part: np.average(part._v, weights=part.contest_votes)
        )

    post_veto = weighted("v24_post_strong_veto_pred")
    reference = weighted("anchored_pred")
    third = float(post_veto.get("C", 0.0))
    ceiling_level = float(weighted("direct_party_recent_base").get("C", 0.0))
    excess = max(third - ceiling_level, 0.0)

    rows = []
    for label, weights in (("live", post_veto), ("reference", reference)):
        share = {slot: float(weights.get(slot, 0.0)) for slot in ("A", "B")}
        total = share["A"] + share["B"]
        split = {slot: share[slot] / total for slot in ("A", "B")} if total else {"A": 0.5, "B": 0.5}
        a = float(post_veto.get("A", 0.0)) + excess * split["A"]
        b = float(post_veto.get("B", 0.0)) + excess * split["B"]
        rows.append(
            {
                "mode": label,
                "slot_A_pp": a * 100,
                "slot_B_pp": b * 100,
                "two_major_gap_pp": (b - a) * 100,
            }
        )
    return pd.DataFrame(rows)


def run(*, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ablation.capture_base(refresh=refresh)
    frames = [_metrics(_apply(base, mode), mode) for mode in ("live", "reference")]
    by_election = pd.concat(frames, ignore_index=True)
    summary = (
        by_election.groupby("mode", sort=False)
        .agg(
            regional_weighted_macro_pp=("regional_weighted_mae_pp", "mean"),
            national_macro_pp=("national_mae_pp", "mean"),
            winners_correct=("winner_correct", "sum"),
        )
        .reset_index()
    )
    return summary, by_election


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    summary, by_election = run(refresh=args.refresh)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")

    print("scored panel (the ceiling binds for 권영길 2002 only: n = 1)")
    print(summary.to_string(index=False))
    print()
    print("by election")
    print(
        by_election.pivot(
            index="election_id", columns="mode", values="national_mae_pp"
        ).round(4).to_string()
    )
    demonstration = target_demonstration()
    if demonstration is not None:
        print()
        print("2025 mechanism demonstration - no outcome is read")
        print(demonstration.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
