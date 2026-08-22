"""Report the regional metric under ex-ante weightings, not only the post-hoc one.

The headline regional figure weights each candidate-region error by
``contest_votes`` - the votes actually cast in the target election. That is a
reasonable post-election diagnostic, but it is not a weighting a forecaster
holds on the eve of the election, so it cannot be quoted as predictive accuracy
without qualification.

Three weightings are reported side by side:

    contest_votes        the shipped headline; the target election's own turnout
    equal_region         every region counts once; needs no data at all
    prior_election_votes the previous scored election's regional volumes

``equal_region`` is the conservative floor: it is always available and assumes
nothing. ``prior_election_votes`` is the closest ex-ante analogue of the
shipped weighting, and it is only defined from the second scored election
onward, since the first has no predecessor in the panel.

A region absent from the previous election - 세종 first appears in 2012 - has
no prior volume, and is given that election's mean regional volume rather than
being dropped, so the panel stays complete. Those substitutions are counted and
reported.

No election outcome is used to choose between the weightings; all three are
printed.
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

ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v26"
OUTPUT_DIR = ROOT / "outputs" / "ex_ante_weighting"
# chronological, so "previous" is well defined
ORDER = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")


def regional_volumes(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per election and region carrying that contest's vote volume."""

    return (
        frame.groupby(["election_id", "region_id"], as_index=False)["contest_votes"]
        .first()
        .rename(columns={"contest_votes": "volume"})
    )


def prior_election_weights(volumes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map each election's regions onto the previous election's volumes."""

    lookup = {
        election: group.set_index("region_id")["volume"]
        for election, group in volumes.groupby("election_id")
    }
    rows: list[dict[str, object]] = []
    substituted: dict[str, int] = {}
    for index, election in enumerate(ORDER):
        if index == 0 or election not in lookup:
            continue
        previous = lookup.get(ORDER[index - 1])
        if previous is None or previous.empty:
            continue
        fallback = float(previous.mean())
        missing = 0
        for region in lookup[election].index:
            if region in previous.index:
                weight = float(previous[region])
            else:
                weight = fallback
                missing += 1
            rows.append(
                {"election_id": election, "region_id": region, "weight": weight}
            )
        substituted[election] = missing
    return pd.DataFrame(rows), substituted


def _weighted_regional_mae(group: pd.DataFrame, weights: pd.Series) -> float:
    aligned = group["region_id"].map(weights)
    if aligned.isna().any() or float(aligned.sum()) <= 0.0:
        return float("nan")
    return float(np.average((group.layer_pred - group.actual).abs(), weights=aligned) * 100)


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    volumes = regional_volumes(frame)
    prior, substituted = prior_election_weights(volumes)
    prior_lookup = {
        election: group.set_index("region_id")["weight"]
        for election, group in prior.groupby("election_id")
    }
    own_lookup = {
        election: group.set_index("region_id")["volume"]
        for election, group in volumes.groupby("election_id")
    }

    rows: list[dict[str, object]] = []
    for election, group in frame.groupby("election_id"):
        election = str(election)
        equal = pd.Series(1.0, index=sorted(group["region_id"].unique()))
        rows.append(
            {
                "election_id": election,
                "contest_votes_pp": _weighted_regional_mae(group, own_lookup[election]),
                "equal_region_pp": _weighted_regional_mae(group, equal),
                "prior_election_votes_pp": (
                    _weighted_regional_mae(group, prior_lookup[election])
                    if election in prior_lookup
                    else float("nan")
                ),
                "regions_substituted": substituted.get(election, 0),
            }
        )
    return pd.DataFrame(rows).sort_values("election_id").reset_index(drop=True), substituted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-dir", type=Path, default=ACTIVE_DIR)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    frame = pd.read_csv(
        args.active_dir / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    by_election, substituted = evaluate(frame)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")

    # The prior-election weighting is undefined for the first election, so a
    # macro over all five would compare it against a different election set.
    # Both are reported: the full panel, and the subset all three cover.
    matched = by_election.loc[by_election["prior_election_votes_pp"].notna()]
    columns = {
        "contest_votes (shipped, post-hoc)": "contest_votes_pp",
        "equal_region (ex ante)": "equal_region_pp",
        "prior_election_votes (ex ante)": "prior_election_votes_pp",
    }
    summary = pd.DataFrame(
        [
            {
                "weighting": name,
                "macro_all_scored_pp": by_election[column].mean(),
                "macro_matched_pp": matched[column].mean(),
            }
            for name, column in columns.items()
        ]
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    print(by_election.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    defined = int(by_election["prior_election_votes_pp"].notna().sum())
    print()
    print(
        f"macro_matched_pp covers the {defined} elections all three weightings define;"
        f" macro_all_scored_pp covers all {len(by_election)}, so only the matched"
        " column compares the three directly."
    )
    if any(substituted.values()):
        detail = ", ".join(f"{k} {v}" for k, v in substituted.items() if v)
        print(f"regions given the previous election's mean volume: {detail}")


if __name__ == "__main__":
    main()
