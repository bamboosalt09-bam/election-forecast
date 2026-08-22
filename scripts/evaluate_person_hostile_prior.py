"""Cap a candidate where their own prior record says the region is hostile.

`electorate_layers` keys every history feature on (region_id, bloc), so no
feature identifies a person across elections. A candidate who changes vehicle
loses their record entirely: 이회창's `candidate_ballot_effective_elections` in
광주 is 5.2416 in 2002 and 0.0000 in 2007. With nothing to anchor him the base
puts him at 11.70 where he took 3.71.

His own prior record says 3.58.

The asymmetry that makes this usable is in the data rather than assumed. Across
his sixteen regions between 2002 and 2007, while his national share collapsed
from 46.6 % to 15.1 %:

    regions where he had polled below 15 %   mean absolute change  1.25 %p
    regions where he had polled at or above  mean absolute change 33.81 %p

Hostile-territory floors are structural and survive a collapse in national
standing; favourable-territory shares scale with it. So a candidate's prior
share is a usable estimate exactly where it was low, which is exactly where the
model fails.

The rule is therefore one-sided and self-scoping:

    for each candidate with a prior presidential record in this region,
    where that prior share was below their own prior national share,
    cap the prediction at the prior share and redistribute the excess.

"Hostile" is defined by the candidate's own prior national level rather than by
a threshold, so no constant is introduced. The rule can only lower, and only
where the model sits above a level the candidate has already been held to.

Only prior elections are read. Nothing about the target election is used.
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

RESULTS = ROOT / "presidential_issue_engine" / "fixed_dataset" / "presidential_results_standardized.csv"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v26"
OUTPUT_DIR = ROOT / "outputs" / "person_hostile_prior"
ORDER = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
# Aggregate rows are not people and must never supply a personal prior.
NON_PERSON_NAMES = ("기타후보 합산",)


def _name_column(frame: pd.DataFrame) -> str:
    return "candidate_name_x" if "candidate_name_x" in frame.columns else "candidate_name"


def hostile_priors(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """The most recent prior share per person and region, where it was hostile."""

    frame = (
        pd.read_csv(RESULTS, encoding="utf-8-sig") if results is None else results.copy()
    )
    frame = frame.loc[~frame["candidate_name"].isin(NON_PERSON_NAMES)]
    position = {election: index for index, election in enumerate(ORDER)}
    frame = frame.loc[frame["election_id"].isin(position)].copy()
    frame["order"] = frame["election_id"].map(position)

    national = (
        frame.groupby(["election_id", "candidate_name"])["vote_share"]
        .mean()
        .rename("national")
    )
    frame = frame.merge(national, on=["election_id", "candidate_name"])

    rows: list[dict[str, object]] = []
    for target in ORDER:
        prior = frame.loc[frame["order"] < position[target]]
        if prior.empty:
            continue
        # the most recent prior appearance wins
        latest = prior.sort_values("order").drop_duplicates(
            ["candidate_name", "region_id"], keep="last"
        )
        hostile = latest.loc[latest["vote_share"] < latest["national"]]
        for row in hostile.itertuples(index=False):
            rows.append(
                {
                    "election_id": target,
                    "candidate_name": str(row.candidate_name),
                    "region_id": str(row.region_id),
                    "prior_election": str(row.election_id),
                    "prior_share": float(row.vote_share),
                }
            )
    return pd.DataFrame(rows)


LOST_HISTORY_COLUMN = "candidate_ballot_effective_elections"


def lost_history(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Election and candidate pairs whose modelled history is empty.

    This is the defect the personal prior is meant to fill. A candidate whose
    bloc channel is populated already has a working anchor, and overriding it
    with a stale personal level is a different and unjustified intervention -
    문재인 kept his party between 2012 and 2017 and his hostile-region support
    rose, so capping him at his own earlier level is simply wrong.
    """

    if LOST_HISTORY_COLUMN not in frame.columns:
        return set()
    name = _name_column(frame)
    depth = frame.groupby(["election_id", name])[LOST_HISTORY_COLUMN].max()
    return {
        (str(election), str(candidate))
        for (election, candidate), value in depth.items()
        if float(value) <= 0.0
    }


def apply_cap(
    frame: pd.DataFrame,
    priors: pd.DataFrame,
    *,
    only_lost_history: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap at the personal hostile prior and redistribute within each region."""

    name = _name_column(frame)
    eligible = lost_history(frame) if only_lost_history else None
    lookup = {
        (str(row.election_id), str(row.candidate_name), str(row.region_id)): float(
            row.prior_share
        )
        for row in priors.itertuples(index=False)
    }
    out = frame.copy()
    audit: list[dict[str, object]] = []
    for (election, region), group in out.groupby(["election_id", "region_id"], sort=False):
        capped_total = 0.0
        capped_index: list[int] = []
        for index, row in group.iterrows():
            if eligible is not None and (str(election), str(row[name])) not in eligible:
                continue
            prior = lookup.get((str(election), str(row[name]), str(region)))
            if prior is None:
                continue
            current = float(row["layer_pred"])
            if current <= prior:
                continue
            out.at[index, "layer_pred"] = prior
            capped_total += current - prior
            capped_index.append(index)
            audit.append(
                {
                    "election_id": str(election),
                    "region_id": str(region),
                    "candidate_name": str(row[name]),
                    "before": current,
                    "prior_cap": prior,
                    "released": current - prior,
                }
            )
        if capped_total <= 0.0:
            continue
        recipients = group.index.difference(capped_index)
        weights = out.loc[recipients, "layer_pred"]
        total = float(weights.sum())
        if total <= 0.0:
            continue
        out.loc[recipients, "layer_pred"] = weights + capped_total * (weights / total)
    denominator = out.groupby(["election_id", "region_id"])["layer_pred"].transform("sum")
    out["layer_pred"] = out["layer_pred"] / denominator
    return out, pd.DataFrame(audit)


def apply_reshape(
    frame: pd.DataFrame, priors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold the candidate's level and move the excess to their own other regions.

    Capping alone breaks what was already right. 이회창's national prediction is
    14.81 against a realised 15.10, so his level needs no correction at all -
    only his shape does. Handing the capped mass to rival candidates lowers a
    level that was accurate, which is why the cap improves regional error and
    degrades national error at the same time.

    Here the released mass stays with the candidate and is redistributed across
    their remaining regions in proportion to the vote each region casts, so the
    vote-weighted national level is preserved by construction.
    """

    name = _name_column(frame)
    eligible = lost_history(frame)
    lookup = {
        (str(row.election_id), str(row.candidate_name), str(row.region_id)): float(
            row.prior_share
        )
        for row in priors.itertuples(index=False)
    }
    out = frame.copy()
    audit: list[dict[str, object]] = []
    for (election, candidate), group in out.groupby(["election_id", name], sort=False):
        if (str(election), str(candidate)) not in eligible:
            continue
        capped = {}
        for index, row in group.iterrows():
            prior = lookup.get((str(election), str(candidate), str(row["region_id"])))
            if prior is None:
                continue
            current = float(row["layer_pred"])
            if current > prior:
                capped[index] = (current, prior)
        if not capped:
            continue
        released_votes = sum(
            (current - prior) * float(out.at[index, "contest_votes"])
            for index, (current, prior) in capped.items()
        )
        recipients = group.index.difference(list(capped))
        if not len(recipients) or released_votes <= 0.0:
            continue
        for index, (current, prior) in capped.items():
            out.at[index, "layer_pred"] = prior
            audit.append(
                {
                    "election_id": str(election),
                    "candidate_name": str(candidate),
                    "region_id": str(out.at[index, "region_id"]),
                    "before": current,
                    "prior_cap": prior,
                    "released": current - prior,
                }
            )
        votes = out.loc[recipients, "contest_votes"].astype(float)
        out.loc[recipients, "layer_pred"] = out.loc[recipients, "layer_pred"] + (
            released_votes / float(votes.sum())
        )
    denominator = out.groupby(["election_id", "region_id"])["layer_pred"].transform("sum")
    out["layer_pred"] = out["layer_pred"] / denominator
    return out, pd.DataFrame(audit)


def metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    name = _name_column(frame)
    rows: list[dict[str, object]] = []
    for election, group in frame.groupby("election_id"):
        actual = group.groupby(name).apply(
            lambda part: np.average(part.actual, weights=part.contest_votes)
        )
        predicted = group.groupby(name).apply(
            lambda part: np.average(part.layer_pred, weights=part.contest_votes)
        )
        rows.append(
            {
                "variant": label,
                "election_id": str(election),
                "regional_weighted_mae_pp": float(
                    np.average(
                        (group.layer_pred - group.actual).abs(),
                        weights=group.contest_votes,
                    )
                    * 100
                ),
                "national_mae_pp": float((predicted - actual).abs().mean() * 100),
                "worst_cell_pp": float((group.layer_pred - group.actual).abs().max() * 100),
                "winner_correct": bool(predicted.idxmax() == actual.idxmax()),
            }
        )
    return pd.DataFrame(rows)


def run(active_dir: Path = ACTIVE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(
        active_dir / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    priors = hostile_priors()
    scoped, scoped_audit = apply_cap(frame, priors, only_lost_history=True)
    broad, _ = apply_cap(frame, priors, only_lost_history=False)
    audit = scoped_audit
    by_election = pd.concat(
        [
            metrics(frame, "baseline"),
            metrics(scoped, "cap_only"),
            metrics(broad, "cap_every_repeat"),
            metrics(apply_reshape(frame, priors)[0], "reshape_level_held"),
        ],
        ignore_index=True,
    )
    summary = (
        by_election.groupby("variant", sort=False)
        .agg(
            regional_weighted_macro_pp=("regional_weighted_mae_pp", "mean"),
            national_macro_pp=("national_mae_pp", "mean"),
            worst_cell_pp=("worst_cell_pp", "max"),
            winners_correct=("winner_correct", "sum"),
        )
        .reset_index()
    )
    return summary, by_election, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-dir", type=Path, default=ACTIVE_DIR)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    summary, by_election, audit = run(args.active_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUTPUT_DIR / "audit.csv", index=False, encoding="utf-8-sig")

    print(summary.round(4).to_string(index=False))
    print()
    print(
        by_election.pivot(
            index="election_id", columns="variant", values="regional_weighted_mae_pp"
        ).round(3).to_string()
    )
    print()
    print(f"cells capped: {len(audit)}")
    if not audit.empty:
        print(
            audit.groupby(["election_id", "candidate_name"])
            .agg(regions=("region_id", "nunique"), mean_released=("released", "mean"))
            .round(4)
            .to_string()
        )


if __name__ == "__main__":
    main()
