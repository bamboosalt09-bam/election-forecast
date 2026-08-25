"""How much of the reported metric depends on the weighting, not the model.

This exists because of a proposal that was investigated and rejected: make the
reported headline weight regional errors by the previous election's volumes,
the way V30's transforms do, instead of by ``contest_votes``.

The proposal was wrong, and the reason is worth keeping. A model reading
``contest_votes`` is a leak - it consumes turnout that does not exist until the
count, which is what V30 fixed. A *metric* aggregating by ``contest_votes`` is
not a leak: national vote share is by definition the vote-weighted mean of
regional shares, the weights are the actual votes, and the model is predicting
shares rather than turnout. Scoring against the real aggregation gives the
model nothing; it states the target correctly.

What the investigation did produce is a sensitivity measurement, which this
script reproduces: how far the reported figures move when the aggregation
weight is swapped. For 2022 that movement crosses the winner call, which says
something real about how thin that call is.

Read-only. It reports and changes nothing. See
``docs/METRIC_WEIGHTING_20260825.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import forecast_time_region_weights as ftw

OUTPUT_DIR = ROOT / "outputs" / "metric_weighting_sensitivity"
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"


def _artifact(version: str) -> Path:
    return ROOT / "outputs" / f"active_presidential_nested_{version}" / "nested_predictions.csv"


def _weighted(frame: pd.DataFrame, weight: pd.Series) -> tuple[float, float, float]:
    """Regional macro, national macro and winner accuracy under one weighting.

    The weighting applies to the **prediction** only. A forecaster aggregates
    regional forecasts into a national one with the turnout they expect, and
    that expectation is what this weight stands for; the realised national
    result is then whatever the actual votes made it, and is always aggregated
    by ``contest_votes``.

    Reweighting the realised side too would compare the forecast against a
    counterfactual - "what the result would have been had regions turned out
    like last time" - which is not a thing that happened and not what anyone
    was trying to predict.
    """

    frame = frame.assign(_w=weight.to_numpy(dtype=float))
    frame["_abs_pp"] = (frame["layer_pred"] - frame["actual"]).abs() * 100.0
    regional: list[float] = []
    national: list[float] = []
    winners = 0
    for _, group in frame.groupby("election_id"):
        regional.append(float(np.average(group["_abs_pp"], weights=group["_w"])))
        errors, levels, actuals = [], {}, {}
        for slot, rows in group.groupby("slot"):
            predicted = float(np.average(rows["layer_pred"], weights=rows["_w"]))
            realised = float(np.average(rows["actual"], weights=rows["contest_votes"]))
            errors.append(abs(predicted - realised) * 100.0)
            levels[str(slot)], actuals[str(slot)] = predicted, realised
        national.append(float(np.mean(errors)))
        winners += int(max(levels, key=levels.get) == max(actuals, key=actuals.get))
    return float(np.mean(regional)), float(np.mean(national)), winners / len(regional)


def by_election(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["forecast_time"] = ftw.build(frame)
    frame["_abs_pp"] = (frame["layer_pred"] - frame["actual"]).abs() * 100.0
    rows = []
    for election, group in frame.groupby("election_id"):
        rows.append(
            {
                "election_id": str(election),
                "contest_votes_pp": float(
                    np.average(group["_abs_pp"], weights=group["contest_votes"])
                ),
                "forecast_time_pp": float(
                    np.average(group["_abs_pp"], weights=group["forecast_time"])
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    active = json.loads(POINTER.read_text(encoding="utf-8"))["active_version"]
    versions = [f"v{n}" for n in range(23, int(active[1:]) + 1)]

    ladder = []
    for version in versions:
        path = _artifact(version)
        if not path.is_file():
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        post = _weighted(frame, frame["contest_votes"])
        ante = _weighted(frame, ftw.build(frame))
        ladder.append(
            {
                "version": version,
                "active": version == active,
                "rows": int(len(frame)),
                "contest_votes_regional_pp": post[0],
                "contest_votes_national_pp": post[1],
                "forecast_time_regional_pp": ante[0],
                "forecast_time_national_pp": ante[1],
                "winner_accuracy": ante[2],
                "winner_accuracy_unchanged_by_weighting": ante[2] == post[2],
            }
        )

    detail = by_election(
        pd.read_csv(_artifact(active), encoding="utf-8-sig", low_memory=False)
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(ladder)
    frame.to_csv(OUTPUT_DIR / "sensitivity_by_version.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUTPUT_DIR / "active_by_election.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "sensitivity_summary.json").write_bytes(
        (
            json.dumps(
                {
                    "schema": "metric_weighting_sensitivity_v1",
                    "active_version": active,
                    "reported_headline_weighting": "contest_votes",
                    "alternative_weighting": "forecast_time_region_weight",
                    "note": (
                        "contest_votes is the reported headline and is not a leak: it is "
                        "the definition of national vote share, and the model predicts "
                        "shares rather than turnout. The alternative column measures how "
                        "far the figures move when the aggregation weight is swapped."
                    ),
                    "post_2022_outcomes_used": False,
                    "by_version": ladder,
                    "active_by_election": detail.to_dict(orient="records"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    )
    print(frame.to_string(index=False))
    print()
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
