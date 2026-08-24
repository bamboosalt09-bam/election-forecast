"""Regional weights a forecaster actually holds on the eve of an election.

The V27 and V29 postprocess transforms weight each candidate's national level by
``contest_votes`` - the *target* election's own regional turnout. That number
exists only after the votes are counted, so a transform using it consumes an
outcome of the election it is predicting. The prospective 2025 path already
avoids it, substituting the previous election's volumes, which is itself the
admission that it is not forecast-time information.

This module supplies the substitution for the scored panel as well, so both
paths weight the same way and the historical figures describe something a
forecast could actually have produced.

The rule:

* an election with a predecessor in the panel uses that predecessor's regional
  volumes;
* a region absent from the predecessor - 세종 first appears in 2012 - takes the
  predecessor's mean regional volume rather than being dropped;
* 2002's predecessor, 1997, is a warmup election outside the scored panel, so
  its regional turnout is carried separately in
  ``fixed_dataset/pres_1997_regional_turnout.csv``. Every scored election
  therefore uses the same rule; none falls back to equal regions.

The leak this closes was wide open and carried very little, which is the reason
to close it rather than a reason to leave it: the transform only ever used the
weight to locate each candidate's national level, and regional vote shares move
slowly between elections.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Chronological, so "previous" is well defined.
SCORED_ORDER = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
WEIGHT_COLUMN = "forecast_time_region_weight"
#: 1997 is the warmup predecessor of the first scored election. Its regional
#: turnout is not in the scored results table, so it is carried on its own.
#: Source: 국사편찬위원회 한국사데이터베이스, 제15대 대통령선거 (1997-12-18).
#: All three columns sum exactly to the published national totals - electorate
#: 32,290,416, votes cast 26,042,633, valid votes 25,642,438 - which is how the
#: transcription was checked.
WARMUP_TURNOUT = (
    Path(__file__).resolve().parent / "fixed_dataset" / "pres_1997_regional_turnout.csv"
)
WARMUP_PREDECESSOR = {"pres_2002": "pres_1997"}


def _regional_volumes(frame: pd.DataFrame, weight_source: str) -> pd.DataFrame:
    return (
        frame.groupby(["election_id", "region_id"], as_index=False)[weight_source]
        .first()
        .rename(columns={weight_source: "volume"})
    )


def _warmup_volumes(election: str | None) -> pd.Series | None:
    """Regional valid votes for a warmup election outside the scored panel."""

    if election is None or not WARMUP_TURNOUT.is_file():
        return None
    table = pd.read_csv(WARMUP_TURNOUT, encoding="utf-8-sig")
    rows = table.loc[table["election_id"].astype(str).eq(election)]
    if rows.empty:
        return None
    # valid votes, to match what contest_votes counts in the scored panel
    return rows.set_index("region_id")["valid_votes"].astype(float)


def build(frame: pd.DataFrame, weight_source: str = "contest_votes") -> pd.Series:
    """Forecast-time weight per row of ``frame``.

    ``weight_source`` names the column holding each election's own regional
    volume. Only *previous* elections' values are ever read from it; the target
    election's own volume never reaches the result.
    """

    missing = sorted({"election_id", "region_id", weight_source} - set(frame.columns))
    if missing:
        raise KeyError(f"forecast-time weights need {missing}")

    volumes = _regional_volumes(frame, weight_source)
    lookup = {
        str(election): group.set_index("region_id")["volume"]
        for election, group in volumes.groupby("election_id")
    }

    weights: dict[str, dict[str, float]] = {}
    for index, election in enumerate(SCORED_ORDER):
        if election not in lookup:
            continue
        regions = list(lookup[election].index)
        if index == 0:
            warmup = _warmup_volumes(WARMUP_PREDECESSOR.get(election))
            if warmup is not None and not warmup.empty:
                fallback = float(warmup.mean())
                weights[election] = {
                    region: float(warmup[region]) if region in warmup.index else fallback
                    for region in regions
                }
                continue
            # only if the warmup table is unavailable: every region counts once
            weights[election] = {region: 1.0 for region in regions}
            continue
        previous = lookup.get(SCORED_ORDER[index - 1])
        if previous is None or previous.empty:
            weights[election] = {region: 1.0 for region in regions}
            continue
        fallback = float(previous.mean())
        weights[election] = {
            region: float(previous[region]) if region in previous.index else fallback
            for region in regions
        }

    unknown = sorted(set(frame["election_id"].astype(str)) - set(weights))
    if unknown:
        raise ValueError(f"no forecast-time weight rule for {unknown}")

    return pd.Series(
        [
            weights[str(election)][str(region)]
            for election, region in zip(frame["election_id"], frame["region_id"])
        ],
        index=frame.index,
        dtype=float,
        name=WEIGHT_COLUMN,
    )


def attach(frame: pd.DataFrame, weight_source: str = "contest_votes") -> pd.DataFrame:
    """Return ``frame`` with the forecast-time weight column added."""

    out = frame.copy()
    out[WEIGHT_COLUMN] = build(out, weight_source)
    return out
