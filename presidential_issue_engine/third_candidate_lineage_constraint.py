"""One-sided ceiling on third candidates without major-party split lineage.

Rationale
---------
Across the V24 scored panel the third candidate splits cleanly by whether the
candidate's vehicle descends from a large-scale split of a governing or main
opposition party:

    major-split lineage   이회창 2007 15.1%   안철수 2017 21.4%
    self-founded / minor  권영길 2002  3.9%   강지원 2012 0.2%   심상정 2022 2.4%

The engine's base stage does not carry this distinction, so a self-founded
third candidate can be assigned a level drawn from the strong third candidates
the model has seen. This module therefore caps such a candidate at the direct
party evidence its own bloc has actually accumulated, and redistributes the
excess to the two majors in proportion to their predicted shares.

The rule is one-sided: it can only lower a third candidate that the model has
placed above its own party base, and it is inert everywhere else. The ceiling
is the bloc's own ``direct_party_recent_base`` at a factor of exactly one, so
the module introduces no fitted parameter.

Lineage is a documented pre-election fact recorded in
``fixed_dataset/v24/third_candidate_lineage.csv``; no election outcome is read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINEAGE_TABLE = ROOT / "presidential_issue_engine" / "fixed_dataset" / "v24" / "third_candidate_lineage.csv"

_TRUE = {"1", "true", "yes", "y"}


def load_lineage(path: Path | str | None = None) -> pd.DataFrame:
    """Return the declared third-candidate lineage table."""

    source = Path(path) if path is not None else LINEAGE_TABLE
    frame = pd.read_csv(source, encoding="utf-8-sig")
    required = {"election_id", "candidate_name", "major_split_lineage", "available_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"third_candidate_lineage is missing columns: {missing}")
    frame["major_split_lineage"] = (
        frame["major_split_lineage"].astype(str).str.strip().str.lower().isin(_TRUE)
    )
    return frame


def self_founded_elections(lineage: pd.DataFrame) -> set[str]:
    """Elections whose third candidate lacks major-party split lineage."""

    weak = lineage.loc[~lineage["major_split_lineage"]]
    return set(weak["election_id"].astype(str))


def apply_lineage_ceiling(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "layer_pred",
    output_column: str = "layer_pred",
    lineage: pd.DataFrame | None = None,
    ceiling_column: str = "direct_party_recent_base",
    slot_column: str = "slot",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap self-founded third candidates at their own direct party evidence.

    Returns the adjusted frame and a per-region audit of every applied cap.
    """

    lineage = load_lineage() if lineage is None else lineage
    weak = self_founded_elections(lineage)
    out = frame.copy()
    out[output_column] = pd.to_numeric(out[prediction_column], errors="coerce")
    audit_rows: list[dict[str, object]] = []

    for election_id, election_rows in out.groupby("election_id", sort=False):
        if str(election_id) not in weak:
            continue
        for region_id, region in election_rows.groupby("region_id", sort=False):
            third = region.loc[region[slot_column].astype(str).eq("C")]
            majors = region.loc[region[slot_column].astype(str).isin(["A", "B"])]
            if third.empty or majors.empty:
                continue
            index = third.index[0]
            current = float(out.at[index, output_column])
            ceiling = float(pd.to_numeric(third[ceiling_column], errors="coerce").iloc[0])
            if not (ceiling > 0.0) or current <= ceiling:
                continue
            excess = current - ceiling
            weights = out.loc[majors.index, output_column]
            total = float(weights.sum())
            if total <= 0.0:
                continue
            out.at[index, output_column] = ceiling
            out.loc[majors.index, output_column] = weights + excess * (weights / total)
            audit_rows.append(
                {
                    "election_id": str(election_id),
                    "region_id": str(region_id),
                    "candidate_name": str(third.get("candidate_name", pd.Series([""])).iloc[0]),
                    "before": current,
                    "ceiling": ceiling,
                    "excess_redistributed": excess,
                }
            )

    group = out.groupby(["election_id", "region_id"])[output_column]
    out[output_column] = out[output_column] / group.transform("sum")
    return out, pd.DataFrame(audit_rows)
