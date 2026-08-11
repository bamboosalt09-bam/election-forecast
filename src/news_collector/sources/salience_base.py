"""Common salience contract — every source normalizes the same way, with provenance.

Issue salience can come from different *measurement instruments* depending on the
era:

- **DataLab search-volume** (네이버 데이터랩) — covers 2016-01-01 onward only.
- **BIGKinds article-count** (한국언론진흥재단) — covers ~1990 onward, so it is the
  consistent instrument for the WHOLE 2002–2025 panel and the only option for the
  16·17·18대 (2002·2007·2012) elections that predate DataLab.
- (extensible) GDELT article-count, etc.

Methodology (answers "왜 이렇게 처리했나"):
- Salience is a **within-election relative** measure. Each instrument's raw values
  are min-max normalized *within one election* → ``[0, 1]``. We therefore never
  claim cross-election absolute comparability; we compare salience to vote share
  *inside* each election.
- Every output row records its ``instrument`` (provenance). Mixing instruments
  across the panel is explicit and auditable, never silent.
- Recommended design: use **BIGKinds for all 6 elections** (one consistent
  instrument) and use **DataLab as an independent cross-check for 2016+** (two
  instruments agreeing strengthens validity).

Canonical output columns:
``election_id, issue_name, period, raw_value, salience_score, instrument``
"""

from __future__ import annotations

import pandas as pd

CANONICAL_COLUMNS = ["election_id", "issue_name", "period", "raw_value", "salience_score", "instrument"]


def normalize_within_election(
    rows: pd.DataFrame, value_col: str, election_id: str, instrument: str
) -> pd.DataFrame:
    """Min-max normalize ``value_col`` within one election → salience in [0, 1].

    ``rows`` must have ``issue_name`` and ``period`` columns. The result carries
    the ``instrument`` provenance tag so downstream steps can audit which
    measurement produced each salience value.
    """

    if rows.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = rows.copy()
    df["raw_value"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    peak = df["raw_value"].max()
    df["salience_score"] = (df["raw_value"] / peak).round(4) if peak > 0 else 0.0
    df["election_id"] = election_id
    df["instrument"] = instrument
    return df[CANONICAL_COLUMNS].sort_values(["issue_name", "period"]).reset_index(drop=True)


def combine_salience(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack salience frames from multiple instruments, keeping provenance."""

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)[CANONICAL_COLUMNS]
