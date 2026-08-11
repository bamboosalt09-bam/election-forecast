"""BIGKinds article-count → issue salience (covers 2002+, incl. pre-DataLab era).

Why BIGKinds: Naver DataLab search-volume only goes back to 2016-01-01, so the
16·17·18대 (2002·2007·2012) elections need a different instrument. BIGKinds
(한국언론진흥재단 뉴스 빅데이터) indexes major outlets from ~1990, and its
"기간별 기사량(trend)" gives an article-count time series per keyword — a salience
signal available for the entire 2002–2025 panel. It is therefore both the
*historical* source and the *consistent* instrument for all six elections.

How it is processed (answers "무슨 방법으로 처리했나"):
1. The user runs a BIGKinds 뉴스분석 → 기간별 트렌드 search for each issue's
   keywords (from ``issue_keywords.csv``) and exports the count time series.
2. That export is mapped into a long CSV: ``issue_name, period, count``.
3. This module min-max normalizes counts *within the election* (per
   ``salience_base``) and tags every row ``instrument="bigkinds_count"``.

We do NOT store article bodies — only the aggregated counts (copyright-safe,
matching the project's metadata-only policy).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from news_collector.sources.salience_base import normalize_within_election

REQUIRED_COLUMNS = {"issue_name", "period", "count"}


def load_bigkinds_counts(path: str | Path) -> pd.DataFrame:
    """Read a long-format BIGKinds count export: ``issue_name, period, count``.

    Accepts a ``date`` column as an alias for ``period``. Raises a clear error if
    the export was not mapped into the expected shape.
    """

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "period" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "period"})
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"BIGKinds counts file is missing columns {missing}. "
            "Map your export to columns: issue_name, period, count."
        )
    return df[["issue_name", "period", "count"]]


def counts_to_salience(counts: pd.DataFrame, election_id: str) -> pd.DataFrame:
    """Normalize BIGKinds counts → salience [0,1] within the election."""

    return normalize_within_election(counts, "count", election_id, instrument="bigkinds_count")


def import_bigkinds_salience(path: str | Path, election_id: str) -> pd.DataFrame:
    """Load a BIGKinds count export and return canonical salience rows."""

    return counts_to_salience(load_bigkinds_counts(path), election_id)
