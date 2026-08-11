"""Multi-source salience: BIGKinds import + provenance + cross-instrument combine."""

from __future__ import annotations

import pandas as pd
import pytest

from news_collector.sources.bigkinds_salience import counts_to_salience, load_bigkinds_counts
from news_collector.sources.salience_base import CANONICAL_COLUMNS, combine_salience, normalize_within_election


def test_normalize_within_election_minmax_and_provenance() -> None:
    rows = pd.DataFrame(
        [
            {"issue_name": "housing", "period": "2012-11-01", "v": 50},
            {"issue_name": "housing", "period": "2012-11-08", "v": 100},
            {"issue_name": "security_nk", "period": "2012-11-01", "v": 25},
        ]
    )
    out = normalize_within_election(rows, "v", "pres_2012", "bigkinds_count")
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out["salience_score"].max() == 1.0  # peak (housing 100) → 1.0
    assert (out["instrument"] == "bigkinds_count").all()
    assert (out["election_id"] == "pres_2012").all()


def test_bigkinds_counts_to_salience_covers_pre_datalab() -> None:
    counts = pd.DataFrame(
        [
            {"issue_name": "regional_dev", "period": "2002-11-01", "count": 200},  # 2002 = 16대
            {"issue_name": "regional_dev", "period": "2002-11-15", "count": 400},
            {"issue_name": "economy_growth", "period": "2002-11-01", "count": 100},
        ]
    )
    sal = counts_to_salience(counts, "pres_2002")
    assert sal["salience_score"].between(0, 1).all()
    assert sal.loc[sal["issue_name"] == "economy_growth", "salience_score"].iloc[0] == 0.25  # 100/400


def test_load_bigkinds_counts_requires_mapped_columns(tmp_path) -> None:
    bad = tmp_path / "raw.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_bigkinds_counts(bad)


def test_load_bigkinds_counts_accepts_date_alias(tmp_path) -> None:
    f = tmp_path / "counts.csv"
    f.write_text("issue_name,date,count\nhousing,2012-11-01,10\n", encoding="utf-8-sig")
    df = load_bigkinds_counts(f)
    assert list(df.columns) == ["issue_name", "period", "count"]


def test_combine_salience_keeps_both_instruments() -> None:
    a = normalize_within_election(
        pd.DataFrame([{"issue_name": "x", "period": "p", "v": 1}]), "v", "pres_2012", "bigkinds_count"
    )
    b = normalize_within_election(
        pd.DataFrame([{"issue_name": "x", "period": "p", "v": 1}]), "v", "pres_2022", "datalab_search"
    )
    combined = combine_salience([a, b])
    assert set(combined["instrument"]) == {"bigkinds_count", "datalab_search"}
    assert len(combined) == 2
