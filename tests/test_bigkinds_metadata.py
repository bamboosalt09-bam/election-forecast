"""BIGKinds metadata importer → salience + candidate-issue (offline)."""

from __future__ import annotations

import pandas as pd

from news_collector.sources.bigkinds_metadata import (
    load_bigkinds_metadata,
    metadata_to_candidate_issue,
    metadata_to_salience,
)

KW = {"housing": ["부동산", "집값"], "security_nk": ["대북", "북핵"]}

META = pd.DataFrame(
    [
        {"date": pd.Timestamp("2012-11-05"), "title": "부동산 공약 발표", "keywords": "집값,규제", "persons": "후보갑"},
        {"date": pd.Timestamp("2012-11-06"), "title": "집값 논쟁 가열", "keywords": "부동산", "persons": "후보갑,후보을"},
        {"date": pd.Timestamp("2012-11-07"), "title": "대북 정책 토론", "keywords": "북핵,안보", "persons": "후보을"},
    ]
)


def test_metadata_to_salience_counts_and_normalizes() -> None:
    sal = metadata_to_salience(META, KW, "pres_2012")
    assert set(sal["issue_name"]) == {"housing", "security_nk"}
    assert sal["salience_score"].max() == 1.0
    assert (sal["instrument"] == "bigkinds_meta").all()  # provenance tagged
    assert (sal["election_id"] == "pres_2012").all()


def test_metadata_to_candidate_issue_links_persons_to_slots() -> None:
    names = {"후보갑": "A", "후보을": "B"}
    out = metadata_to_candidate_issue(META, KW, names)
    # 후보갑 ↔ housing appears twice (both housing articles mention 후보갑)
    gap = out[(out["slot"] == "A") & (out["issue_name"] == "housing")]
    assert int(gap["cooccurrence"].iloc[0]) == 2
    # 후보을 ↔ security_nk linked
    assert ((out["slot"] == "B") & (out["issue_name"] == "security_nk")).any()


def test_load_bigkinds_metadata_reads_korean_columns(tmp_path) -> None:
    f = tmp_path / "bk.csv"
    f.write_text("일자,제목,키워드,인물\n20121105,부동산 공약,집값,후보갑\n", encoding="utf-8-sig")
    df = load_bigkinds_metadata(f)
    assert list(df.columns) == ["date", "title", "keywords", "persons"]
    assert df.loc[0, "date"].year == 2012


def test_load_bigkinds_metadata_requires_core_columns(tmp_path) -> None:
    f = tmp_path / "bad.csv"
    f.write_text("foo,bar\n1,2\n", encoding="utf-8-sig")
    try:
        load_bigkinds_metadata(f)
        assert False, "should have raised"
    except ValueError as exc:
        assert "date/title" in str(exc)
