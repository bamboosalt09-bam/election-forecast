"""Salience aggregator: crawled metadata → issue_store rows, schema-conforming."""

from __future__ import annotations

import pandas as pd

from news_analyzer.salience import aggregate_salience, ISSUE_EVENT_COLUMNS

from common.issue_store import validate_issue_frame, issue_columns, rollup_issue_features


def _articles() -> pd.DataFrame:
    # housing dominates; one ethics(scandal) article mentions slot-B candidate.
    return pd.DataFrame(
        [
            {"title": "부동산 집값 급등 논란", "summary": "주택 공급 대책", "available_date": "2022-02-21"},
            {"title": "전세 대책 발표", "summary": "부동산 안정", "available_date": "2022-02-22"},
            {"title": "후보 비리 의혹 수사", "summary": "김후보 연루 의혹", "available_date": "2022-03-01"},
        ]
    )


def _keyword_map():
    return {
        "housing": {"keywords": ["부동산", "집값", "전세", "주택"], "pos": ["안정", "공급"], "neg": ["급등"]},
        "corruption_integrity": {"keywords": ["비리", "의혹", "수사"], "pos": ["무혐의"], "neg": ["의혹", "수사"]},
    }


def test_output_columns_match_issue_store_schema() -> None:
    # The inline column list must stay identical to common.issue_store.IssueEventRow.
    assert ISSUE_EVENT_COLUMNS == issue_columns()


def test_aggregate_salience_conforms_and_normalizes() -> None:
    out = aggregate_salience(
        _articles(),
        _keyword_map(),
        slot_names={"B": ["김후보"]},
        election_id="pres_2022",
        taxonomy_type={"housing": "policy", "corruption_integrity": "scandal"},
    )
    assert not out.empty
    # conforms to the shared contract (this is the whole point of the bridge)
    validate_issue_frame(out)
    # salience is min-max normalized into [0, 1]; max bucket hits 1.0
    assert out["salience_score"].between(0, 1).all()
    assert out["salience_score"].max() == 1.0
    # scandal routed type + slot co-mention captured
    scandal = out.loc[out["issue_name"] == "corruption_integrity"]
    assert (scandal["issue_type"] == "scandal").all()
    assert (scandal["slot"] == "B").any()  # 김후보 co-mention → slot B


def test_bridge_feeds_rollup() -> None:
    out = aggregate_salience(
        _articles(), _keyword_map(), slot_names={"B": ["김후보"]}, election_id="pres_2022",
        taxonomy_type={"housing": "policy", "corruption_integrity": "scandal"},
    )
    sens = pd.DataFrame(
        [
            {"region_id": "sido_11", "issue_name": "housing", "sensitivity_score": 0.7},
            {"region_id": "sido_11", "issue_name": "corruption_integrity", "sensitivity_score": 0.5},
        ]
    )
    feats = rollup_issue_features(out, sens, forecast_date="2022-03-08")
    assert not feats.empty
    assert feats["variable_value"].between(-1, 1).all()
