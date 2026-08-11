from datetime import date

from news_analyzer.aggregate import aggregate_records


def _analysis(article_id: str, stance: float, link: float, issue: str = "경제") -> dict:
    return {
        "article_id": article_id,
        "published_at": "2022-02-01",
        "available_date": "2022-02-01",
        "source_name": "fixture",
        "article_type": "news",
        "candidates": ["홍길동"],
        "parties": [],
        "regions": [],
        "issues": [issue],
        "stance_by_candidate": {"홍길동": stance},
        "candidate_link_score": {"홍길동": link},
        "responsibility_target": [],
        "beneficiary": [],
        "harmed": [],
        "frame_tags": [],
        "region_relevance_score": {},
        "source_reliability_score": 1.0,
        "analysis_confidence": 1.0,
        "needs_human_review": False,
        "error": None,
        "content_hash": article_id,
    }


def test_aggregate_calculates_expected_final_issue_score() -> None:
    result = aggregate_records(
        [_analysis("a1", 0.5, 0.8), _analysis("a2", 1.0, 0.6)],
        date(2022, 2, 9),
        [30],
    )

    row = result[0]
    assert row.article_count == 2
    assert row.weighted_stance == 0.75
    assert row.avg_candidate_link_score == 0.7
    assert row.volume_z_score == 2.0
    assert row.final_issue_score == 1.05
