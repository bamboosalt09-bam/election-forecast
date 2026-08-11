from datetime import date

from news_analyzer.aggregate import aggregate_records


def test_aggregate_excludes_future_available_date() -> None:
    records = [
        {
            "article_id": "visible",
            "published_at": "2022-02-01",
            "available_date": "2022-02-01",
            "source_name": "fixture",
            "article_type": "news",
            "candidates": ["홍길동"],
            "parties": [],
            "regions": [],
            "issues": ["부동산"],
            "stance_by_candidate": {"홍길동": 0.5},
            "candidate_link_score": {"홍길동": 1.0},
            "responsibility_target": [],
            "beneficiary": [],
            "harmed": [],
            "frame_tags": [],
            "region_relevance_score": {},
            "source_reliability_score": 1.0,
            "analysis_confidence": 1.0,
            "needs_human_review": False,
            "error": None,
            "content_hash": "h1",
        },
        {
            "article_id": "future",
            "published_at": "2022-02-01",
            "available_date": "2022-02-10",
            "source_name": "fixture",
            "article_type": "news",
            "candidates": ["홍길동"],
            "parties": [],
            "regions": [],
            "issues": ["부동산"],
            "stance_by_candidate": {"홍길동": 1.0},
            "candidate_link_score": {"홍길동": 1.0},
            "responsibility_target": [],
            "beneficiary": [],
            "harmed": [],
            "frame_tags": [],
            "region_relevance_score": {},
            "source_reliability_score": 1.0,
            "analysis_confidence": 1.0,
            "needs_human_review": False,
            "error": None,
            "content_hash": "h2",
        },
    ]

    result = aggregate_records(records, date(2022, 2, 9), [30])

    assert len(result) == 1
    assert result[0].article_count == 1
    assert result[0].weighted_stance == 0.5
