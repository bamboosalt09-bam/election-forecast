from datetime import date, datetime, timezone

from news_analyzer.schemas import ArticleAnalysis, ArticleCleaned, ArticleRaw


def test_article_schemas_validate() -> None:
    raw = ArticleRaw(
        article_id="a1",
        source_type="local_file",
        source_name="fixture",
        url="https://example.com/a1",
        title="후보 부동산 공약",
        published_at=date(2022, 2, 1),
        collected_at=datetime(2022, 2, 1, tzinfo=timezone.utc),
        available_date=date(2022, 2, 1),
        content_hash="hash-a1",
    )
    cleaned = ArticleCleaned(
        article_id=raw.article_id,
        url=raw.url,
        source_name=raw.source_name,
        title=raw.title,
        published_at=raw.published_at,
        available_date=raw.available_date,
        article_type="news",
        content_hash=raw.content_hash,
    )
    analysis = ArticleAnalysis(
        article_id=cleaned.article_id,
        published_at=cleaned.published_at,
        available_date=cleaned.available_date,
        source_name=cleaned.source_name,
        article_type=cleaned.article_type,
        candidates=["홍길동"],
        issues=["부동산"],
        stance_by_candidate={"홍길동": 0.5},
        candidate_link_score={"홍길동": 0.8},
        analysis_confidence=0.9,
        source_reliability_score=0.7,
    )

    assert analysis.article_id == "a1"
    assert analysis.stance_by_candidate["홍길동"] == 0.5
