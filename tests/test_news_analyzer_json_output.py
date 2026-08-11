from datetime import date

from news_analyzer.analyze import analysis_from_ai_payload
from news_analyzer.schemas import ArticleCleaned


def test_json_parse_failure_sets_error_and_human_review() -> None:
    article = ArticleCleaned(
        article_id="a1",
        url="https://example.com/a1",
        source_name="fixture",
        title="기사",
        published_at=date(2022, 2, 1),
        available_date=date(2022, 2, 1),
        content_hash="h1",
    )

    analysis = analysis_from_ai_payload(article, "not json")

    assert analysis.needs_human_review is True
    assert analysis.error is not None
