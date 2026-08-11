from news_collector.schemas import RawArticle


def test_raw_article_defaults_available_date_from_published_at():
    article = RawArticle(
        article_id="a1",
        source_type="local_file",
        source_name="test",
        url="https://example.com/a",
        title="Title",
        published_at="2021-01-02",
        content_hash="c",
        title_hash="t",
        collection_batch_id="b1",
    )

    assert article.published_at.isoformat() == "2021-01-02"
    assert article.available_date.isoformat() == "2021-01-02"


def test_raw_article_allows_missing_dates():
    article = RawArticle(
        article_id="a2",
        source_type="local_file",
        source_name="test",
        url="https://example.com/b",
        title="Title",
        content_hash="c",
        title_hash="t",
        collection_batch_id="b1",
    )

    assert article.published_at is None
    assert article.available_date is None
