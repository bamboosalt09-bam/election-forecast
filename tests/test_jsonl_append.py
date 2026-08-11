from news_collector.schemas import RawArticle
from news_collector.storage import append_jsonl, read_jsonl


def article(article_id):
    return RawArticle(
        article_id=article_id,
        source_type="local_file",
        source_name="source",
        url=f"https://example.com/{article_id}",
        title=f"Title {article_id}",
        published_at="2021-01-01",
        content_hash=f"content-{article_id}",
        title_hash=f"title-{article_id}",
        collection_batch_id="batch",
    )


def test_append_jsonl_does_not_overwrite_existing_file(tmp_path):
    path = tmp_path / "articles.jsonl"

    append_jsonl(path, [article("a1")])
    append_jsonl(path, [article("a2")])

    rows = read_jsonl(path)
    assert [row["article_id"] for row in rows] == ["a1", "a2"]
