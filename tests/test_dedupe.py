from news_collector.dedupe import DedupeStore, compute_content_hash, compute_title_hash
from news_collector.schemas import RawArticle


def make_article(url, canonical_url=None, title="Same title", body="Body"):
    return RawArticle(
        article_id=url,
        source_type="local_file",
        source_name="source",
        url=url,
        canonical_url=canonical_url,
        title=title,
        published_at="2021-01-01",
        content_hash=compute_content_hash(title, None, body),
        title_hash=compute_title_hash(title),
        collection_batch_id="batch",
    )


def test_dedupe_by_url_canonical_and_content_hash(tmp_path):
    store = DedupeStore(tmp_path / "seen.sqlite")
    first = make_article("https://example.com/a?utm_source=x")
    same_url = make_article("https://example.com/a")
    same_canonical = make_article("https://other.com/a", canonical_url="https://example.com/a")
    same_content = make_article("https://another.com/a")

    assert not store.is_seen(first)
    store.mark_seen(first)

    assert store.is_seen(same_url)
    assert store.is_seen(same_canonical)
    assert store.is_seen(same_content)
