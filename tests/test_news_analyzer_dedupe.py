from datetime import date, datetime, timezone

from news_analyzer.collect import make_content_hash
from news_analyzer.dedupe import dedupe_raw_records


def _raw(article_id: str, url: str, title: str, content_hash: str | None = None) -> dict:
    return {
        "article_id": article_id,
        "source_type": "local_file",
        "source_name": "fixture",
        "url": url,
        "canonical_url": None,
        "title": title,
        "summary": None,
        "body": None,
        "author": None,
        "section": None,
        "published_at": date(2022, 2, 1).isoformat(),
        "collected_at": datetime(2022, 2, 1, tzinfo=timezone.utc).isoformat(),
        "available_date": date(2022, 2, 1).isoformat(),
        "query": None,
        "raw_payload": None,
        "content_hash": content_hash or make_content_hash(title),
    }


def test_dedupe_removes_same_url_and_content_hash() -> None:
    rows = [
        _raw("a1", "https://example.com/news?a=1&utm_source=x", "같은 기사"),
        _raw("a2", "https://example.com/news?a=1", "다른 제목"),
        _raw("a3", "https://example.com/other", "또 다른 제목", content_hash=make_content_hash("같은 기사")),
    ]

    cleaned = dedupe_raw_records(rows)

    assert [row.article_id for row in cleaned] == ["a1"]
