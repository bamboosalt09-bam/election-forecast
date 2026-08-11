"""RSS collector."""

from __future__ import annotations

from news_analyzer.collect import make_content_hash, now_utc, parse_date, stable_article_id, strip_html
from news_analyzer.schemas import ArticleRaw


def collect_rss_articles(feed_urls: list[str]) -> list[ArticleRaw]:
    try:
        import feedparser
    except ImportError as exc:
        raise RuntimeError("feedparser is required for collect-rss. Install with: python -m pip install -e .") from exc

    rows: list[ArticleRaw] = []
    collected_at = now_utc()
    for url in feed_urls:
        feed = feedparser.parse(url)
        source_name = feed.feed.get("title", url)
        for entry in feed.entries:
            title = strip_html(entry.get("title")) or ""
            summary = strip_html(entry.get("summary"))
            link = entry.get("link") or url
            published_at = parse_date(entry.get("published") or entry.get("updated"), collected_at.date())
            rows.append(
                ArticleRaw(
                    article_id=stable_article_id("rss", link, title),
                    source_type="rss",
                    source_name=source_name,
                    url=link,
                    canonical_url=link,
                    title=title,
                    summary=summary,
                    body=None,
                    author=entry.get("author"),
                    section=None,
                    published_at=published_at,
                    collected_at=collected_at,
                    available_date=published_at,
                    query=None,
                    raw_payload=dict(entry),
                    content_hash=make_content_hash(title, summary),
                )
            )
    return rows
