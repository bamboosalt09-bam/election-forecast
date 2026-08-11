"""RSS feed collector."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import uuid

from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import feedparser

from news_collector.checkpoint import CheckpointStore
from news_collector.dedupe import DedupeStore, compute_content_hash, compute_title_hash, extract_domain, stable_article_id
from news_collector.schemas import RawArticle
from news_collector.sources.base import CollectionStats, get_logger
from news_collector.storage import write_articles_partitioned


logger = get_logger(__name__)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = BeautifulSoup(str(value), "html.parser").get_text(" ")
    text = " ".join(text.split())
    return text or None


def _parse_date(value: Any):
    if not value:
        return None
    try:
        return date_parser.parse(str(value)).date()
    except (TypeError, ValueError, OverflowError):
        return None


def read_feeds(path: str | Path) -> list[dict[str, str]]:
    """Read feeds.csv with feed_id,rss_url,source_name style columns."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def articles_from_feed(feed_row: dict[str, str], collection_batch_id: str | None = None) -> list[RawArticle]:
    """Fetch one RSS feed and convert entries to RawArticle records."""
    feed_url = feed_row.get("rss_url") or feed_row.get("feed_url") or feed_row.get("url")
    if not feed_url:
        raise ValueError("feed row must include rss_url, feed_url, or url")
    parsed = feedparser.parse(feed_url)
    source_name = feed_row.get("source_name") or parsed.feed.get("title") or feed_url
    batch_id = collection_batch_id or f"rss-{uuid.uuid4().hex}"
    collected_at = datetime.now(timezone.utc)
    articles: list[RawArticle] = []
    for entry in parsed.entries:
        title = _clean(entry.get("title"))
        if not title:
            continue
        summary = _clean(entry.get("summary") or entry.get("description"))
        body = _clean(entry.get("content", [{}])[0].get("value") if entry.get("content") else None)
        url = _clean(entry.get("link"))
        source_domain = extract_domain(url)
        published_at = _parse_date(entry.get("published") or entry.get("updated"))
        content_hash = compute_content_hash(title, summary, body)
        title_hash = compute_title_hash(title)
        articles.append(
            RawArticle(
                article_id=stable_article_id("rss", str(source_name), url, title),
                source_type="rss",
                provider="rss",
                source_name=str(source_name),
                source_domain=source_domain,
                raw_source_id=_clean(entry.get("id") or entry.get("guid") or entry.get("link")),
                url=url,
                canonical_url=url,
                title=title,
                summary=summary,
                body=body,
                section=",".join(tag.get("term", "") for tag in entry.get("tags", []) if tag.get("term")) or None,
                author=_clean(entry.get("author")),
                published_at=published_at,
                collected_at=collected_at,
                available_date=published_at,
                query=None,
                raw_payload=dict(entry),
                content_hash=content_hash,
                title_hash=title_hash,
                language=feed_row.get("language") or "ko",
                collection_batch_id=batch_id,
            )
        )
    return articles


def collect_rss(
    feeds_path: str | Path,
    out_dir: str | Path,
    seen_db_path: str | Path = "data/cache/seen_urls.sqlite",
    checkpoint_db_path: str | Path = "data/cache/checkpoints.sqlite",
    resume: bool = True,
) -> CollectionStats:
    """Collect all feeds while isolating failures per feed."""
    store = DedupeStore(seen_db_path)
    checkpoints = CheckpointStore(checkpoint_db_path)
    stats = CollectionStats()
    collected = written = skipped = failed = 0
    batch_id = f"rss-{uuid.uuid4().hex}"
    for feed in read_feeds(feeds_path):
        feed_id = feed.get("feed_id") or feed.get("source_name") or feed.get("rss_url") or feed.get("url") or "unknown"
        if resume and checkpoints.get("rss", feed_id) and checkpoints.get("rss", feed_id).status == "success":
            continue
        try:
            articles = articles_from_feed(feed, batch_id)
            fresh, duplicate_count = store.filter_new(articles)
            results = write_articles_partitioned(out_dir, fresh, "rss", year_subdir=True)
            output_file = str(results[-1].file_path) if results else None
            checkpoints.update("rss", feed_id, len(articles), output_file=output_file, collection_batch_id=batch_id)
            collected += len(articles)
            written += sum(result.article_count for result in results)
            skipped += duplicate_count
        except Exception as exc:  # noqa: BLE001 - source failure must not stop the full run.
            failed += 1
            prior = checkpoints.get("rss", feed_id)
            checkpoints.mark_error("rss", feed_id, (prior.error_count if prior else 0) + 1)
            logger.exception("rss feed failed: %s", exc)
    return CollectionStats(collected=collected, written=written, skipped_duplicate=skipped, failed=failed)
