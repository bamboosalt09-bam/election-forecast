"""Naver News Search API collector."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
import csv
import os
import time
import uuid

from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from news_collector.checkpoint import CheckpointStore, checkpoint_item_id
from news_collector.config import require_naver_credentials
from news_collector.dedupe import DedupeStore, compute_content_hash, compute_title_hash, extract_domain, stable_article_id
from news_collector.schemas import RawArticle
from news_collector.sources.base import CollectionStats, get_logger
from news_collector.storage import write_articles_partitioned


NAVER_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"
logger = get_logger(__name__)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = BeautifulSoup(str(value), "html.parser").get_text(" ")
    text = " ".join(text.split())
    return text or None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date_parser.parse(str(value)).date()
    except (TypeError, ValueError, OverflowError):
        return None


def read_query_plan(path: str | Path) -> list[dict[str, str]]:
    """Read enabled query_plan.csv rows."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if str(row.get("enabled", "true")).lower() in {"true", "1", "yes", "y"}]


def _row_window(row: dict[str, str], fallback_start: date | None, fallback_end: date | None) -> tuple[date | None, date | None]:
    """Return a row-specific collection window, accepting old and new column names."""
    start_value = row.get("window_start") or row.get("start_date")
    end_value = row.get("window_end") or row.get("end_date")
    row_start = date.fromisoformat(start_value) if start_value else fallback_start
    row_end = date.fromisoformat(end_value) if end_value else fallback_end
    return row_start, row_end


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3), reraise=True)
def _request(client: httpx.Client, headers: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    response = client.get(NAVER_ENDPOINT, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _article_from_item(
    item: dict[str, Any],
    query: str,
    batch_id: str,
    window_start: date | None,
    window_end: date | None,
) -> RawArticle:
    title = _clean(item.get("title")) or "(untitled)"
    summary = _clean(item.get("description"))
    url = _clean(item.get("link"))
    canonical_url = _clean(item.get("originallink")) or url
    source_domain = extract_domain(canonical_url or url)
    published_at = _parse_date(item.get("pubDate"))
    content_hash = compute_content_hash(title, summary, None)
    title_hash = compute_title_hash(title)
    source_name = source_domain or "unknown_naver_source"
    return RawArticle(
        article_id=stable_article_id("naver_api", source_name, canonical_url or url, title),
        source_type="naver_api",
        provider="naver_api",
        source_name=source_name,
        source_domain=source_domain,
        raw_source_id=_clean(item.get("link")),
        url=url,
        canonical_url=canonical_url,
        title=title,
        summary=summary,
        body=None,
        section=None,
        author=None,
        published_at=published_at,
        collected_at=datetime.now(timezone.utc),
        available_date=published_at,
        query=query,
        raw_payload=item,
        content_hash=content_hash,
        title_hash=title_hash,
        language="ko",
        collection_batch_id=batch_id,
        collection_window_start=window_start,
        collection_window_end=window_end,
    )


def collect_naver(
    queries_path: str | Path,
    out_dir: str | Path,
    start_date: date | None = None,
    end_date: date | None = None,
    seen_db_path: str | Path = "data/cache/seen_urls.sqlite",
    checkpoint_db_path: str | Path = "data/cache/checkpoints.sqlite",
    display: int = 100,
    start: int = 1,
    sort: str = "date",
    max_pages: int = 10,
    rate_limit_seconds: float = 0.25,
    resume: bool = True,
) -> CollectionStats:
    """Collect Naver API search results and append new rows to raw_lake."""
    if not os.getenv("NAVER_CLIENT_ID") or not os.getenv("NAVER_CLIENT_SECRET"):
        raise RuntimeError("Naver collection skipped: NAVER_CLIENT_ID and NAVER_CLIENT_SECRET are not set")
    client_id, client_secret = require_naver_credentials()
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    store = DedupeStore(seen_db_path)
    checkpoints = CheckpointStore(checkpoint_db_path)
    batch_id = f"naver-{uuid.uuid4().hex}"
    collected = written = skipped = failed = 0

    with httpx.Client() as client:
        for row in read_query_plan(queries_path):
            query_id = row.get("query_id") or row["query"]
            query = row["query"]
            window_start, window_end = _row_window(row, start_date, end_date)
            scoped_id = checkpoint_item_id(
                query_id,
                window_start.isoformat() if window_start else None,
                window_end.isoformat() if window_end else None,
            )
            offset = start
            existing = checkpoints.get("naver_api", scoped_id)
            if resume and existing and existing.status == "complete":
                continue
            if resume and existing and existing.status == "success":
                offset = max(existing.page_offset + display, start)
            try:
                for _ in range(max_pages):
                    params = {"query": query, "display": display, "start": offset, "sort": sort}
                    payload = _request(client, headers, params)
                    items = payload.get("items", [])
                    articles = [_article_from_item(item, query, batch_id, window_start, window_end) for item in items]
                    if window_start or window_end:
                        articles = [
                            article
                            for article in articles
                            if article.published_at
                            and (window_start is None or article.published_at >= window_start)
                            and (window_end is None or article.published_at <= window_end)
                        ]
                    fresh, duplicate_count = store.filter_new(articles)
                    results = write_articles_partitioned(out_dir, fresh, "naver", year_subdir=True)
                    output_file = str(results[-1].file_path) if results else None
                    status = "complete" if len(items) < display else "success"
                    checkpoints.update(
                        "naver_api",
                        scoped_id,
                        offset,
                        status=status,
                        output_file=output_file,
                        collection_batch_id=batch_id,
                    )
                    collected += len(articles)
                    written += sum(result.article_count for result in results)
                    skipped += duplicate_count
                    if len(items) < display:
                        break
                    offset += display
                    time.sleep(rate_limit_seconds)
            except Exception as exc:  # noqa: BLE001 - source failure must not stop the full run.
                failed += 1
                prior = checkpoints.get("naver_api", scoped_id)
                checkpoints.mark_error("naver_api", scoped_id, (prior.error_count if prior else 0) + 1, offset)
                logger.exception("naver query failed: %s", exc)
    return CollectionStats(collected=collected, written=written, skipped_duplicate=skipped, failed=failed)
