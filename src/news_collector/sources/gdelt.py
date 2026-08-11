"""GDELT DOC API collector."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
import uuid

from bs4 import BeautifulSoup
from dateutil import parser as date_parser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from news_collector.checkpoint import CheckpointStore, checkpoint_item_id
from news_collector.dedupe import DedupeStore, compute_content_hash, compute_title_hash, extract_domain, stable_article_id
from news_collector.schemas import RawArticle
from news_collector.sources.base import CollectionStats, get_logger
from news_collector.sources.naver_news import read_query_plan
from news_collector.storage import write_articles_partitioned


GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
logger = get_logger(__name__)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = BeautifulSoup(str(value), "html.parser").get_text(" ")
    text = " ".join(text.split())
    return text or None


def _fmt_dt(value: date, end_of_day: bool = False) -> str:
    dt = datetime.combine(value, time.max if end_of_day else time.min)
    return dt.strftime("%Y%m%d%H%M%S")


def _parse_date(value: Any):
    if not value:
        return None
    try:
        return date_parser.parse(str(value)).date()
    except (TypeError, ValueError, OverflowError):
        return None


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(3), reraise=True)
def _request(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
    response = client.get(GDELT_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _article_from_gdelt(
    item: dict[str, Any],
    query: str,
    batch_id: str,
    window_start: date,
    window_end: date,
) -> RawArticle:
    title = _clean(item.get("title")) or "(untitled)"
    summary = _clean(item.get("seendate"))
    url = _clean(item.get("url"))
    source_domain = _clean(item.get("domain")) or extract_domain(url)
    source_name = source_domain or "GDELT"
    published_at = _parse_date(item.get("seendate"))
    content_hash = compute_content_hash(title, summary, None)
    return RawArticle(
        article_id=stable_article_id("gdelt", source_name, url, title),
        source_type="gdelt",
        provider="gdelt",
        source_name=source_name,
        source_domain=source_domain,
        raw_source_id=url,
        url=url,
        canonical_url=url,
        title=title,
        summary=summary,
        body=None,
        section=_clean(item.get("sourcecountry")),
        author=None,
        published_at=published_at,
        collected_at=datetime.now(timezone.utc),
        available_date=published_at,
        query=query,
        raw_payload=item,
        content_hash=content_hash,
        title_hash=compute_title_hash(title),
        language=_clean(item.get("language")) or None,
        collection_batch_id=batch_id,
        collection_window_start=window_start,
        collection_window_end=window_end,
    )


def collect_gdelt(
    queries_path: str | Path,
    out_dir: str | Path,
    start_date: date,
    end_date: date,
    seen_db_path: str | Path = "data/cache/seen_urls.sqlite",
    checkpoint_db_path: str | Path = "data/cache/checkpoints.sqlite",
    max_records: int = 250,
    resume: bool = True,
) -> CollectionStats:
    """Collect GDELT DOC API articles."""
    store = DedupeStore(seen_db_path)
    checkpoints = CheckpointStore(checkpoint_db_path)
    batch_id = f"gdelt-{uuid.uuid4().hex}"
    collected = written = skipped = failed = 0
    with httpx.Client() as client:
        for row in read_query_plan(queries_path):
            query_id = row.get("query_id") or row["query"]
            row_start = date.fromisoformat(row.get("window_start") or row.get("start_date") or start_date.isoformat())
            row_end = date.fromisoformat(row.get("window_end") or row.get("end_date") or end_date.isoformat())
            scoped_id = checkpoint_item_id(query_id, row_start.isoformat(), row_end.isoformat())
            if resume and checkpoints.get("gdelt", scoped_id) and checkpoints.get("gdelt", scoped_id).status == "complete":
                continue
            try:
                params = {
                    "query": row["query"],
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": max_records,
                    "startdatetime": _fmt_dt(row_start),
                    "enddatetime": _fmt_dt(row_end, end_of_day=True),
                }
                payload = _request(client, params)
                articles = [_article_from_gdelt(item, row["query"], batch_id, row_start, row_end) for item in payload.get("articles", [])]
                fresh, duplicate_count = store.filter_new(articles)
                results = write_articles_partitioned(out_dir, fresh, "gdelt", year_subdir=True)
                output_file = str(results[-1].file_path) if results else None
                checkpoints.update(
                    "gdelt",
                    scoped_id,
                    len(articles),
                    status="complete",
                    output_file=output_file,
                    collection_batch_id=batch_id,
                )
                collected += len(articles)
                written += sum(result.article_count for result in results)
                skipped += duplicate_count
            except Exception as exc:  # noqa: BLE001 - source failure must not stop the full run.
                failed += 1
                prior = checkpoints.get("gdelt", scoped_id)
                checkpoints.mark_error("gdelt", scoped_id, (prior.error_count if prior else 0) + 1)
                logger.exception("gdelt query failed: %s", exc)
    return CollectionStats(collected=collected, written=written, skipped_duplicate=skipped, failed=failed)
