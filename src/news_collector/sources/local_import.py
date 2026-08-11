"""Import user-provided CSV, JSONL, and XLSX files into RawArticle JSONL."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from dateutil import parser as date_parser
import pandas as pd
import yaml

from news_collector.dedupe import DedupeStore, compute_content_hash, compute_title_hash, extract_domain, stable_article_id
from news_collector.schemas import RawArticle
from news_collector.sources.base import CollectionStats
from news_collector.storage import iter_jsonl, write_articles_partitioned


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
        text = BeautifulSoup(str(value), "html.parser").get_text(" ")
    text = " ".join(text.split())
    return text or None


def _parse_date(value: Any):
    if value is None or pd.isna(value) or value == "":
        return None
    return date_parser.parse(str(value)).date()


def load_mapping(path: str | Path) -> dict[str, str]:
    """Load a YAML column mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}
    return {str(key): str(value) for key, value in mapping.items() if value is not None}


def read_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Read supported local file records."""
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    if suffix == ".xlsx":
        return pd.read_excel(path).to_dict("records")
    if suffix == ".jsonl":
        return list(iter_jsonl(path))
    raise ValueError(f"unsupported local import type: {suffix}")


def articles_from_file(
    input_path: str | Path,
    mapping_path: str | Path,
    collection_batch_id: str | None = None,
    keep_raw_payload: bool = True,
) -> list[RawArticle]:
    """Convert local records to RawArticle instances."""
    path = Path(input_path)
    mapping = load_mapping(mapping_path)
    batch_id = collection_batch_id or f"local-{uuid.uuid4().hex}"
    collected_at = datetime.now(timezone.utc)
    articles: list[RawArticle] = []

    for index, record in enumerate(read_records(path)):
        get = lambda key: record.get(mapping[key]) if key in mapping else None
        title = _clean(get("title_col") or record.get("title") or record.get("headline") or path.stem)
        if not title:
            continue
        summary = _clean(get("summary_col") or record.get("summary") or record.get("description"))
        body = _clean(get("body_col") or record.get("body") or record.get("text"))
        source_name = _clean(get("source_col") or record.get("source_name")) or path.stem
        url = _clean(get("url_col") or record.get("url") or record.get("link")) or f"local://{path.name}/{index}"
        canonical_url = _clean(get("canonical_url_col") or record.get("canonical_url")) or url
        source_domain = _clean(get("source_domain_col") or record.get("source_domain")) or extract_domain(canonical_url)
        published_at = _parse_date(get("published_at_col") or record.get("published_at") or record.get("date"))
        available_date = _parse_date(get("available_date_col") or record.get("available_date")) or published_at
        content_hash = compute_content_hash(title, summary, body)
        title_hash = compute_title_hash(title)
        articles.append(
            RawArticle(
                article_id=stable_article_id("local_file", source_name, url, title),
                source_type="local_file",
                provider="local_file",
                source_name=source_name,
                source_domain=source_domain,
                raw_source_id=_clean(get("raw_source_id_col") or record.get("raw_source_id")),
                url=url,
                canonical_url=canonical_url,
                title=title,
                summary=summary,
                body=body,
                section=_clean(get("section_col") or record.get("section")),
                article_type=_clean(get("article_type_col") or record.get("article_type")),
                author=_clean(record.get("author")),
                published_at=published_at,
                collected_at=collected_at,
                available_date=available_date,
                query=_clean(record.get("query")),
                raw_payload=record if keep_raw_payload else None,
                content_hash=content_hash,
                title_hash=title_hash,
                language=_clean(record.get("language")) or "ko",
                collection_batch_id=batch_id,
            )
        )
    return articles


def import_local_file(
    input_path: str | Path,
    mapping_path: str | Path,
    out_dir: str | Path,
    seen_db_path: str | Path = "data/cache/seen_urls.sqlite",
    collection_batch_id: str | None = None,
    keep_raw_payload: bool = True,
) -> CollectionStats:
    """Import a local file into partitioned raw_lake JSONL."""
    articles = articles_from_file(input_path, mapping_path, collection_batch_id, keep_raw_payload)
    store = DedupeStore(seen_db_path)
    fresh, skipped = store.filter_new(articles)
    results = write_articles_partitioned(out_dir, fresh, Path(input_path).stem, year_subdir=True)
    return CollectionStats(collected=len(articles), written=sum(result.article_count for result in results), skipped_duplicate=skipped)
