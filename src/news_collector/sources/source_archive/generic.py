"""Generic source/date archive adapter for list pages."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from news_collector.dedupe import compute_content_hash, compute_title_hash, extract_domain, stable_article_id
from news_collector.schemas import RawArticle
from news_collector.source_archive_parser import parse_generic_article_list
from news_collector.sources.source_archive.base import SourceArchiveAdapter, SourceArchiveTask


class GenericArchiveAdapter(SourceArchiveAdapter):
    """Template-based adapter for archive list pages."""

    def build_list_url(self, task_date: date, category_id: str, page: int) -> str:
        template = self.source_config.get("archive_url_template", "").strip()
        if not template:
            raise ValueError(f"archive_url_template is empty for source_id={self.source_config.get('source_id')}")
        return template.format(
            date=task_date.isoformat(),
            yyyymmdd=task_date.strftime("%Y%m%d"),
            yyyy=task_date.strftime("%Y"),
            mm=task_date.strftime("%m"),
            dd=task_date.strftime("%d"),
            category_id=category_id,
            page=page,
            base_url=self.source_config.get("base_url", "").rstrip("/"),
        )

    def parse_article_list(self, html: str) -> list[dict[str, Any]]:
        return parse_generic_article_list(html, self.source_config.get("base_url", ""))

    def normalize_article(
        self,
        item: dict[str, Any],
        source_config: dict[str, str],
        task: SourceArchiveTask,
        collection_batch_id: str,
    ) -> RawArticle:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip() or None
        snippet = item.get("snippet")
        published_at = item.get("published_at") or task.date
        source_id = source_config["source_id"]
        source_name = source_config.get("source_name") or source_id
        source_domain = source_config.get("source_domain") or extract_domain(url)
        content_hash = compute_content_hash(title, str(snippet or ""), None)
        return RawArticle(
            article_id=stable_article_id("source_archive", source_name, url, title),
            source_type="source_archive",
            provider=source_id,
            source_name=source_name,
            source_domain=source_domain,
            raw_source_id=url,
            url=url,
            canonical_url=url,
            title=title,
            summary=str(snippet) if snippet else None,
            body=None,
            section=task.category_id,
            author=None,
            published_at=published_at,
            available_date=published_at,
            collected_at=datetime.now(timezone.utc),
            query=None,
            raw_payload=item.get("raw_payload") or item,
            content_hash=content_hash,
            title_hash=compute_title_hash(title),
            language="ko",
            collection_batch_id=collection_batch_id,
            collection_window_start=task.date,
            collection_window_end=task.date,
        )

