"""Common interface for source/date archive adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any

from news_collector.schemas import RawArticle


@dataclass(frozen=True)
class SourceArchiveTask:
    """One source/date/category/page task from archive_request_plan.csv."""

    task_id: str
    source_id: str
    date: date
    category_id: str
    page: int
    status: str = "pending"
    last_error: str | None = None
    output_file: str | None = None


class SourceArchiveAdapter(ABC):
    """Adapter contract for source/date archive list pages."""

    def __init__(self, source_config: dict[str, str]):
        self.source_config = source_config

    @abstractmethod
    def build_list_url(self, task_date: date, category_id: str, page: int) -> str:
        """Build a list page URL for a source/date/category/page."""

    @abstractmethod
    def parse_article_list(self, html: str) -> list[dict[str, Any]]:
        """Extract article metadata from a list page HTML string."""

    @abstractmethod
    def normalize_article(
        self,
        item: dict[str, Any],
        source_config: dict[str, str],
        task: SourceArchiveTask,
        collection_batch_id: str,
    ) -> RawArticle:
        """Convert parsed metadata to RawArticle without collecting article body."""

