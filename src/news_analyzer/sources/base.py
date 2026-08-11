"""Shared source collector protocol."""

from __future__ import annotations

from typing import Protocol

from news_analyzer.schemas import ArticleRaw


class Collector(Protocol):
    def collect(self) -> list[ArticleRaw]:
        """Return raw article records."""
