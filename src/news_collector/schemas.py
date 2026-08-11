"""Pydantic schemas for append-only raw news records."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from typing import Any

from dateutil import parser as date_parser
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def now_utc() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date_parser.parse(str(value)).date()


def _parse_datetime(value: Any) -> datetime:
    if value is None or value == "":
        return now_utc()
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = date_parser.parse(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def stable_hash(text: str) -> str:
    """Return a stable SHA-256 hash for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RawArticle(BaseModel):
    """Single raw article record persisted to raw_lake JSONL."""

    model_config = ConfigDict(extra="forbid")

    article_id: str
    source_type: str
    provider: str | None = None
    source_name: str
    source_domain: str | None = None
    raw_source_id: str | None = None
    url: str | None = None
    canonical_url: str | None = None
    title: str
    summary: str | None = None
    body: str | None = None
    section: str | None = None
    article_type: str | None = None
    author: str | None = None
    published_at: date | None = None
    collected_at: datetime = Field(default_factory=now_utc)
    available_date: date | None = None
    query: str | None = None
    raw_payload: dict[str, Any] | None = None
    content_hash: str
    title_hash: str
    duplicate_group_id: str | None = None
    language: str | None = "ko"
    collection_batch_id: str
    collection_window_start: date | None = None
    collection_window_end: date | None = None

    @field_validator("title", "source_type", "source_name", "article_id", "collection_batch_id")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("published_at", "available_date", "collection_window_start", "collection_window_end", mode="before")
    @classmethod
    def parse_dates(cls, value: Any) -> date | None:
        return _parse_date(value)

    @field_validator("collected_at", mode="before")
    @classmethod
    def parse_collected_at(cls, value: Any) -> datetime:
        return _parse_datetime(value)

    @model_validator(mode="after")
    def default_derived_fields(self) -> "RawArticle":
        if self.available_date is None and self.published_at is not None:
            self.available_date = self.published_at
        if self.provider is None:
            self.provider = self.source_type
        if self.duplicate_group_id is None:
            self.duplicate_group_id = self.content_hash
        return self
