"""SQLite-backed duplicate prevention for raw news collection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sqlite3
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from news_collector.schemas import RawArticle


TRACKING_PREFIXES = ("utm_",)
TRACKING_PARAMS = {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "ref", "spm"}
SPACE_RE = re.compile(r"\s+")


def normalize_url(url: str | None) -> str | None:
    """Normalize a URL enough for stable duplicate checks."""
    if not url:
        return None
    value = url.strip()
    if not value:
        return None
    split = urlsplit(value)
    scheme = split.scheme.lower() or "https"
    netloc = split.netloc.lower()
    path = re.sub(r"/+", "/", split.path).rstrip("/")
    kept = []
    for key, val in parse_qsl(split.query, keep_blank_values=False):
        lower = key.lower()
        if lower in TRACKING_PARAMS or any(lower.startswith(prefix) for prefix in TRACKING_PREFIXES):
            continue
        kept.append((key, val))
    query = urlencode(sorted(kept))
    return urlunsplit((scheme, netloc, path or "/", query, ""))


def canonicalize_url(url: str | None) -> str | None:
    """Return canonical URL candidate used for dedupe."""
    return normalize_url(url)


def extract_domain(url: str | None) -> str | None:
    """Extract a lower-cased hostname from a URL."""
    if not url:
        return None
    netloc = urlsplit(url.strip()).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


def normalize_title(title: str) -> str:
    """Strip HTML and normalize whitespace in a title."""
    text = BeautifulSoup(title or "", "html.parser").get_text(" ")
    return SPACE_RE.sub(" ", text).strip().lower()


def hash_text(text: str | None) -> str:
    """Hash normalized text with SHA-256."""
    normalized = SPACE_RE.sub(" ", (text or "").strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_content_hash(title: str | None, summary: str | None = None, body: str | None = None) -> str:
    """Hash title, summary, and body as a duplicate-resistant content key."""
    text = "\n".join(part.strip() for part in (title, summary, body) if part and part.strip())
    return hash_text(text)


def compute_title_hash(title: str) -> str:
    """Hash a normalized title."""
    return hash_text(normalize_title(title))


def stable_article_id(source_type: str, source_name: str, url: str | None, title: str) -> str:
    """Build a deterministic article id from source metadata."""
    basis = "|".join([source_type, source_name, normalize_url(url) or "", normalize_title(title)])
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def dedupe_keys(article: RawArticle) -> list[tuple[str, str]]:
    """Return all dedupe keys for an article."""
    keys: list[tuple[str, str]] = []
    url = normalize_url(article.url)
    canonical = canonicalize_url(article.canonical_url)
    if url:
        keys.append(("url", url))
    if canonical:
        keys.append(("canonical_url", canonical))
    if article.content_hash:
        keys.append(("content_hash", article.content_hash))
    return keys


class DedupeStore:
    """Persistent duplicate key store."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen (
                    key_type TEXT NOT NULL,
                    key_value TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    source_name TEXT,
                    published_at TEXT,
                    first_seen_at TEXT NOT NULL,
                    PRIMARY KEY (key_type, key_value)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS duplicate_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_type TEXT NOT NULL,
                    key_value TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_name TEXT,
                    collection_batch_id TEXT,
                    skipped_at TEXT NOT NULL
                )
                """
            )

    def is_seen(self, article: RawArticle) -> bool:
        keys = dedupe_keys(article)
        if not keys:
            return False
        with self._connect() as conn:
            for key_type, key_value in keys:
                row = conn.execute(
                    "SELECT 1 FROM seen WHERE key_type = ? AND key_value = ? LIMIT 1",
                    (key_type, key_value),
                ).fetchone()
                if row:
                    return True
        return False

    def mark_duplicate(self, article: RawArticle) -> None:
        """Record that a candidate row was skipped as a duplicate."""
        keys = dedupe_keys(article)
        if not keys:
            return
        key_type, key_value = keys[0]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO duplicate_events (
                    key_type, key_value, article_id, source_type, source_name,
                    collection_batch_id, skipped_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key_type,
                    key_value,
                    article.article_id,
                    article.source_type,
                    article.source_name,
                    article.collection_batch_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def mark_seen(self, article: RawArticle) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                key_type,
                key_value,
                article.article_id,
                article.source_name,
                article.published_at.isoformat() if article.published_at else None,
                now,
            )
            for key_type, key_value in dedupe_keys(article)
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO seen
                    (key_type, key_value, article_id, source_name, published_at, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def filter_new(self, articles: Iterable[RawArticle]) -> tuple[list[RawArticle], int]:
        fresh: list[RawArticle] = []
        skipped = 0
        for article in articles:
            if self.is_seen(article):
                skipped += 1
                self.mark_duplicate(article)
                continue
            self.mark_seen(article)
            fresh.append(article)
        return fresh, skipped


_default_store: DedupeStore | None = None


def _store(db_path: str | Path = "data/cache/seen_urls.sqlite") -> DedupeStore:
    global _default_store
    target = Path(db_path)
    if _default_store is None or _default_store.db_path != target:
        _default_store = DedupeStore(target)
    return _default_store


def is_seen(article: RawArticle, db_path: str | Path = "data/cache/seen_urls.sqlite") -> bool:
    """Return whether an article has already been collected."""
    return _store(db_path).is_seen(article)


def mark_seen(article: RawArticle, db_path: str | Path = "data/cache/seen_urls.sqlite") -> None:
    """Persist all dedupe keys for an article."""
    _store(db_path).mark_seen(article)
