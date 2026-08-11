"""SQLite checkpoint storage for resumable collectors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class Checkpoint:
    """One collector resume checkpoint."""

    source_type: str
    item_id: str
    page_offset: int = 0
    last_success_at: str | None = None
    status: str = "pending"
    error_count: int = 0
    output_file: str | None = None
    collection_batch_id: str | None = None


class CheckpointStore:
    """Persistent checkpoint database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    source_type TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    page_offset INTEGER NOT NULL DEFAULT 0,
                    last_success_at TEXT,
                    status TEXT NOT NULL,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    output_file TEXT,
                    collection_batch_id TEXT,
                    PRIMARY KEY (source_type, item_id)
                )
                """
            )

    def get(self, source_type: str, item_id: str) -> Checkpoint | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE source_type = ? AND item_id = ?",
                (source_type, item_id),
            ).fetchone()
        return Checkpoint(**dict(row)) if row else None

    def update(
        self,
        source_type: str,
        item_id: str,
        page_offset: int,
        status: str = "success",
        error_count: int = 0,
        output_file: str | None = None,
        collection_batch_id: str | None = None,
    ) -> Checkpoint:
        last_success_at = datetime.now(timezone.utc).isoformat() if status in {"success", "complete"} else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (
                    source_type, item_id, page_offset, last_success_at, status,
                    error_count, output_file, collection_batch_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_type, item_id) DO UPDATE SET
                    page_offset = excluded.page_offset,
                    last_success_at = COALESCE(excluded.last_success_at, checkpoints.last_success_at),
                    status = excluded.status,
                    error_count = excluded.error_count,
                    output_file = excluded.output_file,
                    collection_batch_id = excluded.collection_batch_id
                """,
                (
                    source_type,
                    item_id,
                    page_offset,
                    last_success_at,
                    status,
                    error_count,
                    output_file,
                    collection_batch_id,
                ),
            )
        existing = self.get(source_type, item_id)
        if existing is None:
            raise RuntimeError("checkpoint update failed")
        return existing

    def mark_error(self, source_type: str, item_id: str, error_count: int, page_offset: int = 0) -> Checkpoint:
        """Record a source-specific error without stopping the full run."""
        return self.update(source_type, item_id, page_offset, status="error", error_count=error_count)


def checkpoint_item_id(query_id: str, window_start: str | None = None, window_end: str | None = None) -> str:
    """Build the scoped checkpoint key for a source query window."""
    if window_start or window_end:
        return f"{query_id}|{window_start or ''}|{window_end or ''}"
    return query_id
