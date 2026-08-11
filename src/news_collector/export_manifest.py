"""Create manifest.csv summaries for raw_lake files."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from news_collector.schemas import RawArticle
from news_collector.storage import iter_jsonl


def _duplicate_count_for_batches(input_dir: Path, batch_ids: set[str]) -> int:
    """Count recorded duplicate skips for the batches in a raw_lake file."""
    if not batch_ids:
        return 0
    project_root = input_dir.parent.parent if input_dir.name == "raw_lake" else Path(".")
    db_path = project_root / "data" / "cache" / "seen_urls.sqlite"
    if not db_path.exists():
        return 0
    placeholders = ",".join("?" for _ in batch_ids)
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM duplicate_events WHERE collection_batch_id IN ({placeholders})",
                sorted(batch_ids),
            ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def export_manifest(input_dir: str | Path, out_path: str | Path) -> list[dict[str, str | int | None]]:
    """Summarize raw_lake JSONL partitions."""
    manifest: list[dict[str, str | int | None]] = []
    for path in sorted(Path(input_dir).rglob("*.jsonl")):
        articles = [RawArticle.model_validate(row) for row in iter_jsonl(path)]
        if not articles:
            continue
        published_dates = [article.published_at for article in articles if article.published_at]
        available_dates = [article.available_date for article in articles if article.available_date]
        batch_ids = {article.collection_batch_id for article in articles if article.collection_batch_id}
        manifest.append(
            {
                "file_path": str(path),
                "source_type": articles[0].source_type,
                "provider": articles[0].provider,
                "source_name": articles[0].source_name,
                "article_count": len(articles),
                "min_published_at": min(published_dates).isoformat() if published_dates else None,
                "max_published_at": max(published_dates).isoformat() if published_dates else None,
                "min_available_date": min(available_dates).isoformat() if available_dates else None,
                "max_available_date": max(available_dates).isoformat() if available_dates else None,
                "missing_published_at_count": sum(1 for article in articles if article.published_at is None),
                "missing_available_date_count": sum(1 for article in articles if article.available_date is None),
                "unique_url_count": len({article.url for article in articles if article.url}),
                "unique_canonical_url_count": len({article.canonical_url for article in articles if article.canonical_url}),
                "unique_title_hash_count": len({article.title_hash for article in articles if article.title_hash}),
                "source_domain_count": len({article.source_domain for article in articles if article.source_domain}),
                "query_count": len({article.query for article in articles if article.query}),
                "batch_id": "|".join(sorted(batch_ids)),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content_hash_count": len({article.content_hash for article in articles}),
                "duplicate_skipped_count": _duplicate_count_for_batches(Path(input_dir), batch_ids),
            }
        )

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_path",
                "source_type",
                "provider",
                "source_name",
                "article_count",
                "min_published_at",
                "max_published_at",
                "min_available_date",
                "max_available_date",
                "missing_published_at_count",
                "missing_available_date_count",
                "unique_url_count",
                "unique_canonical_url_count",
                "unique_title_hash_count",
                "source_domain_count",
                "query_count",
                "batch_id",
                "created_at",
                "content_hash_count",
                "duplicate_skipped_count",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)
    return manifest
