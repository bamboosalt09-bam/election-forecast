"""Append-only JSONL storage helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from news_collector.schemas import RawArticle


@dataclass(frozen=True)
class JsonlWriteResult:
    """Summary of one append operation."""

    file_path: Path
    article_count: int


def article_month(article: RawArticle) -> str:
    """Return YYYY_MM partition suffix using available, published, then collected date."""
    basis: date = article.available_date or article.published_at or article.collected_at.date()
    return f"{basis.year:04d}_{basis.month:02d}"


def article_year(article: RawArticle) -> str:
    """Return YYYY partition using available, published, then collected date."""
    basis: date = article.available_date or article.published_at or article.collected_at.date()
    return f"{basis.year:04d}"


def append_jsonl(path: str | Path, rows: Iterable[RawArticle | dict[str, Any]]) -> int:
    """Append rows to a JSONL file without truncating existing content."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, RawArticle) else row
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON objects from a JSONL file."""
    source = Path(path)
    if not source.exists():
        return
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {source} line {line_no}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all JSONL rows into memory."""
    return list(iter_jsonl(path))


def write_articles_partitioned(
    out_dir: str | Path,
    articles: Iterable[RawArticle],
    prefix: str,
    year_subdir: bool = False,
) -> list[JsonlWriteResult]:
    """Append articles to monthly source partitions."""
    partitions: dict[Path, list[RawArticle]] = defaultdict(list)
    base = Path(out_dir)
    for article in articles:
        parent = base / article_year(article) if year_subdir else base
        partitions[parent / f"{prefix}_{article_month(article)}.jsonl"].append(article)

    results: list[JsonlWriteResult] = []
    for path, rows in sorted(partitions.items(), key=lambda item: str(item[0])):
        results.append(JsonlWriteResult(path, append_jsonl(path, rows)))
    return results
