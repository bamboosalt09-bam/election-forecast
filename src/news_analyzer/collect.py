"""Collectors and JSONL helpers for news data."""

from __future__ import annotations

from datetime import date, datetime, timezone
import csv
import hashlib
from html import unescape
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from news_analyzer.schemas import ArticleRaw


TAG_RE = re.compile(r"<[^>]+>")


def strip_html(value: str | None) -> str | None:
    if value is None:
        return None
    return unescape(TAG_RE.sub("", value)).strip()


def parse_date(value: Any, fallback: date | None = None) -> date:
    if value is None or value == "":
        if fallback is not None:
            return fallback
        raise ValueError("missing date")
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    parsed = pd.to_datetime(value, errors="raise", utc=False)
    return parsed.date()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def make_content_hash(*parts: str | None) -> str:
    text = "\n".join(part.strip() for part in parts if part and part.strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_article_id(source_type: str, url: str, title: str) -> str:
    return hashlib.sha1(f"{source_type}|{url}|{title}".encode("utf-8")).hexdigest()


def append_jsonl(path: str | Path, rows: Iterable[ArticleRaw | dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, ArticleRaw) else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generate_queries_from_plan(path: str | Path) -> list[str]:
    frame = pd.read_csv(path)
    queries: set[str] = set()
    for row in frame.to_dict("records"):
        direct = str(row.get("query", "")).strip()
        candidate = str(row.get("candidate_name", "")).strip()
        party = str(row.get("party_name", "")).strip()
        region = str(row.get("region_name", "")).strip()
        issue = str(row.get("issue_name", "")).strip()
        if direct:
            queries.add(direct)
        if candidate:
            queries.add(candidate)
            for suffix in (party, "대선", "공약", "논란", region, issue):
                if suffix:
                    queries.add(f"{candidate} {suffix}")
        if party and issue:
            queries.add(f"{party} {issue}")
        if issue:
            queries.add(f"{issue} 대선")
    return sorted(queries)


def local_file_articles(input_path: str | Path, source_name: str = "local_file") -> list[ArticleRaw]:
    path = Path(input_path)
    collected_at = now_utc()
    fallback_date = collected_at.date()

    if path.suffix.lower() == ".jsonl":
        records = read_jsonl(path)
    elif path.suffix.lower() == ".csv":
        records = pd.read_csv(path).to_dict("records")
    elif path.suffix.lower() == ".txt":
        records = [{"title": path.stem, "body": path.read_text(encoding="utf-8"), "url": str(path)}]
    else:
        raise ValueError(f"unsupported local file type: {path.suffix}")

    rows: list[ArticleRaw] = []
    for record in records:
        title = strip_html(str(record.get("title") or record.get("headline") or path.stem)) or path.stem
        summary = strip_html(record.get("summary") or record.get("description"))
        body = strip_html(record.get("body") or record.get("body_text") or record.get("text"))
        url = str(record.get("url") or record.get("link") or record.get("canonical_url") or f"local://{path.name}/{len(rows)}")
        published_at = parse_date(record.get("published_at") or record.get("published") or record.get("date"), fallback_date)
        available_date = parse_date(record.get("available_date"), published_at)
        content_hash = str(record.get("content_hash") or make_content_hash(title, summary, body))
        rows.append(
            ArticleRaw(
                article_id=str(record.get("article_id") or stable_article_id("local_file", url, title)),
                source_type="local_file",
                source_name=str(record.get("source_name") or source_name),
                url=url,
                canonical_url=record.get("canonical_url"),
                title=title,
                summary=summary,
                body=body,
                author=record.get("author"),
                section=record.get("section"),
                published_at=published_at,
                collected_at=collected_at,
                available_date=available_date,
                query=record.get("query"),
                raw_payload=record,
                content_hash=content_hash,
            )
        )
    return rows


def read_csv_column(path: str | Path, column: str) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row[column].strip() for row in reader if row.get(column, "").strip()]
