"""Run source/date archive request plans and write RawArticle metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import random
import time
from typing import Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import uuid

import httpx
import yaml

from news_collector.dedupe import DedupeStore
from news_collector.sources.base import CollectionStats
from news_collector.sources.source_archive.base import SourceArchiveAdapter, SourceArchiveTask
from news_collector.sources.source_archive.generic import GenericArchiveAdapter
from news_collector.sources.source_archive.yonhap import YonhapArchiveAdapter
from news_collector.storage import append_jsonl


HttpGet = Callable[[str, dict[str, str]], tuple[int, str]]


@dataclass(frozen=True)
class CrawlPolicy:
    """Sequential archive crawl policy."""

    max_concurrency: int = 1
    min_delay_seconds: float = 3.0
    random_delay_seconds_min: float = 2.0
    random_delay_seconds_max: float = 8.0
    stop_on_403: bool = True
    cooldown_on_429_minutes: int = 120
    retry_5xx: int = 3
    body_collection: bool = False
    user_agent: str = "StudentElectionForecastPilot/0.1"
    respect_robots_txt: bool = True


def load_crawl_policy(path: str | Path) -> CrawlPolicy:
    """Load crawl policy YAML with conservative defaults."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return CrawlPolicy(**{key: value for key, value in payload.items() if key in CrawlPolicy.__dataclass_fields__})


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_plan(path: str | Path, rows: list[dict[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "source_id", "date", "category_id", "page", "status", "last_error", "output_file"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _enabled(row: dict[str, str]) -> bool:
    return str(row.get("enabled", "true")).strip().lower() in {"true", "1", "yes", "y"}


def load_sources(path: str | Path = "data/config/source_list.csv") -> dict[str, dict[str, str]]:
    """Load enabled source configs by source_id."""
    return {row["source_id"]: row for row in _read_csv(path) if _enabled(row)}


def adapter_for_source(source_config: dict[str, str]) -> SourceArchiveAdapter:
    """Select an archive adapter by parser_type."""
    parser_type = source_config.get("parser_type", "generic")
    if parser_type == "yonhap_archive":
        return YonhapArchiveAdapter(source_config)
    if parser_type in {"generic", "generic_archive"}:
        return GenericArchiveAdapter(source_config)
    raise ValueError(f"unsupported parser_type={parser_type}")


def _default_http_get(url: str, headers: dict[str, str]) -> tuple[int, str]:
    with httpx.Client(follow_redirects=True, timeout=20) as client:
        response = client.get(url, headers=headers)
    return response.status_code, response.text


def _robots_allowed(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception:  # noqa: BLE001 - inability to read robots is handled conservatively by robotparser state.
        return True
    return parser.can_fetch(user_agent, url)


def _task_from_row(row: dict[str, str]) -> SourceArchiveTask:
    return SourceArchiveTask(
        task_id=row["task_id"],
        source_id=row["source_id"],
        date=date.fromisoformat(row["date"]),
        category_id=row["category_id"],
        page=int(row.get("page") or 1),
        status=row.get("status") or "pending",
        last_error=row.get("last_error") or None,
        output_file=row.get("output_file") or None,
    )


def _sleep(policy: CrawlPolicy) -> None:
    delay = policy.min_delay_seconds + random.uniform(policy.random_delay_seconds_min, policy.random_delay_seconds_max)
    if delay > 0:
        time.sleep(delay)


def collect_source_archive(
    request_plan_path: str | Path,
    crawl_policy_path: str | Path,
    out_dir: str | Path = "data/raw_lake/source_archive",
    resume: bool = True,
    source_list_path: str | Path = "data/config/source_list.csv",
    seen_db_path: str | Path = "data/cache/seen_urls.sqlite",
    http_get: HttpGet | None = None,
    sleep_between_requests: bool = True,
) -> CollectionStats:
    """Execute archive_request_plan.csv sequentially and write metadata-only RawArticle JSONL."""
    policy = load_crawl_policy(crawl_policy_path)
    if policy.body_collection:
        raise ValueError("source archive collector only supports body_collection=false")
    sources = load_sources(source_list_path)
    rows = _read_csv(request_plan_path)
    get = http_get or _default_http_get
    store = DedupeStore(seen_db_path)
    batch_id = f"source-archive-{uuid.uuid4().hex}"
    collected = written = skipped = failed = 0

    for row in rows:
        if resume and row.get("status") in {"done", "skipped_duplicate"}:
            continue
        task = _task_from_row(row)
        source_config = sources.get(task.source_id)
        if source_config is None:
            row["status"] = "failed"
            row["last_error"] = f"source_id not enabled or missing: {task.source_id}"
            failed += 1
            continue
        try:
            adapter = adapter_for_source(source_config)
            url = adapter.build_list_url(task.date, task.category_id, task.page)
            if policy.respect_robots_txt and not _robots_allowed(url, policy.user_agent):
                row["status"] = "failed"
                row["last_error"] = "robots.txt disallows request"
                failed += 1
                continue
            status_code = 0
            html = ""
            for attempt in range(policy.retry_5xx + 1):
                status_code, html = get(url, {"User-Agent": policy.user_agent})
                if status_code < 500 or attempt == policy.retry_5xx:
                    break
            if status_code == 403 and policy.stop_on_403:
                row["status"] = "failed"
                row["last_error"] = "403 forbidden; stopped without bypass"
                failed += 1
                continue
            if status_code == 429:
                row["status"] = "cooldown"
                row["last_error"] = f"429 rate limited; cooldown {policy.cooldown_on_429_minutes} minutes"
                failed += 1
                continue
            if status_code >= 400:
                row["status"] = "failed"
                row["last_error"] = f"HTTP {status_code}"
                failed += 1
                continue
            articles = [
                adapter.normalize_article(item, source_config, task, batch_id)
                for item in adapter.parse_article_list(html)
                if item.get("title") and item.get("url")
            ]
            fresh, duplicate_count = store.filter_new(articles)
            output_file = str(Path(out_dir) / f"source={task.source_id}" / f"year={task.date.year:04d}" / f"month={task.date.month:02d}" / "part.jsonl")
            append_jsonl(output_file, fresh)
            row["output_file"] = output_file
            row["status"] = "done" if fresh else "skipped_duplicate"
            row["last_error"] = ""
            collected += len(articles)
            written += len(fresh)
            skipped += duplicate_count
            if sleep_between_requests:
                _sleep(policy)
        except Exception as exc:  # noqa: BLE001 - task failures are checkpointed in the request plan.
            row["status"] = "failed"
            row["last_error"] = str(exc)
            failed += 1

    _write_plan(request_plan_path, rows)
    return CollectionStats(collected=collected, written=written, skipped_duplicate=skipped, failed=failed)

