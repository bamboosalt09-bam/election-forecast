"""Build source/date archive request plans."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass(frozen=True)
class ArchivePlanRow:
    """One archive request task."""

    task_id: str
    source_id: str
    date: str
    category_id: str
    page: int
    status: str
    last_error: str
    output_file: str


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _enabled(row: dict[str, str]) -> bool:
    return str(row.get("enabled", "true")).strip().lower() in {"true", "1", "yes", "y"}


def _date_range(start_date: str, end_date: str, end_exclusive: bool) -> list[date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if not end_exclusive:
        end = end + timedelta(days=1)
    days: list[date] = []
    cursor = start
    while cursor < end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def output_file_for_task(out_dir: str | Path, source_id: str, task_date: date) -> str:
    """Return partitioned source archive output path."""
    return str(Path(out_dir) / f"source={source_id}" / f"year={task_date.year:04d}" / f"month={task_date.month:02d}" / "part.jsonl")


def build_source_archive_plan(
    sources_path: str | Path,
    categories_path: str | Path,
    source_id: str,
    start_date: str,
    end_date: str,
    out_path: str | Path,
    end_exclusive: bool = False,
    window: str = "day",
    out_dir: str | Path = "data/raw_lake/source_archive",
) -> list[ArchivePlanRow]:
    """Write archive_request_plan.csv for source/date/category/page collection."""
    if window != "day":
        raise ValueError("source archive currently supports window=day")
    sources = [row for row in _read_csv(sources_path) if row.get("source_id") == source_id and _enabled(row)]
    categories = [row for row in _read_csv(categories_path) if _enabled(row)]
    if not sources:
        rows: list[ArchivePlanRow] = []
    else:
        rows = []
        for task_date in _date_range(start_date, end_date, end_exclusive):
            for category in categories:
                category_id = category["category_id"]
                task_id = f"{source_id}_{task_date.strftime('%Y%m%d')}_{category_id}_p001"
                rows.append(
                    ArchivePlanRow(
                        task_id=task_id,
                        source_id=source_id,
                        date=task_date.isoformat(),
                        category_id=category_id,
                        page=1,
                        status="pending",
                        last_error="",
                        output_file=output_file_for_task(out_dir, source_id, task_date),
                    )
                )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "source_id", "date", "category_id", "page", "status", "last_error", "output_file"],
        )
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])
    return rows

