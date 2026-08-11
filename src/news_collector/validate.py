"""Validation reporting for raw_lake JSONL files."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import ValidationError

from news_collector.schemas import RawArticle
from news_collector.storage import iter_jsonl


def validate_raw_lake(input_dir: str | Path, report_path: str | Path) -> dict[str, int]:
    """Validate raw_lake files and write a CSV report."""
    rows: list[dict[str, str]] = []
    valid = invalid = missing_available_date = 0
    for path in Path(input_dir).rglob("*.jsonl"):
        for line_no, payload in enumerate(iter_jsonl(path), start=1):
            try:
                article = RawArticle.model_validate(payload)
                valid += 1
                if article.available_date is None:
                    missing_available_date += 1
                    rows.append(
                        {
                            "file_path": str(path),
                            "line_no": str(line_no),
                            "article_id": article.article_id,
                            "severity": "warning",
                            "message": "available_date is null",
                        }
                    )
            except ValidationError as exc:
                invalid += 1
                rows.append(
                    {
                        "file_path": str(path),
                        "line_no": str(line_no),
                        "article_id": str(payload.get("article_id", "")),
                        "severity": "error",
                        "message": str(exc).replace("\n", " "),
                    }
                )

    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_path", "line_no", "article_id", "severity", "message"])
        writer.writeheader()
        writer.writerows(rows)
    return {"valid": valid, "invalid": invalid, "missing_available_date": missing_available_date}

