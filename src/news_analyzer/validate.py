"""Validation helpers for analyzer outputs."""

from __future__ import annotations

from pathlib import Path

from news_analyzer.collect import read_jsonl
from news_analyzer.schemas import ArticleAnalysis


def validate_analysis_file(path: str | Path) -> int:
    count = 0
    for row in read_jsonl(path):
        ArticleAnalysis.model_validate(row)
        count += 1
    return count
