"""Cleaning entry points."""

from __future__ import annotations

from pathlib import Path
import json

from news_analyzer.collect import read_jsonl
from news_analyzer.config import DEFAULT_CONFIG, AnalyzerConfig
from news_analyzer.dedupe import dedupe_raw_records


def clean_file(input_path: str | Path, output_path: str | Path, config: AnalyzerConfig = DEFAULT_CONFIG) -> int:
    records = read_jsonl(input_path)
    cleaned = dedupe_raw_records(records, config)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in cleaned:
            handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return len(cleaned)
