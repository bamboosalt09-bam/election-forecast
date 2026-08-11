"""Aggregate article analysis JSONL into candidate issue scores."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import json
import math

import pandas as pd

from news_analyzer.collect import read_jsonl
from news_analyzer.config import DEFAULT_CONFIG, AnalyzerConfig
from news_analyzer.schemas import AggregatedIssueScore, ArticleAnalysis


def candidate_id(candidate_name: str) -> str:
    return candidate_name.strip().lower().replace(" ", "_")


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _weighted_stance(row: ArticleAnalysis, candidate: str, config: AnalyzerConfig) -> float:
    stance = row.stance_by_candidate.get(candidate, 0.0)
    article_weight = config.article_type_weights.get(row.article_type, config.article_type_weights["unknown"])
    return stance * article_weight


def _volume_score(current_count: int, baseline_count: int, window_days: int, baseline_days: int) -> float:
    expected = baseline_count * (window_days / baseline_days) if baseline_days else 0.0
    if expected <= 0:
        return float(current_count)
    return (current_count - expected) / math.sqrt(expected)


def aggregate_records(
    records: list[dict],
    forecast_date: date,
    window_days: list[int],
    config: AnalyzerConfig = DEFAULT_CONFIG,
) -> list[AggregatedIssueScore]:
    analyses = [ArticleAnalysis.model_validate(record) for record in records]
    eligible = [row for row in analyses if row.available_date <= forecast_date and row.candidates and row.issues]
    output: list[AggregatedIssueScore] = []

    for days in window_days:
        window_start = forecast_date - timedelta(days=days)
        baseline_start = window_start - timedelta(days=config.baseline_days)
        in_window = [row for row in eligible if window_start <= row.published_at <= forecast_date]
        in_baseline = [row for row in eligible if baseline_start <= row.published_at < window_start]

        groups: dict[tuple[str, str], list[ArticleAnalysis]] = defaultdict(list)
        baseline_counts: dict[str, int] = defaultdict(int)
        for row in in_window:
            for candidate in row.candidates:
                for issue in row.issues:
                    groups[(candidate, issue)].append(row)
        for row in in_baseline:
            for issue in row.issues:
                baseline_counts[issue] += 1

        for (candidate, issue), rows in sorted(groups.items()):
            weighted_stances = [_weighted_stance(row, candidate, config) for row in rows]
            link_scores = [row.candidate_link_score.get(candidate, 0.0) for row in rows]
            confidences = [row.analysis_confidence for row in rows]
            reliabilities = [row.source_reliability_score for row in rows]
            weighted_stance = _mean(weighted_stances)
            avg_link = _mean(link_scores)
            avg_confidence = _mean(confidences)
            avg_reliability = _mean(reliabilities)
            volume_z = _volume_score(len(rows), baseline_counts[issue], days, config.baseline_days)
            final = round(volume_z * weighted_stance * avg_link * avg_confidence * avg_reliability, 12)
            output.append(
                AggregatedIssueScore(
                    date=forecast_date,
                    window_start=window_start,
                    window_end=forecast_date,
                    candidate_id=candidate_id(candidate),
                    candidate_name=candidate,
                    issue_name=issue,
                    region_id="ALL",
                    article_count=len(rows),
                    weighted_stance=weighted_stance,
                    avg_candidate_link_score=avg_link,
                    avg_confidence=avg_confidence,
                    source_reliability_avg=avg_reliability,
                    volume_z_score=volume_z,
                    final_issue_score=final,
                    available_date=forecast_date,
                )
            )
    return output


def aggregate_file(
    analysis_path: str | Path,
    output_path: str | Path,
    forecast_date: date,
    window_days: list[int],
    config: AnalyzerConfig = DEFAULT_CONFIG,
) -> int:
    rows = aggregate_records(read_jsonl(analysis_path), forecast_date, window_days, config)
    frame = pd.DataFrame([row.model_dump(mode="json") for row in rows])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(columns=list(AggregatedIssueScore.model_fields))
    frame.to_csv(target, index=False, encoding="utf-8")
    return len(frame)


def aggregate_file_jsonl(
    analysis_path: str | Path,
    output_path: str | Path,
    forecast_date: date,
    window_days: list[int],
    config: AnalyzerConfig = DEFAULT_CONFIG,
) -> int:
    rows = aggregate_records(read_jsonl(analysis_path), forecast_date, window_days, config)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return len(rows)
