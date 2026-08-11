"""Rule-based article analysis and JSON result handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from news_analyzer.collect import read_jsonl
from news_analyzer.config import DEFAULT_CONFIG, AnalyzerConfig
from news_analyzer.schemas import ArticleAnalysis, ArticleCleaned


REQUIRED_AI_FIELDS = {
    "candidates",
    "parties",
    "regions",
    "issues",
    "article_type",
    "stance_by_candidate",
    "candidate_link_score",
    "responsibility_target",
    "beneficiary",
    "harmed",
    "frame_tags",
    "region_relevance_score",
    "source_reliability_score",
    "analysis_confidence",
    "needs_human_review",
}


def parse_ai_json(payload: str) -> dict[str, Any]:
    """Parse a model response that must contain a single JSON object."""

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("AI output must be a JSON object")
    missing = sorted(REQUIRED_AI_FIELDS - set(data))
    if missing:
        raise ValueError(f"AI output is missing fields: {missing}")
    return data


def failed_analysis(article: ArticleCleaned, error: Exception | str) -> ArticleAnalysis:
    return ArticleAnalysis(
        article_id=article.article_id,
        published_at=article.published_at,
        available_date=article.available_date,
        source_name=article.source_name,
        article_type=article.article_type,
        candidates=article.candidate_mentions,
        parties=article.party_mentions,
        regions=article.region_mentions,
        issues=article.issue_keyword_matches,
        needs_human_review=True,
        error=str(error),
        content_hash=article.content_hash,
    )


def rule_based_analysis(article: ArticleCleaned, config: AnalyzerConfig = DEFAULT_CONFIG) -> ArticleAnalysis:
    issues = article.issue_keyword_matches or ["general"]
    candidates = article.candidate_mentions or []
    confidence = 0.65 if candidates or article.party_mentions or article.issue_keyword_matches else 0.25
    link_score = {candidate: 0.8 for candidate in candidates}
    stance = {candidate: 0.0 for candidate in candidates}
    return ArticleAnalysis(
        article_id=article.article_id,
        published_at=article.published_at,
        available_date=article.available_date,
        source_name=article.source_name,
        article_type=article.article_type,
        candidates=candidates,
        parties=article.party_mentions,
        regions=article.region_mentions,
        issues=issues,
        stance_by_candidate=stance,
        candidate_link_score=link_score,
        responsibility_target=[],
        beneficiary=[],
        harmed=[],
        frame_tags=[],
        region_relevance_score={region: 0.7 for region in article.region_mentions},
        source_reliability_score=0.7,
        analysis_confidence=confidence,
        needs_human_review=False,
        error=None,
        content_hash=article.content_hash,
    )


def analysis_from_ai_payload(article: ArticleCleaned, payload: str) -> ArticleAnalysis:
    try:
        data = parse_ai_json(payload)
        return ArticleAnalysis(
            article_id=article.article_id,
            published_at=article.published_at,
            available_date=article.available_date,
            source_name=article.source_name,
            content_hash=article.content_hash,
            **data,
        )
    except Exception as exc:
        return failed_analysis(article, exc)


def existing_article_ids(path: str | Path) -> set[str]:
    return {str(row["article_id"]) for row in read_jsonl(path) if "article_id" in row}


def analyze_file(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    resume: bool = False,
    force: bool = False,
    config: AnalyzerConfig = DEFAULT_CONFIG,
) -> int:
    records = read_jsonl(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    skip_ids = existing_article_ids(target) if resume and not force else set()
    written = 0
    with target.open("a" if resume and not force else "w", encoding="utf-8") as handle:
        for record in records:
            if limit is not None and written >= limit:
                break
            article = ArticleCleaned.model_validate(record)
            if article.article_id in skip_ids:
                continue
            analysis = rule_based_analysis(article, config)
            handle.write(json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False) + "\n")
            written += 1
    return written
