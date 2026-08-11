"""Pydantic schemas used by the news analyzer pipeline."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ArticleType = Literal["news", "editorial", "column", "interview", "factcheck", "press_release", "unknown"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArticleRaw(StrictBaseModel):
    article_id: str
    source_type: str
    source_name: str
    url: str
    canonical_url: str | None = None
    title: str
    summary: str | None = None
    body: str | None = None
    author: str | None = None
    section: str | None = None
    published_at: date
    collected_at: datetime
    available_date: date
    query: str | None = None
    raw_payload: dict[str, Any] | None = None
    content_hash: str


class ArticleCleaned(StrictBaseModel):
    article_id: str
    url: str
    canonical_url: str | None = None
    source_name: str
    title: str
    summary: str | None = None
    body_text: str | None = None
    published_at: date
    available_date: date
    article_type: ArticleType = "unknown"
    language: str = "ko"
    candidate_mentions: list[str] = Field(default_factory=list)
    party_mentions: list[str] = Field(default_factory=list)
    region_mentions: list[str] = Field(default_factory=list)
    issue_keyword_matches: list[str] = Field(default_factory=list)
    content_hash: str


class ArticleAnalysis(StrictBaseModel):
    article_id: str
    published_at: date
    available_date: date
    source_name: str
    article_type: ArticleType = "unknown"
    candidates: list[str] = Field(default_factory=list)
    parties: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    stance_by_candidate: dict[str, float] = Field(default_factory=dict)
    candidate_link_score: dict[str, float] = Field(default_factory=dict)
    responsibility_target: list[str] = Field(default_factory=list)
    beneficiary: list[str] = Field(default_factory=list)
    harmed: list[str] = Field(default_factory=list)
    frame_tags: list[str] = Field(default_factory=list)
    region_relevance_score: dict[str, float] = Field(default_factory=dict)
    source_reliability_score: float = 0.7
    analysis_confidence: float = 0.5
    needs_human_review: bool = False
    error: str | None = None
    content_hash: str | None = None

    @field_validator("source_reliability_score", "analysis_confidence")
    @classmethod
    def validate_probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        return value

    @field_validator("stance_by_candidate")
    @classmethod
    def validate_stance(cls, value: dict[str, float]) -> dict[str, float]:
        for score in value.values():
            if not -1.0 <= score <= 1.0:
                raise ValueError("stance scores must be between -1.0 and 1.0")
        return value

    @field_validator("candidate_link_score", "region_relevance_score")
    @classmethod
    def validate_link_scores(cls, value: dict[str, float]) -> dict[str, float]:
        for score in value.values():
            if not 0.0 <= score <= 1.0:
                raise ValueError("link scores must be between 0.0 and 1.0")
        return value


class AggregatedIssueScore(StrictBaseModel):
    date: date
    window_start: date
    window_end: date
    candidate_id: str
    candidate_name: str
    issue_name: str
    region_id: str = "ALL"
    article_count: int
    weighted_stance: float
    avg_candidate_link_score: float
    avg_confidence: float
    source_reliability_avg: float
    volume_z_score: float
    final_issue_score: float
    available_date: date
