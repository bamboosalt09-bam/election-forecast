"""Configuration for the news analyzer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class AnalyzerConfig:
    rate_limit_seconds: float = 0.2
    max_concurrency: int = 4
    retry_count: int = 3
    timeout_seconds: float = 20.0
    editorial_weight: float = 0.3
    baseline_days: int = 90
    similarity_threshold: float = 0.92
    article_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "news": 1.0,
            "editorial": 0.3,
            "column": 0.3,
            "interview": 0.8,
            "factcheck": 1.2,
            "press_release": 0.4,
            "unknown": 0.5,
        }
    )
    candidate_keywords: tuple[str, ...] = ()
    party_keywords: tuple[str, ...] = ()
    region_keywords: tuple[str, ...] = ()
    issue_keywords: tuple[str, ...] = (
        "부동산",
        "일자리",
        "경제",
        "복지",
        "외교",
        "안보",
        "검찰",
        "교육",
        "청년",
        "연금",
    )


DEFAULT_CONFIG = AnalyzerConfig()


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable without forcing dotenv as a dependency."""

    return os.environ.get(name, default)
