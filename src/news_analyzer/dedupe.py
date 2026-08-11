"""Article cleaning and duplicate removal."""

from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from news_analyzer.config import DEFAULT_CONFIG, AnalyzerConfig
from news_analyzer.schemas import ArticleCleaned, ArticleRaw


TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w가-힣]+", re.UNICODE)


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in TRACKING_PARAMS])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, ""))


def normalize_title(title: str) -> str:
    text = PUNCT_RE.sub(" ", title.lower())
    return SPACE_RE.sub(" ", text).strip()


def title_hash(title: str) -> str:
    return hashlib.sha1(normalize_title(title).encode("utf-8")).hexdigest()


def infer_article_type(title: str, body: str | None) -> str:
    text = f"{title} {body or ''}"
    if "팩트체크" in text:
        return "factcheck"
    if "사설" in text:
        return "editorial"
    if "칼럼" in text or "기고" in text:
        return "column"
    if "인터뷰" in text:
        return "interview"
    if "보도자료" in text:
        return "press_release"
    return "news"


def find_mentions(text: str, keywords: tuple[str, ...]) -> list[str]:
    if not keywords:
        return []
    return sorted({keyword for keyword in keywords if keyword and keyword in text})


def raw_to_cleaned(raw: ArticleRaw, config: AnalyzerConfig = DEFAULT_CONFIG) -> ArticleCleaned:
    text = " ".join(part for part in [raw.title, raw.summary or "", raw.body or ""] if part)
    canonical = raw.canonical_url or canonicalize_url(raw.url)
    return ArticleCleaned(
        article_id=raw.article_id,
        url=raw.url,
        canonical_url=canonical,
        source_name=raw.source_name,
        title=raw.title,
        summary=raw.summary,
        body_text=raw.body,
        published_at=raw.published_at,
        available_date=raw.available_date,
        article_type=infer_article_type(raw.title, raw.body),
        language="ko",
        candidate_mentions=find_mentions(text, config.candidate_keywords),
        party_mentions=find_mentions(text, config.party_keywords),
        region_mentions=find_mentions(text, config.region_keywords),
        issue_keyword_matches=find_mentions(text, config.issue_keywords),
        content_hash=raw.content_hash,
    )


def is_near_duplicate(left: ArticleCleaned, right: ArticleCleaned, threshold: float) -> bool:
    if left.source_name != right.source_name or left.published_at != right.published_at:
        return False
    ratio = SequenceMatcher(None, normalize_title(left.title), normalize_title(right.title)).ratio()
    return ratio >= threshold


def dedupe_raw_records(records: list[dict], config: AnalyzerConfig = DEFAULT_CONFIG) -> list[ArticleCleaned]:
    seen_urls: set[str] = set()
    seen_canonicals: set[str] = set()
    seen_content_hashes: set[str] = set()
    seen_title_hashes: set[str] = set()
    kept: list[ArticleCleaned] = []

    for record in records:
        cleaned = raw_to_cleaned(ArticleRaw.model_validate(record), config)
        canonical = cleaned.canonical_url or canonicalize_url(cleaned.url)
        normalized_url = canonicalize_url(cleaned.url)
        normalized_title_hash = title_hash(cleaned.title)

        if normalized_url in seen_urls:
            continue
        if canonical in seen_canonicals:
            continue
        if cleaned.content_hash in seen_content_hashes:
            continue
        if normalized_title_hash in seen_title_hashes:
            continue
        if any(is_near_duplicate(cleaned, existing, config.similarity_threshold) for existing in kept):
            continue

        seen_urls.add(normalized_url)
        seen_canonicals.add(canonical)
        seen_content_hashes.add(cleaned.content_hash)
        seen_title_hashes.add(normalized_title_hash)
        kept.append(cleaned)
    return kept
