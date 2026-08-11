"""Naver News Search API collector."""

from __future__ import annotations

from datetime import date
import time

from news_analyzer.collect import make_content_hash, now_utc, parse_date, stable_article_id, strip_html
from news_analyzer.config import DEFAULT_CONFIG, env
from news_analyzer.schemas import ArticleRaw


NAVER_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"


def _request(query: str, start: int, display: int, sort: str) -> dict:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for collect-naver. Install with: python -m pip install -e .") from exc

    client_id = env("NAVER_CLIENT_ID")
    client_secret = env("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("NAVER_CLIENT_ID and NAVER_CLIENT_SECRET are required")
    last_error: Exception | None = None
    for attempt in range(DEFAULT_CONFIG.retry_count):
        try:
            with httpx.Client(timeout=DEFAULT_CONFIG.timeout_seconds) as client:
                response = client.get(
                    NAVER_ENDPOINT,
                    headers={"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret},
                    params={"query": query, "start": start, "display": display, "sort": sort},
                )
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(4.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Naver API request failed after retries: {last_error}")


def collect_naver_articles(
    queries: list[str],
    start_date: date,
    end_date: date,
    display: int = 100,
    max_pages: int = 1,
    sort: str = "date",
) -> list[ArticleRaw]:
    rows: list[ArticleRaw] = []
    collected_at = now_utc()
    for query in queries:
        for page in range(max_pages):
            payload = _request(query, start=1 + page * display, display=display, sort=sort)
            for item in payload.get("items", []):
                published_at = parse_date(item.get("pubDate"), collected_at.date())
                if not start_date <= published_at <= end_date:
                    continue
                title = strip_html(item.get("title")) or ""
                summary = strip_html(item.get("description"))
                url = item.get("originallink") or item.get("link")
                content_hash = make_content_hash(title, summary)
                rows.append(
                    ArticleRaw(
                        article_id=stable_article_id("naver_api", url, title),
                        source_type="naver_api",
                        source_name="Naver News Search API",
                        url=url,
                        canonical_url=item.get("originallink") or item.get("link"),
                        title=title,
                        summary=summary,
                        body=None,
                        author=None,
                        section=None,
                        published_at=published_at,
                        collected_at=collected_at,
                        available_date=published_at,
                        query=query,
                        raw_payload=item,
                        content_hash=content_hash,
                    )
                )
            time.sleep(DEFAULT_CONFIG.rate_limit_seconds)
    return rows
