"""HTML parsers for source/date archive list pages."""

from __future__ import annotations

from datetime import date
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from dateutil import parser as date_parser


DATE_RE = re.compile(r"(20\d{2}|19\d{2})[-./년 ]\s?(\d{1,2})[-./월 ]\s?(\d{1,2})")


def parse_date_text(value: Any) -> date | None:
    """Parse a date from HTML attributes or nearby list text."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    try:
        return date_parser.parse(text, fuzzy=True).date()
    except (TypeError, ValueError, OverflowError):
        match = DATE_RE.search(text)
        if not match:
            return None
        year, month, day = match.groups()
        return date(int(year), int(month), int(day))


def _snippet_from_parent(parent) -> str | None:
    for selector in [".summary", ".desc", ".lead", "p"]:
        found = parent.select_one(selector)
        if found:
            text = found.get_text(" ", strip=True)
            if text:
                return text
    return None


def parse_generic_article_list(html: str, base_url: str = "") -> list[dict[str, Any]]:
    """Extract title, URL, optional snippet, and date from a simple archive page."""
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for link in soup.select("a[href]"):
        title = link.get_text(" ", strip=True)
        href = link.get("href")
        if not title or not href:
            continue
        url = urljoin(base_url, href)
        if url in seen_urls:
            continue
        parent = link.find_parent(["article", "li", "div", "tr"]) or link.parent
        time_node = parent.select_one("time") if parent else None
        date_value = None
        if time_node:
            date_value = time_node.get("datetime") or time_node.get_text(" ", strip=True)
        if date_value is None and parent:
            date_value = parent.get_text(" ", strip=True)
        items.append(
            {
                "title": title,
                "url": url,
                "published_at": parse_date_text(date_value),
                "snippet": _snippet_from_parent(parent) if parent else None,
                "raw_payload": {"href": href},
            }
        )
        seen_urls.add(url)
    return items

