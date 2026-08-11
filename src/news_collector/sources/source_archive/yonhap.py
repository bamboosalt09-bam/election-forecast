"""Yonhap (연합뉴스) archive adapter — robots-permitted category listings, metadata only.

Legality (verified): yna.co.kr/robots.txt allows ``/{category}/all`` listing pages
for ``User-agent: *`` and only disallows ``/search/``. So we crawl the category
listing feed (NOT keyword search) and read article metadata already present in the
static HTML: the article id (``data-cid="AKR{yyyymmdd}..."`` — date is embedded),
the title, and the article URL. **No article body is fetched or stored.**

Scope honesty: these listings are a *recent* feed. They suit ongoing / going-forward
collection (and the current election period), but do not provide a deep date-targeted
archive back to 2002 — Yonhap's date-filtered access lives under the robots-disallowed
``/search/`` path, which we do not touch.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from bs4 import BeautifulSoup

from news_collector.sources.source_archive.generic import GenericArchiveAdapter

_CID = re.compile(r"^AKR(\d{8})")


class YonhapArchiveAdapter(GenericArchiveAdapter):
    """Parse Yonhap category listing pages into article metadata."""

    def build_list_url(self, task_date: date, category_id: str, page: int) -> str:
        template = (self.source_config.get("archive_url_template") or "").strip()
        if template:
            return super().build_list_url(task_date, category_id, page)
        base = (self.source_config.get("base_url") or "https://www.yna.co.kr").rstrip("/")
        return f"{base}/{category_id}/all?page={page}"

    def parse_article_list(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[dict[str, Any]] = []
        for li in soup.select("li[data-cid]"):
            cid = (li.get("data-cid") or "").strip()
            m = _CID.match(cid)
            if not m:
                continue
            anchor = li.select_one("strong.tit-wrap a") or li.select_one("a.tit") or li.find("a")
            title = ""
            if anchor and anchor.get_text(strip=True):
                title = anchor.get_text(strip=True)
            if not title:
                img = li.find("img")
                title = (img.get("alt") or "").strip() if img else ""
            url = anchor.get("href") if anchor and anchor.get("href") else None
            published_at = self._date_from_cid(m.group(1))
            if title and url:
                items.append(
                    {"title": title, "url": url, "published_at": published_at, "raw_payload": {"cid": cid}}
                )
        return items

    @staticmethod
    def _date_from_cid(yyyymmdd: str) -> date | None:
        try:
            return date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
        except (ValueError, TypeError):
            return None
