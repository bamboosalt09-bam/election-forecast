"""Yonhap archive adapter — listing parse + date-from-id (offline, synthetic HTML)."""

from __future__ import annotations

from datetime import date

from news_collector.sources.source_archive.yonhap import YonhapArchiveAdapter

# Synthetic HTML mirroring Yonhap's list structure (no real article content).
SAMPLE = """
<ul class="list01">
  <li data-cid="AKR20221115083351001">
    <strong class="tit-wrap"><a href="https://www.yna.co.kr/view/AKR20221115083351001">부동산 정책 관련 기사 제목</a></strong>
  </li>
  <li data-cid="AKR20221116090000002">
    <strong class="tit-wrap"><a href="https://www.yna.co.kr/view/AKR20221116090000002">대북 안보 관련 기사 제목</a></strong>
  </li>
  <li data-cid="NOTAKR123"><a href="/x">광고/비기사</a></li>
</ul>
"""


def _adapter() -> YonhapArchiveAdapter:
    return YonhapArchiveAdapter({"source_id": "yonhap", "source_name": "연합뉴스", "base_url": "https://www.yna.co.kr"})


def test_build_list_url_uses_category_page_not_search() -> None:
    url = _adapter().build_list_url(date(2022, 11, 15), "politics", 3)
    assert url == "https://www.yna.co.kr/politics/all?page=3"
    assert "/search/" not in url  # robots-disallowed path must never be built


def test_parse_extracts_title_url_and_date_from_cid() -> None:
    items = _adapter().parse_article_list(SAMPLE)
    assert len(items) == 2  # NOTAKR row skipped
    first = items[0]
    assert first["title"] == "부동산 정책 관련 기사 제목"
    assert first["url"].endswith("AKR20221115083351001")
    assert first["published_at"] == date(2022, 11, 15)  # date parsed from data-cid


def test_normalize_article_is_metadata_only() -> None:
    adapter = _adapter()
    item = adapter.parse_article_list(SAMPLE)[0]
    from news_collector.sources.source_archive.base import SourceArchiveTask

    task = SourceArchiveTask(task_id="t1", source_id="yonhap", date=date(2022, 11, 15), category_id="politics", page=1)
    art = adapter.normalize_article(item, {"source_id": "yonhap", "source_name": "연합뉴스"}, task, "batch1")
    assert art.body is None  # never collect body
    assert art.published_at == date(2022, 11, 15)
    assert art.title == "부동산 정책 관련 기사 제목"
