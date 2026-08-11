from news_collector.source_archive_parser import parse_generic_article_list


def test_parse_generic_article_list_metadata_only():
    html = """
    <ul>
      <li>
        <a href="/news/1">첫 기사</a>
        <time datetime="2001-01-02">2001-01-02</time>
        <p class="summary">목록 요약</p>
        <div class="body">본문은 수집하지 않는다</div>
      </li>
    </ul>
    """

    items = parse_generic_article_list(html, "https://example.com")

    assert items[0]["title"] == "첫 기사"
    assert items[0]["url"] == "https://example.com/news/1"
    assert items[0]["published_at"].isoformat() == "2001-01-02"
    assert "body" not in items[0]

