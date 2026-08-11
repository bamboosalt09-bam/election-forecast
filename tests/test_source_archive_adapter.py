from datetime import date

from news_collector.sources.source_archive.base import SourceArchiveAdapter, SourceArchiveTask
from news_collector.sources.source_archive.generic import GenericArchiveAdapter


def test_adapter_interface_methods_exist():
    assert hasattr(SourceArchiveAdapter, "build_list_url")
    assert hasattr(SourceArchiveAdapter, "parse_article_list")
    assert hasattr(SourceArchiveAdapter, "normalize_article")


def test_generic_adapter_build_parse_normalize():
    adapter = GenericArchiveAdapter(
        {
            "source_id": "sample",
            "source_name": "Sample News",
            "source_domain": "example.com",
            "base_url": "https://example.com",
            "archive_url_template": "{base_url}/archive/{yyyymmdd}/{category_id}/{page}",
        }
    )
    url = adapter.build_list_url(date(2001, 1, 1), "all", 1)
    items = adapter.parse_article_list('<li><a href="/n/1">Title</a><time datetime="2001-01-01"></time></li>')
    article = adapter.normalize_article(
        items[0],
        adapter.source_config,
        SourceArchiveTask("sample_20010101_all_p001", "sample", date(2001, 1, 1), "all", 1),
        "batch",
    )

    assert url == "https://example.com/archive/20010101/all/1"
    assert article.source_type == "source_archive"
    assert article.provider == "sample"
    assert article.body is None

