"""High-level collection wrappers used by the CLI."""

from news_collector.sources.gdelt import collect_gdelt
from news_collector.sources.local_import import import_local_file
from news_collector.sources.naver_news import collect_naver
from news_collector.sources.rss import collect_rss

__all__ = ["collect_gdelt", "collect_naver", "collect_rss", "import_local_file"]

