"""Source/date archive collection adapters."""

from news_collector.sources.source_archive.base import SourceArchiveAdapter
from news_collector.sources.source_archive.generic import GenericArchiveAdapter
from news_collector.sources.source_archive.yonhap import YonhapArchiveAdapter

__all__ = ["GenericArchiveAdapter", "SourceArchiveAdapter", "YonhapArchiveAdapter"]

