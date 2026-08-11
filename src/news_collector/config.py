"""Configuration helpers for news collectors."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class CollectorConfig:
    """Runtime settings shared by collectors."""

    project_root: Path = Path(".")
    cache_dir: Path = Path("data/cache")
    raw_lake_dir: Path = Path("data/raw_lake")
    logs_dir: Path = Path("data/logs")
    keep_raw_payload: bool = True
    naver_display: int = 100
    naver_start: int = 1
    naver_sort: str = "date"
    naver_max_pages: int = 10
    naver_rate_limit_seconds: float = 0.25
    gdelt_max_records: int = 250

    @property
    def seen_db_path(self) -> Path:
        return self.project_root / self.cache_dir / "seen_urls.sqlite"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.project_root / self.cache_dir / "checkpoints.sqlite"


def load_config(project_root: str | Path = ".") -> CollectorConfig:
    """Load configuration from .env and process environment."""
    root = Path(project_root)
    load_dotenv(root / ".env")
    return CollectorConfig(
        project_root=root,
        keep_raw_payload=os.getenv("NEWS_COLLECTOR_KEEP_RAW_PAYLOAD", "true").lower() != "false",
        naver_display=int(os.getenv("NAVER_DISPLAY", "100")),
        naver_start=int(os.getenv("NAVER_START", "1")),
        naver_sort=os.getenv("NAVER_SORT", "date"),
        naver_max_pages=int(os.getenv("NAVER_MAX_PAGES", "10")),
        naver_rate_limit_seconds=float(os.getenv("NAVER_RATE_LIMIT_SECONDS", "0.25")),
        gdelt_max_records=int(os.getenv("GDELT_MAX_RECORDS", "250")),
    )


def require_naver_credentials() -> tuple[str, str]:
    """Return Naver API credentials or raise a clear error."""
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("NAVER_CLIENT_ID and NAVER_CLIENT_SECRET are required for Naver collection")
    return client_id, client_secret

