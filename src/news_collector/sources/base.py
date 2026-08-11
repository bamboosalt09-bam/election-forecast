"""Shared collector primitives."""

from __future__ import annotations

from dataclasses import dataclass
import logging


@dataclass(frozen=True)
class CollectionStats:
    """Counts reported by collectors."""

    collected: int = 0
    written: int = 0
    skipped_duplicate: int = 0
    failed: int = 0


def get_logger(name: str) -> logging.Logger:
    """Return a collector logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

