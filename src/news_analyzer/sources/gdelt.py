"""Minimal GDELT document API collector scaffold."""

from __future__ import annotations

from datetime import date


def fetch_gdelt_documents(query: str, start_date: date, end_date: date) -> dict:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for GDELT collection. Install with: python -m pip install -e .") from exc

    response = httpx.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": query,
            "mode": "artlist",
            "format": "json",
            "startdatetime": start_date.strftime("%Y%m%d000000"),
            "enddatetime": end_date.strftime("%Y%m%d235959"),
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()
