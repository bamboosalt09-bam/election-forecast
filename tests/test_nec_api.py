"""NEC(선관위) API response parsing — offline."""

from __future__ import annotations

from news_collector.sources.nec_api import parse_items


def test_parse_items_list() -> None:
    payload = {"response": {"body": {"items": {"item": [{"name": "A"}, {"name": "B"}]}}}}
    assert [i["name"] for i in parse_items(payload)] == ["A", "B"]


def test_parse_items_single_normalized_to_list() -> None:
    payload = {"response": {"body": {"items": {"item": {"name": "solo"}}}}}
    out = parse_items(payload)
    assert isinstance(out, list) and out[0]["name"] == "solo"


def test_parse_items_empty() -> None:
    assert parse_items({"response": {"body": {"items": ""}}}) == []
    assert parse_items({}) == []
