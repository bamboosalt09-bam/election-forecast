from __future__ import annotations

import json

import httpx
import pytest

from news_collector.sources.public_data_api import (
    PublicDataApiClient,
    PublicDataApiError,
    parse_openapi_payload,
)


def test_parse_openapi_xml_and_api_error() -> None:
    xml = """<response><header><resultCode>INFO-00</resultCode><resultMsg>NORMAL</resultMsg></header><body><items><item><name>A</name></item></items><totalCount>1</totalCount></body></response>"""
    items, total = parse_openapi_payload(xml)
    assert items == [{"name": "A"}]
    assert total == 1

    error = """<response><header><resultCode>ERROR-340</resultCode><resultMsg>missing</resultMsg></header><body /></response>"""
    with pytest.raises(PublicDataApiError, match="ERROR-340"):
        parse_openapi_payload(error)


def test_client_paginates_caches_and_never_persists_key(tmp_path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        page = int(request.url.params["pageNo"])
        payload = {
            "response": {
                "header": {"resultCode": "INFO-00", "resultMsg": "NORMAL"},
                "body": {
                    "items": {"item": [{"name": f"row-{page}"}]},
                    "totalCount": 2,
                },
            }
        }
        return httpx.Response(200, json=payload, request=request)

    secret = "not-to-be-persisted"
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = PublicDataApiClient(
        base_url="https://example.test/service",
        service_key=secret,
        cache_dir=tmp_path,
        http_client=http_client,
    )
    records, provenance = client.fetch_all("operation", params={"name": "테스트"}, num_rows=1)
    assert [record["name"] for record in records] == ["row-1", "row-2"]
    assert len(provenance) == 2
    assert all("serviceKey" not in json.dumps(item) for item in provenance)
    assert len(calls) == 2

    cache_text = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json")
    )
    assert secret not in cache_text

    offline = PublicDataApiClient(
        base_url="https://example.test/service",
        cache_dir=tmp_path,
    )
    cached, _ = offline.fetch_all(
        "operation", params={"name": "테스트"}, num_rows=1, offline=True
    )
    assert cached == records


def test_client_does_not_retry_unapproved_service() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403, text="forbidden", request=request)

    client = PublicDataApiClient(
        base_url="https://example.test/service",
        service_key="temporary",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(PublicDataApiError, match="specific service is approved"):
        client.fetch_page("operation", params={"pageNo": 1})
    assert calls == 1
