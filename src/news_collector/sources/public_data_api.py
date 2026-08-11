"""Defensive client for Korean public-data OpenAPI services.

The client keeps credentials in process memory only. Cache keys, request
metadata, and persisted payloads never include the service key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import httpx


SUCCESS_CODES = {"00", "INFO-00"}


class PublicDataApiError(RuntimeError):
    """Raised when a public-data service returns an API-level error."""


@dataclass(frozen=True)
class ApiPage:
    items: list[dict[str, Any]]
    total_count: int | None
    metadata: dict[str, Any]


def _text(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else element.text.strip()


def parse_openapi_payload(payload: Any) -> tuple[list[dict[str, Any]], int | None]:
    """Parse common data.go.kr JSON or XML response shapes."""

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        root = ET.fromstring(payload)
        if root.tag.endswith("OpenAPI_ServiceResponse"):
            code = _text(root.find(".//returnReasonCode"))
            message = _text(root.find(".//returnAuthMsg")) or _text(
                root.find(".//errMsg")
            )
            raise PublicDataApiError(f"public-data authentication error {code}: {message}")
        result_code = _text(root.find(".//header/resultCode"))
        result_message = _text(root.find(".//header/resultMsg"))
        if result_code and result_code not in SUCCESS_CODES:
            raise PublicDataApiError(
                f"public-data API error {result_code}: {result_message}"
            )
        items = [
            {child.tag.rsplit("}", 1)[-1]: _text(child) for child in list(item)}
            for item in root.findall(".//items/item")
        ]
        total_text = _text(root.find(".//body/totalCount"))
        return items, int(total_text) if total_text.isdigit() else None

    response = (payload or {}).get("response", {})
    header = response.get("header", {}) if isinstance(response, dict) else {}
    result_code = str(header.get("resultCode", "") or "")
    result_message = str(header.get("resultMsg", "") or "")
    if result_code and result_code not in SUCCESS_CODES:
        raise PublicDataApiError(
            f"public-data API error {result_code}: {result_message}"
        )
    body = response.get("body", {}) if isinstance(response, dict) else {}
    raw_items = body.get("items", {}) if isinstance(body, dict) else {}
    raw_item = raw_items.get("item", []) if isinstance(raw_items, dict) else []
    items = raw_item if isinstance(raw_item, list) else ([raw_item] if raw_item else [])
    total = body.get("totalCount") if isinstance(body, dict) else None
    try:
        total_count = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_count = None
    return items, total_count


class PublicDataApiClient:
    """Paginated, retrying, cache-backed OpenAPI client."""

    def __init__(
        self,
        *,
        base_url: str,
        service_key: str | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = 30.0,
        max_attempts: int = 4,
        backoff_seconds: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_key = service_key or os.getenv("DATA_GO_KR_SERVICE_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.http_client = http_client

    @staticmethod
    def _canonical_params(params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in sorted(params.items())
            if value is not None and value != "" and str(key).lower() != "servicekey"
        }

    def _cache_path(self, operation: str, params: Mapping[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        identity = json.dumps(
            {
                "base_url": self.base_url,
                "operation": operation,
                "params": self._canonical_params(params),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()
        return self.cache_dir / operation / f"{digest}.json"

    @staticmethod
    def _decode_response(response: httpx.Response) -> tuple[Any, bytes]:
        raw = response.content
        content_type = response.headers.get("content-type", "").lower()
        stripped = raw.lstrip()
        if "json" in content_type or stripped.startswith((b"{", b"[")):
            return response.json(), raw
        return raw.decode("utf-8"), raw

    @staticmethod
    def _read_cache(path: Path) -> ApiPage:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        items, total_count = parse_openapi_payload(payload)
        return ApiPage(items=items, total_count=total_count, metadata=envelope["metadata"])

    @staticmethod
    def _write_cache(
        path: Path,
        *,
        payload: Any,
        metadata: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"metadata": dict(metadata), "payload": payload}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def fetch_page(
        self,
        operation: str,
        *,
        params: Mapping[str, Any],
        offline: bool = False,
        refresh: bool = False,
    ) -> ApiPage:
        """Fetch one page, or read exactly the same request from cache."""

        clean_params = self._canonical_params(params)
        clean_params.setdefault("resultType", "json")
        cache_path = self._cache_path(operation, clean_params)
        if cache_path is not None and cache_path.exists() and not refresh:
            return self._read_cache(cache_path)
        if offline:
            raise FileNotFoundError(
                f"no cached public-data response for {operation}: {clean_params}"
            )
        if not self.service_key:
            raise RuntimeError("DATA_GO_KR_SERVICE_KEY is not set")

        query = urlencode(clean_params)
        # data.go.kr accepts encoded or raw keys inconsistently. Keeping the key
        # outside urlencode preserves the portal-provided representation.
        url = f"{self.base_url}/{operation}?serviceKey={self.service_key}&{query}"
        safe_url = f"{self.base_url}/{operation}"
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.http_client is None:
                    response = httpx.get(url, timeout=self.timeout)
                else:
                    response = self.http_client.get(url, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"transient HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    raise PublicDataApiError(
                        f"public-data HTTP {response.status_code} for {safe_url}; "
                        "verify that this specific service is approved for the key"
                    )
                payload, raw = self._decode_response(response)
                items, total_count = parse_openapi_payload(payload)
                metadata = {
                    "source_url": safe_url,
                    "operation": operation,
                    "params": clean_params,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "cache_schema": "public_data_api_cache_v1",
                }
                if cache_path is not None:
                    self._write_cache(cache_path, payload=payload, metadata=metadata)
                return ApiPage(items=items, total_count=total_count, metadata=metadata)
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise PublicDataApiError(
            f"public-data request failed after {self.max_attempts} attempts: {safe_url}"
        ) from last_error

    def fetch_all(
        self,
        operation: str,
        *,
        params: Mapping[str, Any],
        num_rows: int = 100,
        offline: bool = False,
        refresh: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fetch all pages and return records plus per-page provenance."""

        page_number = 1
        records: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        while True:
            page_params = dict(params)
            page_params.update({"pageNo": page_number, "numOfRows": num_rows})
            page = self.fetch_page(
                operation,
                params=page_params,
                offline=offline,
                refresh=refresh,
            )
            records.extend(page.items)
            provenance.append(page.metadata)
            if not page.items:
                break
            if page.total_count is not None and len(records) >= page.total_count:
                break
            if len(page.items) < num_rows and page.total_count is None:
                break
            page_number += 1
        return records, provenance
