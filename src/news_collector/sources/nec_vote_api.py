"""Client and parsers for NEC vote/count OpenAPI.

Service: VoteXmntckInfoInqireService2

The service key is intentionally read from DATA_GO_KR_SERVICE_KEY unless callers
pass it explicitly. Do not persist service keys in source-controlled files.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import httpx


NEC_VOTE_BASE = "https://apis.data.go.kr/9760000/VoteXmntckInfoInqireService2"
COUNT_OPERATION = "getXmntckSttusInfoInqire"
TURNOUT_OPERATION = "getVoteSttusInfoInqire"


def _text_of(element: ET.Element) -> str:
    return "" if element.text is None else element.text.strip()


def _xml_to_dict_items(text: str) -> tuple[list[dict[str, str]], int | None]:
    root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)

    if root.tag == "OpenAPI_ServiceResponse":
        header = root.find(".//cmmMsgHeader")
        message = _text_of(header.find("returnAuthMsg")) if header is not None else "service error"
        code = _text_of(header.find("returnReasonCode")) if header is not None else ""
        raise RuntimeError(f"OpenAPI service error {code}: {message}")

    result_code = _text_of(root.find(".//header/resultCode"))
    result_msg = _text_of(root.find(".//header/resultMsg"))
    if result_code and result_code not in {"INFO-00", "00"}:
        raise RuntimeError(f"NEC API error {result_code}: {result_msg}")

    items: list[dict[str, str]] = []
    for item in root.findall(".//items/item"):
        items.append({child.tag: _text_of(child) for child in list(item)})

    total_text = _text_of(root.find(".//body/totalCount"))
    total = int(total_text) if total_text.isdigit() else None
    return items, total


def parse_items(payload: Any) -> tuple[list[dict[str, Any]], int | None]:
    """Return API items and totalCount from XML text or JSON-like dict payload."""

    if isinstance(payload, (str, bytes)):
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        return _xml_to_dict_items(text)

    body = (payload or {}).get("response", {}).get("body", {})
    items = (body or {}).get("items", {})
    raw_item = items.get("item", []) if isinstance(items, dict) else []
    parsed = raw_item if isinstance(raw_item, list) else ([raw_item] if raw_item else [])
    total = body.get("totalCount")
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None
    return parsed, total_int


def _service_key(service_key: str | None = None) -> str:
    key = service_key or os.getenv("DATA_GO_KR_SERVICE_KEY")
    if not key:
        raise RuntimeError("DATA_GO_KR_SERVICE_KEY is not set")
    return key


def call_api(
    operation: str,
    service_key: str | None = None,
    timeout: float = 30.0,
    **params: Any,
) -> tuple[list[dict[str, Any]], int | None]:
    """Call a VoteXmntckInfoInqireService2 operation and parse items."""

    key = _service_key(service_key)
    clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
    clean_params.setdefault("resultType", "xml")
    url = f"{NEC_VOTE_BASE}/{operation}?serviceKey={key}&{urlencode(clean_params)}"
    response = httpx.get(url, timeout=timeout)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"NEC vote API HTTP {response.status_code} for {operation}; "
            "check DATA_GO_KR_SERVICE_KEY authorization"
        ) from None

    content_type = response.headers.get("content-type", "")
    if "json" in content_type:
        return parse_items(response.json())
    return parse_items(response.text)


def fetch_all_count_items(
    *,
    sg_id: str,
    sg_typecode: str,
    service_key: str | None = None,
    sgg_name: str | None = None,
    sd_name: str | None = None,
    wiw_name: str | None = None,
    num_rows: int = 100,
) -> list[dict[str, Any]]:
    """Fetch all count-result items for an election/type/location filter."""

    page = 1
    all_items: list[dict[str, Any]] = []
    while True:
        items, total = call_api(
            COUNT_OPERATION,
            service_key=service_key,
            pageNo=page,
            numOfRows=num_rows,
            sgId=sg_id,
            sgTypecode=sg_typecode,
            sggName=sgg_name,
            sdName=sd_name,
            wiwName=wiw_name,
        )
        all_items.extend(items)
        if not items:
            break
        if total is not None and len(all_items) >= total:
            break
        page += 1
    return all_items


def count_item_candidate_rows(item: dict[str, Any], max_candidates: int = 50) -> list[dict[str, Any]]:
    """Explode jdNN/hbjNN/dugsuNN fields into candidate/party vote rows."""

    rows: list[dict[str, Any]] = []
    for idx in range(1, max_candidates + 1):
        suffix = f"{idx:02d}"
        party = str(item.get(f"jd{suffix}", "") or "").strip()
        candidate = str(item.get(f"hbj{suffix}", "") or "").strip()
        votes = str(item.get(f"dugsu{suffix}", "") or "").replace(",", "").strip()
        if not party and not candidate and votes in {"", "0", "0.0"}:
            continue
        try:
            parsed_votes = float(votes) if votes else 0.0
        except ValueError:
            parsed_votes = 0.0
        rows.append(
            {
                "sgId": item.get("sgId"),
                "sgTypecode": item.get("sgTypecode"),
                "sggName": item.get("sggName"),
                "sdName": item.get("sdName"),
                "wiwName": item.get("wiwName"),
                "party": party,
                "candidate": candidate,
                "votes": parsed_votes,
                "position": idx,
            }
        )
    return rows
