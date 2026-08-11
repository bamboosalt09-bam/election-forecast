"""Central Election Commission public-data API helpers.

Services used here:

* ``ScgnPresElctExctSttnService``: presidential election schedule metadata.
* ``CndaSrchService``: name-based integrated candidate history search.

Candidate-search results can contain different people with the same name.
Entity resolution therefore belongs in the downstream candidate-history
compiler and must use the target election, party, and birthday.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from news_collector.sources.public_data_api import (
    PublicDataApiClient,
    parse_openapi_payload,
)


NEC_BASE = "https://apis.data.go.kr/9760000"
CANDIDATE_SEARCH_BASE = f"{NEC_BASE}/CndaSrchService"
CANDIDATE_REGISTRY_BASE = f"{NEC_BASE}/PofelcddInfoInqireService"
PRESIDENTIAL_ELECTION_BASE = f"{NEC_BASE}/ScgnPresElctExctSttnService"


def parse_items(payload: Any) -> list[dict[str, Any]]:
    """Normalize a NEC response to a list of item dictionaries."""

    items, _ = parse_openapi_payload(payload)
    return items


def list_presidential_elections(
    service_key: str | None = None,
    num_rows: int = 30,
) -> list[dict[str, Any]]:
    """List presidential-election schedule metadata."""

    client = PublicDataApiClient(
        base_url=PRESIDENTIAL_ELECTION_BASE,
        service_key=service_key,
    )
    records, _ = client.fetch_all(
        "getScgnPresElctExctSttnInqire",
        params={},
        num_rows=num_rows,
    )
    return records


def search_candidate(
    name: str,
    service_key: str | None = None,
    num_rows: int = 100,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Return all integrated candidate-search records for a Korean name."""

    client = PublicDataApiClient(
        base_url=CANDIDATE_SEARCH_BASE,
        service_key=service_key,
        cache_dir=cache_dir,
    )
    records, _ = client.fetch_all(
        "getCndaSrchInqire",
        params={"name": name},
        num_rows=num_rows,
        offline=offline,
        refresh=refresh,
    )
    return records


def fetch_registered_candidates(
    *,
    sg_id: str,
    sg_typecode: str,
    service_key: str | None = None,
    num_rows: int = 100,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    sgg_name: str | None = None,
    sd_name: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch the official registered-candidate roster for one election."""

    client = PublicDataApiClient(
        base_url=CANDIDATE_REGISTRY_BASE,
        service_key=service_key,
        cache_dir=cache_dir,
    )
    return client.fetch_all(
        "getPofelcddRegistSttusInfoInqire",
        params={
            "sgId": sg_id,
            "sgTypecode": sg_typecode,
            "sggName": sgg_name,
            "sdName": sd_name,
        },
        num_rows=num_rows,
        offline=offline,
        refresh=refresh,
    )
