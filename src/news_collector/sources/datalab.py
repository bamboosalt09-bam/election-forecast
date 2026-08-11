"""Naver DataLab 검색어트렌드 collector → issue salience time series.

For the statistics competition, issue *salience* (how much an issue was in the
public eye) is best measured by a citable, reproducible search-volume index
rather than by scraping article bodies. Naver DataLab's Search Trend API returns
a relative search-volume ratio per keyword group over time — exactly an issue
salience signal.

Design notes
------------
- DataLab allows ≤ 5 keyword groups per request, and ratios are normalized
  *within each request* (the max data point in the request = 100). To make
  series comparable ACROSS requests, every request includes a fixed **anchor**
  keyword group; each issue's series is rescaled so the anchor's mean = 1. The
  anchor's true search volume is constant (same keyword), so this yields a
  cross-request-comparable scale. Finally all issues are min-max normalized to
  ``[0, 1]`` within the election (per ISSUE_CODEBOOK §3.1).
- HTTP is isolated in ``_post_datalab`` so the math is unit-testable offline.
- Output is candidate-agnostic issue salience. Slot/direction (who an issue
  helps or hurts) stay curated — see ``apply_salience_to_events``.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

DATALAB_ENDPOINT = "https://openapi.naver.com/v1/datalab/search"
MAX_GROUPS_PER_REQUEST = 5  # Naver limit
MAX_KEYWORDS_PER_GROUP = 20


def load_issue_keywords(path: str | Path) -> dict[str, list[str]]:
    """Read issue_keywords.csv → {issue_name: [keywords]} for DataLab groups."""

    import csv

    out: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("issue_name") or "").strip()
            kws = [t.strip() for t in (row.get("keywords") or "").split("|") if t.strip()]
            if name and kws:
                out[name] = kws
    return out


def build_keyword_groups(
    keyword_map: dict[str, list[str]], anchor_name: str, batch_size: int = 4
) -> list[list[dict[str, Any]]]:
    """Batch issues into DataLab requests, reserving one slot for the anchor.

    Returns a list of requests; each request is a list of ``{groupName, keywords}``
    dicts including the anchor group last.
    """

    issues = [
        {"groupName": name, "keywords": kws[:MAX_KEYWORDS_PER_GROUP]}
        for name, kws in keyword_map.items()
        if kws
    ]
    anchor = {"groupName": anchor_name, "keywords": [anchor_name]}
    requests: list[list[dict[str, Any]]] = []
    for i in range(0, len(issues), batch_size):
        batch = issues[i : i + batch_size] + [anchor]
        requests.append(batch)
    return requests


def _results_to_series(results: list[dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    """DataLab ``results`` → {groupName: [(period, ratio), ...]}."""

    out: dict[str, list[tuple[str, float]]] = {}
    for grp in results:
        name = grp.get("title") or grp.get("groupName")
        out[name] = [(d["period"], float(d.get("ratio", 0.0))) for d in grp.get("data", [])]
    return out


def rescale_by_anchor(
    series: dict[str, list[tuple[str, float]]], anchor_name: str
) -> dict[str, list[tuple[str, float]]]:
    """Divide every group's ratios by the anchor's mean in the same request."""

    anchor = series.get(anchor_name) or []
    anchor_vals = [r for _, r in anchor if r > 0]
    factor = (sum(anchor_vals) / len(anchor_vals)) if anchor_vals else 0.0
    if factor <= 0:
        return {k: v for k, v in series.items() if k != anchor_name}
    return {
        name: [(p, r / factor) for p, r in data]
        for name, data in series.items()
        if name != anchor_name
    }


def normalize_salience(
    rescaled: dict[str, list[tuple[str, float]]], election_id: str, instrument: str = "datalab_search"
) -> pd.DataFrame:
    """Min-max normalize across all issues/periods → salience in [0, 1] (with provenance)."""

    from news_collector.sources.salience_base import normalize_within_election

    rows = pd.DataFrame(
        [
            {"issue_name": name, "period": p, "rescaled": r}
            for name, data in rescaled.items()
            for p, r in data
        ]
    )
    return normalize_within_election(rows, "rescaled", election_id, instrument)


def _post_datalab(
    body: dict[str, Any], client_id: str, client_secret: str
) -> list[dict[str, Any]]:  # pragma: no cover - thin HTTP wrapper
    import httpx
    from tenacity import retry, stop_after_attempt, wait_exponential

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }

    @retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3), reraise=True)
    def _call() -> list[dict[str, Any]]:
        with httpx.Client() as client:
            resp = client.post(DATALAB_ENDPOINT, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            return resp.json().get("results", [])

    return _call()


def collect_issue_salience(
    keyword_map: dict[str, list[str]],
    start_date: date | str,
    end_date: date | str,
    election_id: str,
    anchor_keyword: str = "선거",
    time_unit: str = "week",
    client_id: str | None = None,
    client_secret: str | None = None,
    poster: Callable[[dict[str, Any], str, str], list[dict[str, Any]]] = _post_datalab,
) -> pd.DataFrame:
    """Fetch + anchor-rescale + normalize → salience time series DataFrame.

    ``poster`` is injectable so tests run without network. Returns columns:
    ``election_id, issue_name, period, rescaled, salience_score``.
    """

    cid = client_id or os.getenv("NAVER_CLIENT_ID")
    secret = client_secret or os.getenv("NAVER_CLIENT_SECRET")
    if not cid or not secret:
        raise RuntimeError("DataLab requires NAVER_CLIENT_ID and NAVER_CLIENT_SECRET")
    start = start_date if isinstance(start_date, str) else start_date.isoformat()
    end = end_date if isinstance(end_date, str) else end_date.isoformat()

    combined: dict[str, list[tuple[str, float]]] = {}
    for group in build_keyword_groups(keyword_map, anchor_keyword):
        body = {
            "startDate": start,
            "endDate": end,
            "timeUnit": time_unit,
            "keywordGroups": group,
        }
        results = poster(body, cid, secret)
        rescaled = rescale_by_anchor(_results_to_series(results), anchor_keyword)
        combined.update(rescaled)
    return normalize_salience(combined, election_id)


def apply_salience_to_events(events: pd.DataFrame, salience: pd.DataFrame) -> pd.DataFrame:
    """Overwrite curated issue_events ``salience_score`` from DataLab salience.

    Per ISSUE_CODEBOOK §3: salience is data-driven (DataLab), while slot/direction
    stay curated. Matches on ``election_id + issue_name`` and the event's
    ``available_date`` falling in the salience ``period`` week. When no salience
    row matches, the curated value is kept.
    """

    if events.empty or salience.empty:
        return events
    out = events.copy()
    sal = salience.copy()
    sal["period"] = pd.to_datetime(sal["period"], errors="coerce")
    out["_d"] = pd.to_datetime(out["available_date"], errors="coerce")

    def _match(row: pd.Series) -> float:
        cand = sal[(sal["election_id"] == row.get("election_id")) & (sal["issue_name"] == row.get("issue_name"))]
        if cand.empty or pd.isna(row["_d"]):
            return row.get("salience_score", 0.0)
        # nearest weekly period at or before the event date
        prior = cand[cand["period"] <= row["_d"]]
        pick = (prior if not prior.empty else cand).sort_values("period").iloc[-1]
        return float(pick["salience_score"])

    out["salience_score"] = out.apply(_match, axis=1)
    return out.drop(columns=["_d"])
