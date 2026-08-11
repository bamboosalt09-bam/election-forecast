"""Build query_plan.csv files from election and issue keywords."""

from __future__ import annotations

import calendar
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class QueryRow:
    """One query-window plan row."""

    query_id: str
    query: str
    category: str
    priority: int
    window_start: str
    window_end: str
    enabled: bool = True


def _read_terms(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        terms: list[str] = []
        for row in reader:
            for value in row.values():
                if value and value.strip():
                    terms.append(value.strip())
                    break
        return terms


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def build_windows(start_date: str, end_date: str, window: str = "quarterly") -> list[tuple[str, str]]:
    """Split a date range into monthly, quarterly, yearly, or single windows."""
    start = _parse_day(start_date)
    end = _parse_day(end_date)
    if start > end:
        raise ValueError("start_date must be before or equal to end_date")
    if window == "none":
        return [(start.isoformat(), end.isoformat())]

    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        if window == "monthly":
            last_day = calendar.monthrange(cursor.year, cursor.month)[1]
            window_end = date(cursor.year, cursor.month, last_day)
        elif window == "quarterly":
            quarter_end_month = ((cursor.month - 1) // 3 + 1) * 3
            last_day = calendar.monthrange(cursor.year, quarter_end_month)[1]
            window_end = date(cursor.year, quarter_end_month, last_day)
        elif window == "yearly":
            window_end = date(cursor.year, 12, 31)
        else:
            raise ValueError("window must be one of: monthly, quarterly, yearly, none")
        if window_end > end:
            window_end = end
        windows.append((cursor.isoformat(), window_end.isoformat()))
        cursor = window_end + timedelta(days=1)
    return windows


def generate_base_queries(
    base_issues: Iterable[str],
    candidates: Iterable[str],
    parties: Iterable[str],
    regions: Iterable[str],
) -> list[tuple[str, str, int]]:
    """Generate issue-first queries, with candidate combinations as support queries."""
    queries: list[tuple[str, str, int]] = []
    seen: set[str] = set()

    def add(query: str, category: str, priority: int) -> None:
        normalized = " ".join(query.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            queries.append((normalized, category, priority))

    election_terms = ["대선", "총선", "지방선거", "정권교체", "정권심판", "여당", "야당"]
    core_policy_terms = ["경제", "물가", "부동산", "연금", "일자리", "복지", "안보"]
    issue_terms = list(dict.fromkeys([*election_terms, *core_policy_terms, *base_issues]))
    candidate_terms = list(candidates)
    party_terms = list(parties)
    region_terms = list(regions)

    for issue in issue_terms:
        add(issue, "election_issue", 1)
        add(f"{issue} 대선", "election_issue", 1)

    for issue in issue_terms:
        for party in party_terms:
            add(f"{issue} {party}", "issue_party", 2)
        for candidate in candidate_terms:
            add(f"{issue} {candidate}", "issue_candidate", 2)

    for candidate in candidate_terms:
        add(f"{candidate} 대선", "candidate_support", 2)
        add(f"{candidate} 공약", "candidate_support", 2)
        add(f"{candidate} 논란", "candidate_support", 2)

    for party in party_terms:
        add(f"{party} 대선", "party_support", 2)

    for region in region_terms:
        add(f"{region} 대선", "region_support", 2)
        for issue in issue_terms:
            add(f"{region} {issue}", "region_issue", 3)

    return queries


def generate_query_rows(
    base_issues: Iterable[str],
    candidates: Iterable[str],
    parties: Iterable[str],
    regions: Iterable[str],
    start_date: str = "2021-01-01",
    end_date: str = "2022-03-09",
    window: str = "quarterly",
) -> list[QueryRow]:
    """Generate query-window rows without assigning candidate scores."""
    base_queries = generate_base_queries(base_issues, candidates, parties, regions)
    windows = build_windows(start_date, end_date, window)
    rows: list[QueryRow] = []
    index = 1
    for query, category, priority in base_queries:
        for window_start, window_end in windows:
            rows.append(
                QueryRow(
                    query_id=f"q{index:04d}",
                    query=query,
                    category=category,
                    priority=priority,
                    window_start=window_start,
                    window_end=window_end,
                    enabled=True,
                )
            )
            index += 1
    return rows


def build_query_plan(
    base_issues_path: str | Path,
    candidates_path: str | Path,
    parties_path: str | Path,
    regions_path: str | Path,
    out_path: str | Path,
    start_date: str = "2021-01-01",
    end_date: str = "2022-03-09",
    window: str = "quarterly",
) -> list[QueryRow]:
    """Read keyword CSVs and write query_plan.csv."""
    rows = generate_query_rows(
        _read_terms(base_issues_path),
        _read_terms(candidates_path),
        _read_terms(parties_path),
        _read_terms(regions_path),
        start_date=start_date,
        end_date=end_date,
        window=window,
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "query", "category", "priority", "window_start", "window_end", "enabled"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query_id": row.query_id,
                    "query": row.query,
                    "category": row.category,
                    "priority": row.priority,
                    "window_start": row.window_start,
                    "window_end": row.window_end,
                    "enabled": str(row.enabled).lower(),
                }
            )
    return rows

