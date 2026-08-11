"""Election-window aware issue term weights."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

ELECTION_ORDER = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")


def _election_index(election_id: str) -> int:
    try:
        return ELECTION_ORDER.index(election_id)
    except ValueError:
        return len(ELECTION_ORDER)


def _is_active(election_id: str, start_election: str, end_election: str) -> bool:
    current = _election_index(election_id)
    start = _election_index(start_election) if start_election else 0
    end = _election_index(end_election) if end_election else len(ELECTION_ORDER) - 1
    return start <= current <= end


def load_campaign_issue_terms(
    path: str | Path,
    election_ids: Iterable[str] = ELECTION_ORDER,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[tuple[str, str], float]]]:
    """Load election-specific campaign terms.

    Returns ``(terms_by_election, weights_by_election)``:
    - terms_by_election[election_id][issue_name] -> extra active terms
    - weights_by_election[election_id][(issue_name, term)] -> term-specific weight
    """

    terms_by_election: dict[str, dict[str, list[str]]] = {election_id: {} for election_id in election_ids}
    weights_by_election: dict[str, dict[tuple[str, str], float]] = {election_id: {} for election_id in election_ids}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            issue_name = (row.get("issue_name") or "").strip()
            term = (row.get("term") or "").strip()
            if not issue_name or not term:
                continue
            weight = float(row.get("weight") or 1.0)
            start_election = (row.get("start_election") or "").strip()
            end_election = (row.get("end_election") or "").strip()
            for election_id in election_ids:
                if not _is_active(election_id, start_election, end_election):
                    continue
                terms_by_election[election_id].setdefault(issue_name, []).append(term)
                weights_by_election[election_id][(issue_name, term)] = weight
    return terms_by_election, weights_by_election


def merge_issue_terms(
    base_terms: dict[str, list[str]],
    extra_terms: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Merge base and election-specific issue terms without duplicates."""

    merged = {issue: list(terms) for issue, terms in base_terms.items()}
    for issue, terms in extra_terms.items():
        existing = set(merged.get(issue, []))
        for term in terms:
            if term not in existing:
                merged.setdefault(issue, []).append(term)
                existing.add(term)
    return merged
