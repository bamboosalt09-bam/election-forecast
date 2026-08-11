"""Build constituency-level Assembly results from NEC count records."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterable

import pandas as pd

from election_forecast.features.region_bloc_prior import normalize_bloc
from news_collector.sources.nec_vote_api import count_item_candidate_rows
from presidential_issue_engine.build_bloc_history_from_nec import REGION_IDS


SCHEMA_VERSION = "nec_assembly_district_history_v1"
INDEPENDENT_LABELS = {"", "무소속", "무소속후보자"}


def build_assembly_district_history(
    items: Iterable[dict[str, Any]],
    *,
    election_id: str,
    election_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Normalize one NEC Assembly election to one row per district candidate.

    NEC exposes duplicate national, province, municipality, and polling-area
    levels. ``sdName != 합계`` and ``wiwName == 합계`` selects exactly one
    province-level total for each constituency.
    """

    source_date = pd.Timestamp(election_date)
    rows: list[dict[str, Any]] = []
    seen_districts: set[tuple[str, str]] = set()
    for item in items:
        region_name = str(item.get("sdName", "") or "").strip()
        municipality = str(item.get("wiwName", "") or "").strip()
        district_name = str(item.get("sggName", "") or "").strip()
        region_id = REGION_IDS.get(region_name)
        if region_id is None or municipality != "합계" or not district_name:
            continue
        district_key = (region_id, district_name)
        if district_key in seen_districts:
            continue
        seen_districts.add(district_key)
        candidate_rows = count_item_candidate_rows(item)
        valid_votes = sum(float(row["votes"]) for row in candidate_rows)
        if valid_votes <= 0.0:
            continue
        winner_votes = max(float(row["votes"]) for row in candidate_rows)
        for candidate in candidate_rows:
            party = str(candidate.get("party", "") or "").strip()
            candidate_name = str(candidate.get("candidate", "") or "").strip()
            votes = float(candidate.get("votes", 0.0) or 0.0)
            bloc = "무소속" if party in INDEPENDENT_LABELS else normalize_bloc(party)
            rows.append(
                {
                    "election_id": election_id,
                    "election_date": source_date.date().isoformat(),
                    "region_id": region_id,
                    "region_name": region_name,
                    "district_name": district_name,
                    "party_name": party or "무소속",
                    "bloc": bloc,
                    "candidate_name": candidate_name,
                    "candidate_votes": votes,
                    "district_valid_votes": valid_votes,
                    "candidate_vote_share": votes / valid_votes,
                    "candidate_won": bool(votes == winner_votes),
                    "available_date": (source_date + timedelta(days=1)).date().isoformat(),
                    "source_type": "nec_assembly_constituency_count",
                    "derivation_version": SCHEMA_VERSION,
                }
            )
    columns = [
        "election_id",
        "election_date",
        "region_id",
        "region_name",
        "district_name",
        "party_name",
        "bloc",
        "candidate_name",
        "candidate_votes",
        "district_valid_votes",
        "candidate_vote_share",
        "candidate_won",
        "available_date",
        "source_type",
        "derivation_version",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["election_date", "region_id", "district_name", "candidate_votes"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

