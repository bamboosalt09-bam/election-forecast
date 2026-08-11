"""Point-in-time candidate history and regional-base derivation.

Only facts published by the NEC are accepted here. Target-election outcomes are
masked, and only records dated strictly before the forecast election can create
regional features.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import pandas as pd

from election_forecast.features.region_bloc_prior import PRESIDENTIAL_ELECTION_DATES
from presidential_issue_engine.build_bloc_history_from_nec import REGION_IDS


SCHEMA_VERSION = "official_candidate_history_v1"
REGIONAL_BASE_SCHEMA_VERSION = "official_candidate_regional_base_v1"

HISTORY_COLUMNS = [
    "target_election_id",
    "target_election_date",
    "target_slot",
    "target_candidate_name",
    "target_party_name",
    "person_key",
    "birthday",
    "source_sg_id",
    "source_election_date",
    "source_election_name",
    "source_sg_typecode",
    "source_candidate_id",
    "source_region_name",
    "source_district_name",
    "source_municipality_name",
    "source_party_name",
    "source_status",
    "source_job",
    "source_career1",
    "source_career2",
    "source_is_prior",
    "prior_election_won",
    "available_date",
    "entity_match_method",
    "entity_match_confidence",
    "source_url",
    "source_record_sha256",
    "derivation_version",
]

REGIONAL_BASE_COLUMNS = [
    "election_id",
    "slot",
    "candidate_name",
    "region_id",
    "regional_affinity",
    "organization_depth",
    "available_date",
    "confidence",
    "source_type",
    "notes",
    "source_election_ids",
    "source_election_types",
    "source_record_count",
    "evidence_score",
    "entity_match_confidence",
    "provenance_class",
    "derivation_version",
]


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", _clean(value))


def _election_date_from_id(value: Any) -> pd.Timestamp | None:
    text = re.sub(r"\D", "", _clean(value))
    if len(text) != 8:
        return None
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _target_date(election_id: str) -> pd.Timestamp:
    match = re.search(r"(19|20)\d{2}", election_id)
    if match is None:
        raise ValueError(f"cannot infer presidential election year: {election_id}")
    year = int(match.group(0))
    date_text = PRESIDENTIAL_ELECTION_DATES.get(year)
    if date_text is None:
        raise ValueError(f"unknown presidential election date: {election_id}")
    return pd.Timestamp(date_text)


def build_candidate_reference(results: pd.DataFrame) -> pd.DataFrame:
    """Build identity-only references without reading votes or vote shares."""

    required = {"election_id", "slot", "candidate_name", "party_name"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"candidate reference is missing columns: {sorted(missing)}")
    reference = results[
        ["election_id", "slot", "candidate_name", "party_name"]
    ].copy()
    reference = reference.loc[
        reference["slot"].astype(str).ne("alpha")
        & reference["candidate_name"].astype(str).str.strip().ne("")
    ]
    reference = reference.drop_duplicates().reset_index(drop=True)
    reference["target_election_date"] = reference["election_id"].map(
        lambda election_id: _target_date(str(election_id)).date().isoformat()
    )
    return reference[
        [
            "election_id",
            "target_election_date",
            "slot",
            "candidate_name",
            "party_name",
        ]
    ].sort_values(["target_election_date", "slot"])


def _record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_birthday(
    records: list[dict[str, Any]],
    *,
    candidate_name: str,
    party_name: str,
    target_date: pd.Timestamp,
) -> tuple[str | None, str, float]:
    exact_name = [
        record
        for record in records
        if _compact(record.get("name")) == _compact(candidate_name)
    ]
    target_records = [
        record
        for record in exact_name
        if _election_date_from_id(record.get("sgId")) == target_date
    ]
    party_matches = [
        record
        for record in target_records
        if _compact(record.get("jdName")) == _compact(party_name)
    ]
    anchor = party_matches or target_records
    birthdays = {_clean(record.get("birthday")) for record in anchor}
    birthdays.discard("")
    if len(birthdays) == 1:
        method = "target_election_party_birthday" if party_matches else "target_election_birthday"
        confidence = 1.0 if party_matches else 0.9
        return next(iter(birthdays)), method, confidence

    historical_party_matches = [
        record
        for record in exact_name
        if _compact(record.get("jdName")) == _compact(party_name)
        and (
            (record_date := _election_date_from_id(record.get("sgId"))) is not None
            and record_date <= target_date
        )
    ]
    birthdays = {_clean(record.get("birthday")) for record in historical_party_matches}
    birthdays.discard("")
    if len(birthdays) == 1:
        return next(iter(birthdays)), "historical_party_birthday", 0.75
    return None, "unresolved_same_name", 0.0


def resolve_candidate_history(
    reference: pd.DataFrame,
    records_by_name: dict[str, list[dict[str, Any]]],
    *,
    source_url: str,
    max_source_date: str | pd.Timestamp = "2022-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve same-name records and mask target/future outcomes.

    Returns ``(history, resolution_audit)``. Unresolved identities are never
    guessed and therefore produce no feature-bearing history rows.
    """

    maximum = pd.Timestamp(max_source_date)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for ref in reference.itertuples(index=False):
        election_id = str(ref.election_id)
        target_date = pd.Timestamp(ref.target_election_date)
        name = _clean(ref.candidate_name)
        party = _clean(ref.party_name)
        records = records_by_name.get(name, [])
        birthday, method, match_confidence = _resolve_birthday(
            records,
            candidate_name=name,
            party_name=party,
            target_date=target_date,
        )
        matched = [
            record
            for record in records
            if birthday
            and _compact(record.get("name")) == _compact(name)
            and _clean(record.get("birthday")) == birthday
        ]
        audits.append(
            {
                "target_election_id": election_id,
                "slot": _clean(ref.slot),
                "candidate_name": name,
                "party_name": party,
                "query_record_count": len(records),
                "matched_record_count": len(matched),
                "birthday": birthday or "",
                "entity_match_method": method,
                "entity_match_confidence": match_confidence,
                "resolved": bool(birthday),
            }
        )
        if birthday is None:
            continue
        person_key = hashlib.sha256(f"{name}|{birthday}".encode("utf-8")).hexdigest()[:20]
        for record in matched:
            source_date = _election_date_from_id(record.get("sgId"))
            if source_date is None or source_date > maximum:
                continue
            is_prior = source_date < target_date
            won = _clean(record.get("elcoYn")).upper() if is_prior else ""
            # Prior election results become public after that election. For a
            # conservative PIT boundary, make them available the following day.
            available_date = (source_date + pd.Timedelta(days=1)).date().isoformat()
            rows.append(
                {
                    "target_election_id": election_id,
                    "target_election_date": target_date.date().isoformat(),
                    "target_slot": _clean(ref.slot),
                    "target_candidate_name": name,
                    "target_party_name": party,
                    "person_key": person_key,
                    "birthday": birthday,
                    "source_sg_id": _clean(record.get("sgId")),
                    "source_election_date": source_date.date().isoformat(),
                    "source_election_name": _clean(record.get("elctNm")),
                    "source_sg_typecode": _clean(record.get("sgTypecode")),
                    "source_candidate_id": _clean(record.get("huboid")),
                    "source_region_name": _clean(record.get("sdName")),
                    "source_district_name": _clean(record.get("sggName")),
                    "source_municipality_name": _clean(record.get("wiwName")),
                    "source_party_name": _clean(record.get("jdName")),
                    "source_status": _clean(record.get("status")),
                    "source_job": _clean(record.get("job")),
                    "source_career1": _clean(record.get("career1")),
                    "source_career2": _clean(record.get("career2")),
                    "source_is_prior": is_prior,
                    "prior_election_won": won,
                    "available_date": available_date,
                    "entity_match_method": method,
                    "entity_match_confidence": match_confidence,
                    "source_url": source_url,
                    "source_record_sha256": _record_hash(record),
                    "derivation_version": SCHEMA_VERSION,
                }
            )
    history = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if not history.empty:
        history = history.sort_values(
            ["target_election_date", "target_slot", "source_election_date"]
        ).reset_index(drop=True)
    audit = pd.DataFrame(audits).sort_values(
        ["target_election_id", "slot"]
    ).reset_index(drop=True)
    return history, audit


OFFICE_SCOPE_WEIGHTS = {
    "2": 0.75,  # constituency National Assembly
    "3": 0.95,  # metropolitan/provincial governor
    "4": 0.75,  # municipal head
    "5": 0.50,  # metropolitan/provincial council
    "6": 0.40,  # municipal council
    "10": 0.55,  # superintendent of education
    "11": 0.35,  # education council
}


def _office_weight(typecode: Any, election_name: Any) -> float:
    code = _clean(typecode)
    name = _clean(election_name)
    if code in {"1", "7", "8", "9"} or "대통령" in name or "비례" in name:
        return 0.0
    if code in OFFICE_SCOPE_WEIGHTS:
        return OFFICE_SCOPE_WEIGHTS[code]
    if "국회의원" in name:
        return 0.75
    if "지방" in name and "선거" in name:
        return 0.40
    return 0.0


def build_official_candidate_regional_base(
    history: pd.DataFrame,
    *,
    half_life_years: float = 12.0,
) -> pd.DataFrame:
    """Compile prior official candidacy history to candidate-region evidence."""

    if history.empty:
        return pd.DataFrame(columns=REGIONAL_BASE_COLUMNS)
    required = {
        "target_election_id",
        "target_election_date",
        "target_slot",
        "target_candidate_name",
        "source_election_date",
        "source_election_name",
        "source_sg_typecode",
        "source_sg_id",
        "source_region_name",
        "source_is_prior",
        "prior_election_won",
        "entity_match_confidence",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"official candidate history is missing columns: {sorted(missing)}")

    frame = history.copy()
    prior_mask = frame["source_is_prior"].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"1", "true", "yes", "y"}
    )
    frame = frame.loc[prior_mask].copy()
    frame["region_id"] = frame["source_region_name"].map(REGION_IDS)
    frame["office_weight"] = [
        _office_weight(typecode, election_name)
        for typecode, election_name in zip(
            frame["source_sg_typecode"], frame["source_election_name"], strict=False
        )
    ]
    frame = frame.loc[
        frame["region_id"].notna() & frame["office_weight"].gt(0.0)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=REGIONAL_BASE_COLUMNS)
    frame["target_date"] = pd.to_datetime(frame["target_election_date"])
    frame["source_date"] = pd.to_datetime(frame["source_election_date"])
    age_years = (frame["target_date"] - frame["source_date"]).dt.days / 365.2425
    frame["recency_weight"] = 0.5 ** (age_years / max(half_life_years, 1e-6))
    frame["prior_win_weight"] = frame["prior_election_won"].map(
        {"Y": 1.0, "N": 0.75}
    ).fillna(0.75)
    frame["record_evidence"] = (
        frame["office_weight"]
        * frame["recency_weight"]
        * frame["prior_win_weight"]
        * pd.to_numeric(frame["entity_match_confidence"], errors="coerce").fillna(0.0)
    )
    grouped = frame.groupby(
        [
            "target_election_id",
            "target_slot",
            "target_candidate_name",
            "region_id",
        ],
        as_index=False,
    ).agg(
        evidence_score=("record_evidence", "sum"),
        source_record_count=("record_evidence", "size"),
        entity_match_confidence=("entity_match_confidence", "min"),
        available_date=("available_date", "max"),
        source_election_ids=(
            "source_sg_id",
            lambda values: "|".join(sorted(set(map(str, values)))),
        ),
        source_election_types=(
            "source_sg_typecode",
            lambda values: "|".join(sorted(set(map(str, values)))),
        ),
    )
    grouped["regional_affinity"] = grouped["evidence_score"].map(
        lambda value: min(0.80, 1.0 - math.exp(-0.75 * float(value)))
    )
    grouped["organization_depth"] = [
        min(0.85, 1.0 - math.exp(-(0.50 * score + 0.15 * max(count - 1, 0))))
        for score, count in zip(
            grouped["evidence_score"], grouped["source_record_count"], strict=False
        )
    ]
    grouped["confidence"] = (
        pd.to_numeric(grouped["entity_match_confidence"], errors="coerce").fillna(0.0)
        * grouped["source_record_count"].map(lambda count: 1.0 - math.exp(-count / 2.0))
    ).clip(0.0, 1.0)
    grouped = grouped.rename(
        columns={
            "target_election_id": "election_id",
            "target_slot": "slot",
            "target_candidate_name": "candidate_name",
        }
    )
    grouped["source_type"] = "nec_prior_candidate_office_history"
    grouped["notes"] = (
        "Deterministic recency-weighted regional evidence from strictly prior "
        "NEC candidacy and office records; target-election outcomes excluded"
    )
    grouped["provenance_class"] = "official_deterministic_source_derived"
    grouped["derivation_version"] = REGIONAL_BASE_SCHEMA_VERSION
    return grouped[REGIONAL_BASE_COLUMNS].sort_values(
        ["election_id", "slot", "regional_affinity"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
