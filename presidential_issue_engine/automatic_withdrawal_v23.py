"""Outcome-free candidate profiles and withdrawal-event compilation.

The raw event registry contains facts only. Candidate political traits are
compiled once from dated structural evidence, while transfer values come from
one universal low/medium/high scenario policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import exp, sqrt

import numpy as np
import pandas as pd

from presidential_issue_engine.preliminary_slots import (
    attenuate_withdrawn_endorsement_transfer,
)


PROFILE_FIELDS = (
    "viability",
    "centrist_appeal",
    "anti_major_party_appeal",
    "regional_base_overlap",
)
TRAIT_FIELDS = ("centrist_appeal", "anti_major_party_appeal")
NEUTRAL_TRAIT = 0.50
TRAIT_HALF_LIFE_YEARS = 12.0
SCENARIOS = {
    "low": {
        "coalition_rate": 0.45,
        "coalition_compliance": 0.60,
        "official_rate": 0.45,
        "official_compliance": 0.60,
        "alternative_rate": 0.15,
        "alternative_compliance": 0.50,
        "official_confidence": 0.60,
        "alternative_confidence": 0.50,
    },
    "medium": {
        "coalition_rate": 0.55,
        "coalition_compliance": 0.70,
        "official_rate": 0.55,
        "official_compliance": 0.70,
        "alternative_rate": 0.25,
        "alternative_compliance": 0.55,
        "official_confidence": 0.70,
        "alternative_confidence": 0.60,
    },
    "high": {
        "coalition_rate": 0.85,
        "coalition_compliance": 0.85,
        "official_rate": 0.85,
        "official_compliance": 0.85,
        "alternative_rate": 0.0,
        "alternative_compliance": 0.0,
        "official_confidence": 0.75,
        "alternative_confidence": 0.50,
    },
}


def _bounded(value: object, default: float = 0.0) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(np.clip(default if pd.isna(numeric) else numeric, 0.0, 1.0))


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _alias_maps(aliases: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    required = {"candidate_id", "canonical_name", "alias"}
    if aliases.empty or not required.issubset(aliases.columns):
        return {}, {}
    alias_to_id = {
        str(row.alias).strip().casefold(): str(row.candidate_id).strip()
        for row in aliases.itertuples(index=False)
    }
    canonical = (
        aliases[["candidate_id", "canonical_name"]]
        .drop_duplicates("candidate_id")
        .set_index("candidate_id")["canonical_name"]
        .astype(str)
        .to_dict()
    )
    return alias_to_id, canonical


def _candidate_id(name: object, alias_to_id: Mapping[str, str]) -> str:
    normalized = str(name).strip().casefold()
    return str(alias_to_id.get(normalized, normalized.replace(" ", "_")))


def _election_timestamp(election_id: str, election_dates: Mapping[str, object]) -> pd.Timestamp:
    # Korean presidential polling opens at 06:00. The hour matters only for
    # the generic reach calculation of a dated support-withdrawal event.
    return pd.Timestamp(election_dates[election_id]) + pd.Timedelta(hours=6)


def _event_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert("Asia/Seoul").tz_localize(None)


def _profile_date(row: pd.Series, election_dates: Mapping[str, object]) -> pd.Timestamp:
    available = pd.to_datetime(row.get("available_date"), errors="coerce")
    if pd.notna(available):
        return available
    return pd.Timestamp(election_dates.get(str(row.get("election_id"))))


def _prior_trait_profile(
    candidate_id: str,
    cutoff: pd.Timestamp,
    derived: pd.DataFrame,
) -> pd.Series | None:
    eligible = derived.loc[
        derived["candidate_id"].astype(str).eq(candidate_id)
        & derived["evidence_date"].lt(cutoff)
    ].sort_values("evidence_date")
    return None if eligible.empty else eligible.iloc[-1]


def _shrunken_trait(value: object, evidence_weight: float) -> float:
    raw = _bounded(value, NEUTRAL_TRAIT)
    return float(NEUTRAL_TRAIT + (raw - NEUTRAL_TRAIT) * np.clip(evidence_weight, 0.0, 1.0))


def _prior_assembly_profile(
    candidate_id: str,
    cutoff: pd.Timestamp,
    assembly_history: pd.DataFrame,
    alias_to_id: Mapping[str, str],
) -> dict[str, object] | None:
    """Derive a conservative candidate profile from strictly prior NEC races."""

    if assembly_history.empty or "candidate_name" not in assembly_history:
        return None
    work = assembly_history.copy()
    work["candidate_id"] = work["candidate_name"].map(
        lambda value: _candidate_id(value, alias_to_id)
    )
    work["evidence_date"] = pd.to_datetime(
        work.get("available_date", work.get("event_date")), errors="coerce"
    )
    work = work.loc[
        work["candidate_id"].astype(str).eq(candidate_id)
        & work["evidence_date"].notna()
        & work["evidence_date"].lt(cutoff)
    ].copy()
    if work.empty:
        return None

    years = (cutoff - work["evidence_date"]).dt.days.clip(lower=0) / 365.25
    weights = np.exp(-years.to_numpy(float) / TRAIT_HALF_LIFE_YEARS)
    evidence_strength = float(1.0 - exp(-float(weights.sum()) / 2.0))
    if "candidate_vote_share" in work.columns:
        shares = pd.to_numeric(
            work["candidate_vote_share"], errors="coerce"
        ).fillna(0.0)
    elif "vote_share" in work.columns:
        shares = pd.to_numeric(work["vote_share"], errors="coerce").fillna(0.0)
    else:
        candidate_votes = pd.to_numeric(
            work.get("candidate_votes", pd.Series(0.0, index=work.index)),
            errors="coerce",
        ).fillna(0.0)
        valid_votes = pd.to_numeric(
            work.get(
                "district_valid_votes",
                work.get("valid_votes", pd.Series(0.0, index=work.index)),
            ),
            errors="coerce",
        ).fillna(0.0)
        shares = candidate_votes.div(valid_votes.where(valid_votes.gt(0.0))).fillna(0.0)
    won = (
        work.get(
            "candidate_won",
            work.get("won", pd.Series(False, index=work.index)),
        )
        .astype(str)
        .str.casefold()
        .isin({"1", "true", "yes", "y"})
        .astype(float)
    )
    personal_share = float(np.average(shares, weights=weights))
    win_rate = float(np.average(won, weights=weights))
    raw_viability = float(np.clip(0.65 * personal_share + 0.35 * win_rate, 0.0, 1.0))
    viability = _shrunken_trait(raw_viability, evidence_strength)

    party = work.get("party_name", pd.Series("", index=work.index)).astype(str)
    major_pattern = "한나라|국민의힘|새누리|민주당|새천년민주|더불어민주"
    nonmajor = (~party.str.contains(major_pattern, regex=True, na=False)).astype(float)
    nonmajor_share = float(np.average(nonmajor, weights=weights))
    centrist = _shrunken_trait(0.50 + 0.25 * nonmajor_share, evidence_strength)
    anti_major = _shrunken_trait(0.50 + 0.15 * nonmajor_share, evidence_strength)

    region = work.get("region_id", pd.Series("", index=work.index)).astype(str)
    region_mass = pd.Series(weights, index=work.index).groupby(region).sum()
    region_distribution = region_mass / max(float(region_mass.sum()), 1e-12)
    concentration = float(np.square(region_distribution.to_numpy(float)).sum())
    regional_overlap = float(np.clip(0.50 * concentration * evidence_strength, 0.0, 1.0))
    confidence = float(np.clip(0.25 + 0.50 * evidence_strength, 0.25, 0.75))
    return {
        "viability": viability,
        "centrist_appeal": centrist,
        "anti_major_party_appeal": anti_major,
        "regional_base_overlap": regional_overlap,
        "confidence": confidence,
        "available_date": work["evidence_date"].max().date().isoformat(),
        "evidence_weight": evidence_strength,
        "evidence_rows": int(len(work)),
    }


def _candidate_attention_profile(
    election_id: str,
    candidate_id: str,
    cutoff: pd.Timestamp,
    attention_history: pd.DataFrame,
) -> dict[str, object] | None:
    """Estimate only pre-withdrawal stature from target-mention breadth."""

    required = {
        "election_id",
        "candidate_id",
        "unique_sentence_count",
        "last_evidence_date",
    }
    if attention_history.empty or not required.issubset(attention_history.columns):
        return None
    work = attention_history.loc[
        attention_history["election_id"].astype(str).eq(election_id)
    ].copy()
    work["last_evidence_date"] = pd.to_datetime(
        work["last_evidence_date"], errors="coerce"
    )
    work = work.loc[
        work["last_evidence_date"].notna()
        & work["last_evidence_date"].lt(cutoff)
    ].copy()
    selected = work.loc[work["candidate_id"].astype(str).eq(candidate_id)]
    if selected.empty:
        return None
    counts = pd.to_numeric(work["unique_sentence_count"], errors="coerce").fillna(0.0)
    reference = float(counts.max())
    count = _bounded(
        float(pd.to_numeric(selected.iloc[-1]["unique_sentence_count"], errors="coerce"))
        / max(reference, 1.0)
    )
    raw_viability = sqrt(count)
    evidence_rows = float(
        pd.to_numeric(selected.iloc[-1]["unique_sentence_count"], errors="coerce")
    )
    evidence_weight = float(1.0 - exp(-max(evidence_rows, 0.0) / 50.0))
    return {
        "viability": _shrunken_trait(raw_viability, evidence_weight),
        "confidence": float(np.clip(0.25 + 0.50 * evidence_weight, 0.25, 0.75)),
        "available_date": selected.iloc[-1]["last_evidence_date"].date().isoformat(),
        "evidence_weight": evidence_weight,
        "evidence_rows": int(evidence_rows),
    }


def build_unified_candidate_profiles(
    active_profile: pd.DataFrame,
    election_derived_profile: pd.DataFrame,
    preliminary_profile: pd.DataFrame,
    preliminary_assignments: pd.DataFrame,
    withdrawal_events: pd.DataFrame,
    aliases: pd.DataFrame,
    election_dates: Mapping[str, object],
    assembly_history: pd.DataFrame | None = None,
    candidate_attention_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one political profile source for ballot and withdrawn candidates."""

    alias_to_id, canonical_names = _alias_maps(aliases)
    assembly_history = pd.DataFrame() if assembly_history is None else assembly_history
    candidate_attention_history = (
        pd.DataFrame()
        if candidate_attention_history is None
        else candidate_attention_history
    )
    derived = election_derived_profile.copy()
    derived["candidate_id"] = derived["candidate_name"].map(
        lambda value: _candidate_id(value, alias_to_id)
    )
    derived["evidence_date"] = pd.to_datetime(
        derived.apply(lambda row: _profile_date(row, election_dates), axis=1),
        errors="coerce",
    )
    preliminary = preliminary_profile.copy()
    if not preliminary.empty:
        preliminary["candidate_id"] = preliminary["candidate_name"].map(
            lambda value: _candidate_id(value, alias_to_id)
        )
    assignment_c = preliminary_assignments.loc[
        preliminary_assignments.get(
            "source_slot", pd.Series("", index=preliminary_assignments.index)
        ).astype(str).eq("C")
    ].copy()
    if not assignment_c.empty:
        assignment_c["candidate_id"] = assignment_c["candidate_name"].map(
            lambda value: _candidate_id(value, alias_to_id)
        )

    initial_events = withdrawal_events.loc[
        withdrawal_events["event_type"].astype(str).eq("coalition_withdrawal")
    ].copy()
    withdrawn_keys = {
        (str(row.election_id), str(row.candidate_id))
        for row in initial_events.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    for source in active_profile.to_dict(orient="records"):
        election_id = str(source["election_id"])
        slot = str(source["slot"])
        candidate_id = _candidate_id(source["candidate_name"], alias_to_id)
        cutoff = pd.Timestamp(election_dates[election_id])
        matching_event = initial_events.loc[
            initial_events["election_id"].astype(str).eq(election_id)
            & initial_events["candidate_id"].astype(str).eq(candidate_id)
        ]
        evidence_cutoff = (
            pd.Timestamp(matching_event.iloc[0]["available_date"])
            if not matching_event.empty
            else cutoff
        )
        attention_profile = _candidate_attention_profile(
            election_id,
            candidate_id,
            evidence_cutoff,
            candidate_attention_history,
        )
        exact = derived.loc[
            derived["election_id"].astype(str).eq(election_id)
            & derived["slot"].astype(str).eq(slot)
        ]
        current_preliminary = preliminary.loc[
            preliminary["election_id"].astype(str).eq(election_id)
            & preliminary["slot"].astype(str).eq(slot)
            & preliminary["candidate_id"].astype(str).eq(candidate_id)
        ] if not preliminary.empty else pd.DataFrame()
        current_assignment = assignment_c.loc[
            assignment_c["election_id"].astype(str).eq(election_id)
            & assignment_c["source_slot"].astype(str).eq(slot)
            & assignment_c["candidate_id"].astype(str).eq(candidate_id)
        ] if not assignment_c.empty else pd.DataFrame()

        output = dict(source)
        trait_source = "universal_neutral_fallback"
        evidence_weight = 0.0
        if not exact.empty:
            selected = exact.sort_values("evidence_date").iloc[-1]
            candidate_id = str(selected["candidate_id"])
            for field in PROFILE_FIELDS:
                output[field] = _bounded(selected[field])
            output["candidate_name"] = canonical_names.get(candidate_id, str(selected["candidate_name"]))
            output["confidence"] = _bounded(selected["confidence"])
            output["available_date"] = pd.Timestamp(selected["evidence_date"]).date().isoformat()
            trait_source = "current_election_structural_evidence"
            evidence_weight = _bounded(selected["confidence"])
        elif not current_preliminary.empty:
            selected = current_preliminary.sort_values("available_date").iloc[-1]
            output["viability"] = _bounded(selected["viability"])
            output["regional_base_overlap"] = _bounded(selected["regional_base_overlap"])
            prior = _prior_trait_profile(candidate_id, cutoff, derived)
            if prior is not None:
                years = max((cutoff - prior["evidence_date"]).days / 365.25, 0.0)
                time_decay = exp(-years / TRAIT_HALF_LIFE_YEARS)
                evidence_weight = _bounded(prior["confidence"]) * time_decay
                for field in TRAIT_FIELDS:
                    output[field] = _shrunken_trait(prior[field], evidence_weight)
                trait_source = "strictly_prior_same_person_structural_evidence_shrunk"
            else:
                for field in TRAIT_FIELDS:
                    output[field] = NEUTRAL_TRAIT
            preliminary_confidence = _bounded(selected["confidence"], 0.25)
            output["confidence"] = float(
                np.clip(np.sqrt(preliminary_confidence * max(evidence_weight, 0.05)), 0.20, 0.75)
            )
            output["candidate_name"] = canonical_names.get(candidate_id, str(selected["candidate_name"]))
            output["available_date"] = str(selected["available_date"])
        elif attention_profile is not None:
            output["viability"] = float(attention_profile["viability"])
            output["centrist_appeal"] = NEUTRAL_TRAIT
            output["anti_major_party_appeal"] = NEUTRAL_TRAIT
            output["regional_base_overlap"] = 0.0
            output["confidence"] = float(attention_profile["confidence"])
            output["candidate_name"] = canonical_names.get(
                candidate_id, str(source["candidate_name"])
            )
            output["available_date"] = str(attention_profile["available_date"])
            trait_source = "strictly_prior_assembly_attention_stature_shrunk"
            evidence_weight = float(attention_profile["evidence_weight"])
        else:
            output["viability"] = NEUTRAL_TRAIT
            output["centrist_appeal"] = NEUTRAL_TRAIT
            output["anti_major_party_appeal"] = NEUTRAL_TRAIT
            output["regional_base_overlap"] = 0.0
            output["confidence"] = 0.25
            output["candidate_name"] = canonical_names.get(candidate_id, str(source["candidate_name"]))
            output["available_date"] = (
                str(matching_event.iloc[0]["available_date"])
                if not matching_event.empty
                else (cutoff - pd.Timedelta(days=1)).date().isoformat()
            )
            trait_source = "universal_neutral_seed"

        role = "withdrawn_preliminary" if (election_id, candidate_id) in withdrawn_keys else "final_third"
        output.update(
            {
                "candidate_id": candidate_id,
                "candidate_role": role,
                "notes": f"Automatic V23 profile; source={trait_source}; target outcome unused",
            }
        )
        rows.append(output)
        audits.append(
            {
                "election_id": election_id,
                "slot": slot,
                "candidate_id": candidate_id,
                "candidate_name": output["candidate_name"],
                "candidate_role": role,
                "profile_source": trait_source,
                "trait_evidence_weight": evidence_weight,
                "centrist_appeal": output["centrist_appeal"],
                "anti_major_party_appeal": output["anti_major_party_appeal"],
                "target_outcome_used": False,
            }
        )

    for event in initial_events.itertuples(index=False):
        election_id = str(event.election_id)
        candidate_id = str(event.candidate_id)
        existing = any(
            str(row["election_id"]) == election_id
            and str(row["candidate_id"]) == candidate_id
            and str(row["candidate_role"]) == "withdrawn_preliminary"
            for row in rows
        )
        if existing:
            continue
        assignment = (
            assignment_c.loc[
                assignment_c["election_id"].astype(str).eq(election_id)
                & assignment_c["candidate_id"].astype(str).eq(candidate_id)
            ]
            if not assignment_c.empty
            else pd.DataFrame()
        )
        cutoff = pd.Timestamp(event.available_date)
        prior = _prior_trait_profile(candidate_id, cutoff, derived)
        assembly_profile = _prior_assembly_profile(
            candidate_id, cutoff, assembly_history, alias_to_id
        )
        attention_profile = _candidate_attention_profile(
            election_id,
            candidate_id,
            cutoff,
            candidate_attention_history,
        )
        evidence_weight = 0.0
        centrist = NEUTRAL_TRAIT
        anti_major = NEUTRAL_TRAIT
        regional = 0.0
        source_name = "universal_neutral_fallback"
        if prior is not None:
            years = max((cutoff - prior["evidence_date"]).days / 365.25, 0.0)
            evidence_weight = _bounded(prior["confidence"]) * exp(
                -years / TRAIT_HALF_LIFE_YEARS
            )
            centrist = _shrunken_trait(prior["centrist_appeal"], evidence_weight)
            anti_major = _shrunken_trait(prior["anti_major_party_appeal"], evidence_weight)
            regional = _bounded(prior["regional_base_overlap"]) * evidence_weight
            source_name = "strictly_prior_same_person_structural_evidence_shrunk"
        elif assembly_profile is not None:
            evidence_weight = float(assembly_profile["evidence_weight"])
            centrist = float(assembly_profile["centrist_appeal"])
            anti_major = float(assembly_profile["anti_major_party_appeal"])
            regional = float(assembly_profile["regional_base_overlap"])
            source_name = "strictly_prior_candidate_election_history_shrunk"
        elif attention_profile is not None:
            evidence_weight = float(attention_profile["evidence_weight"])
            source_name = "strictly_prior_assembly_attention_stature_shrunk"
        available_dates = [str(event.available_date)]
        if not assignment.empty:
            available_dates.append(str(assignment.iloc[-1]["available_date"]))
        output = {
            "election_id": election_id,
            "slot": str(event.source_slot),
            "candidate_id": candidate_id,
            "candidate_name": canonical_names.get(candidate_id, str(event.candidate_name)),
            "candidate_role": "withdrawn_preliminary",
            "viability": (
                float(
                    np.clip(
                        _bounded(assignment.iloc[-1]["pre_withdrawal_mean_share"])
                        / 0.25,
                        0.0,
                        1.0,
                    )
                )
                if prior is not None and not assignment.empty
                else (
                    float(assembly_profile["viability"])
                    if assembly_profile is not None
                    else (
                        float(attention_profile["viability"])
                        if attention_profile is not None
                        else NEUTRAL_TRAIT
                    )
                )
            ),
            "centrist_appeal": centrist,
            "anti_major_party_appeal": anti_major,
            "regional_base_overlap": regional,
            "available_date": max(
                pd.to_datetime(
                    available_dates
                    + (
                        [str(assembly_profile["available_date"])]
                        if assembly_profile is not None
                        else (
                            [str(attention_profile["available_date"])]
                            if attention_profile is not None
                            else []
                        )
                    )
                )
            ).date().isoformat(),
            "confidence": (
                float(assembly_profile["confidence"])
                if assembly_profile is not None
                else (
                    float(attention_profile["confidence"])
                    if attention_profile is not None
                    else float(np.clip(max(evidence_weight, 0.25), 0.25, 0.65))
                )
            ),
            "notes": f"Automatic V23 withdrawn profile; source={source_name}; target outcome unused",
        }
        rows.append(output)
        audits.append(
            {
                "election_id": election_id,
                "slot": str(event.source_slot),
                "candidate_id": candidate_id,
                "candidate_name": output["candidate_name"],
                "candidate_role": "withdrawn_preliminary",
                "profile_source": source_name,
                "trait_evidence_weight": evidence_weight,
                "centrist_appeal": centrist,
                "anti_major_party_appeal": anti_major,
                "target_outcome_used": False,
            }
        )

    output = pd.DataFrame(rows)
    output = output[
        [
            "election_id",
            "slot",
            "candidate_id",
            "candidate_name",
            "candidate_role",
            *PROFILE_FIELDS,
            "available_date",
            "confidence",
            "notes",
        ]
    ].sort_values(["election_id", "slot", "candidate_role", "candidate_id"])
    return output.reset_index(drop=True), pd.DataFrame(audits)


def build_candidate_landscape_from_profiles(
    base_landscape: pd.DataFrame,
    profiles: pd.DataFrame,
    aliases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace every withdrawn political vector from the canonical profile."""

    alias_to_id, _ = _alias_maps(aliases)
    out = base_landscape.copy()
    out["candidate_id"] = out["candidate_name"].map(
        lambda value: _candidate_id(value, alias_to_id)
    )
    withdrawn = profiles.loc[
        profiles["candidate_role"].astype(str).eq("withdrawn_preliminary")
    ].set_index(["election_id", "candidate_id"])
    audit: list[dict[str, object]] = []
    for index in out.index[
        out["candidate_role"].astype(str).str.contains("withdraw", case=False, na=False)
    ]:
        key = (str(out.at[index, "election_id"]), str(out.at[index, "candidate_id"]))
        if key not in withdrawn.index:
            continue
        profile = withdrawn.loc[key]
        centrist = _bounded(profile["centrist_appeal"], NEUTRAL_TRAIT)
        anti_major = _bounded(profile["anti_major_party_appeal"], NEUTRAL_TRAIT)
        regional = _bounded(profile["regional_base_overlap"])
        ideology_mass = max(1.0 - centrist, 0.0)
        values = {
            "conservative": ideology_mass / 2.0,
            "liberal": ideology_mass / 2.0,
            "progressive": ideology_mass / 4.0,
            "centrist": centrist,
            "anti_establishment": anti_major,
            "reform": (centrist + anti_major) / 2.0,
            "regionalist": regional,
            "confidence": min(_bounded(profile["confidence"]), 0.50),
            "available_date": str(profile["available_date"]),
            "candidate_name": str(profile["candidate_name"]),
            "notes": "Automatic V23 withdrawn landscape from canonical candidate profile",
        }
        for column, value in values.items():
            out.at[index, column] = value
        audit.append(
            {
                "election_id": key[0],
                "candidate_id": key[1],
                "candidate_name": values["candidate_name"],
                "target_outcome_used": False,
            }
        )
    existing_keys = set(
        zip(out["election_id"].astype(str), out["candidate_id"].astype(str))
    )
    additions: list[dict[str, object]] = []
    for profile in profiles.loc[
        profiles["candidate_role"].astype(str).eq("withdrawn_preliminary")
    ].itertuples(index=False):
        key = (str(profile.election_id), str(profile.candidate_id))
        if key in existing_keys:
            continue
        centrist = _bounded(profile.centrist_appeal, NEUTRAL_TRAIT)
        anti_major = _bounded(profile.anti_major_party_appeal, NEUTRAL_TRAIT)
        regional = _bounded(profile.regional_base_overlap)
        ideology_mass = max(1.0 - centrist, 0.0)
        additions.append(
            {
                "election_id": str(profile.election_id),
                "slot": str(profile.slot),
                "candidate_name": str(profile.candidate_name),
                "candidate_role": "withdrawn",
                "conservative": ideology_mass / 2.0,
                "liberal": ideology_mass / 2.0,
                "progressive": ideology_mass / 4.0,
                "centrist": centrist,
                "anti_establishment": anti_major,
                "reform": (centrist + anti_major) / 2.0,
                "regionalist": regional,
                "available_date": str(profile.available_date),
                "confidence": min(_bounded(profile.confidence), 0.50),
                "notes": "Automatic V23 withdrawn landscape from canonical candidate profile",
                "candidate_id": str(profile.candidate_id),
            }
        )
        audit.append(
            {
                "election_id": key[0],
                "candidate_id": key[1],
                "candidate_name": str(profile.candidate_name),
                "target_outcome_used": False,
            }
        )
    if additions:
        out = pd.concat([out, pd.DataFrame(additions)], ignore_index=True, sort=False)
    return out.drop(columns="candidate_id"), pd.DataFrame(audit)


def _scenario(days_to_election: float, formal_endorsement: bool) -> str:
    if not formal_endorsement or days_to_election < 3.0:
        return "low"
    if days_to_election < 14.0:
        return "medium"
    return "high"


def compile_withdrawal_transfer_registry(
    events: pd.DataFrame,
    profiles: pd.DataFrame,
    election_dates: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compile one prediction registry from facts, profiles, and common scenarios."""

    profile_lookup = profiles.loc[
        profiles["candidate_role"].astype(str).eq("withdrawn_preliminary")
    ].set_index(["election_id", "candidate_id"])
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    initial = events.loc[events["event_type"].astype(str).eq("coalition_withdrawal")]
    for event in initial.itertuples(index=False):
        election_id = str(event.election_id)
        key = (election_id, str(event.candidate_id))
        if election_id not in election_dates or key not in profile_lookup.index:
            continue
        profile = profile_lookup.loc[key]
        election_time = _election_timestamp(election_id, election_dates)
        event_time = _event_timestamp(event.event_timestamp)
        if pd.isna(event_time) or event_time > election_time:
            continue
        days = max((election_time - event_time).total_seconds() / 86400.0, 0.0)
        scenario = _scenario(days, _as_bool(event.formal_endorsement))
        policy = SCENARIOS[scenario]
        confidence = _bounded(profile["confidence"], 0.25)
        official_target = str(event.target_slot)
        alternatives = [slot for slot in ["A", "B"] if slot != official_target]

        reversals = events.loc[
            events["election_id"].astype(str).eq(election_id)
            & events["candidate_id"].astype(str).eq(str(event.candidate_id))
            & events["event_type"].astype(str).eq("coalition_support_withdrawal")
        ].sort_values("event_timestamp")
        support_retention = 1.0
        reversal_reach = 0.0
        registry_available = max(
            pd.Timestamp(event.available_date), pd.Timestamp(profile["available_date"])
        )
        preliminary_fractions = {
            official_target: float(policy["official_rate"])
            * float(policy["official_compliance"])
        }
        if float(policy["alternative_rate"]) > 0.0:
            for target in alternatives:
                preliminary_fractions[target] = (
                    float(policy["alternative_rate"])
                    * float(policy["alternative_compliance"])
                )
        for reversal in reversals.itertuples(index=False):
            reversal_time = _event_timestamp(reversal.event_timestamp)
            if pd.isna(reversal_time) or reversal_time > election_time:
                continue
            hours = max((election_time - reversal_time).total_seconds() / 3600.0, 0.0)
            reversal_reach = float(1.0 - exp(-hours / 24.0))
            adjusted, retention = attenuate_withdrawn_endorsement_transfer(
                preliminary_fractions,
                target_slot=str(reversal.target_slot),
                event_strength=1.0,
                voter_reach=reversal_reach,
                days_to_election=hours / 24.0,
            )
            preliminary_fractions = adjusted
            support_retention *= retention
            registry_available = max(registry_available, pd.Timestamp(reversal.available_date))

        # A late reversal is represented only in the preliminary transfer layer.
        # Applying the stale endorsement again in the two downstream layers
        # would count the same event twice.
        downstream_enabled = reversals.empty
        target_rows = [official_target]
        if float(policy["alternative_rate"]) > 0.0:
            target_rows.extend(alternatives)
        for target_slot in target_rows:
            is_official = target_slot == official_target
            target_rate = float(
                policy["official_rate"] if is_official else policy["alternative_rate"]
            )
            target_compliance = float(
                policy["official_compliance"]
                if is_official
                else policy["alternative_compliance"]
            )
            target_confidence = float(
                policy["official_confidence"]
                if is_official
                else policy["alternative_confidence"]
            )
            preliminary_fraction = float(preliminary_fractions.get(target_slot, 0.0))
            rows.append(
                {
                    "event_id": str(event.event_id),
                    "election_id": election_id,
                    "candidate_id": str(event.candidate_id),
                    "candidate_name": str(profile["candidate_name"]),
                    "source_slot": str(event.source_slot),
                    "target_slot": target_slot,
                    "event_type": "coalition_withdrawal",
                    "event_date": event_time.date().isoformat(),
                    "available_date": registry_available.date().isoformat(),
                    "formal_endorsement": _as_bool(event.formal_endorsement),
                    "is_official_target": is_official,
                    "scenario_level": scenario,
                    "days_to_election": days,
                    "viability": _bounded(profile["viability"]),
                    "centrist_appeal": _bounded(profile["centrist_appeal"], NEUTRAL_TRAIT),
                    "anti_major_party_appeal": _bounded(profile["anti_major_party_appeal"], NEUTRAL_TRAIT),
                    "regional_base_overlap": _bounded(profile["regional_base_overlap"]),
                    "profile_confidence": confidence,
                    "use_in_coalition_layer": downstream_enabled and is_official,
                    "use_in_withdrawn_feature_layer": downstream_enabled,
                    "use_in_preliminary_layer": True,
                    "coalition_transfer_rate": float(policy["coalition_rate"]),
                    "coalition_voter_compliance": float(policy["coalition_compliance"]),
                    "withdrawn_transfer_rate": target_rate,
                    "withdrawn_voter_compliance": target_compliance,
                    "withdrawn_confidence": target_confidence,
                    "preliminary_transfer_rate": preliminary_fraction,
                    "preliminary_voter_compliance": 1.0,
                    "transfer_rate": preliminary_fraction,
                    "voter_compliance": 1.0,
                    "confidence": target_confidence,
                    "support_retention": support_retention,
                    "reversal_voter_reach": reversal_reach,
                    "source_viability_after_event": _bounded(event.source_viability_after_event),
                    "exclude_source_from_evaluation": _as_bool(event.exclude_source_from_evaluation),
                    "target_outcome_used": False,
                    "notes": "Single-registry common scenario; consumer-specific semantics are explicit",
                }
            )
        audits.append(
            {
                "event_id": str(event.event_id),
                "election_id": election_id,
                "candidate_id": str(event.candidate_id),
                "scenario_level": scenario,
                "days_to_election": days,
                "profile_confidence": confidence,
                "coalition_transfer_rate": float(policy["coalition_rate"]),
                "coalition_voter_compliance": float(policy["coalition_compliance"]),
                "official_transfer_rate": float(policy["official_rate"]),
                "official_voter_compliance": float(policy["official_compliance"]),
                "downstream_layers_enabled": downstream_enabled,
                "support_retention": support_retention,
                "reversal_voter_reach": reversal_reach,
                "target_outcome_used": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audits)
