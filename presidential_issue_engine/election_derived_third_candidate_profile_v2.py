"""Third-candidate stature with separate party preference and organization evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import election_date, normalize_bloc
from presidential_issue_engine.election_derived_third_candidate_profile import (
    COMPETITIVE_BLOC_RATIO,
    _prior_presidential_competitiveness,
    _won_office_count,
)
from presidential_issue_engine.electorate_layers import LAYER_ELECTION_TYPE_WEIGHTS


DIRECT_PARTY_TYPES = frozenset(
    {
        "national_assembly_pr",
        "assembly_pr",
        "metro_council_pr",
        "local_council_pr",
    }
)
ORGANIZATION_TYPES = frozenset(
    {
        "national_assembly_district",
        "assembly_district",
        "metro_council_district",
        "local_council_district",
    }
)
ORGANIZATION_FALLBACK_SCALE = 0.50


def merge_automatic_viability(
    base_profile: pd.DataFrame,
    automatic_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace only viability, preserving separately sourced character traits."""

    keys = ["election_id", "slot"]
    replacement = automatic_profile[
        keys + ["candidate_name", "viability", "available_date", "confidence"]
    ].copy()
    replacement = replacement.rename(
        columns={
            "candidate_name": "automatic_candidate_name",
            "viability": "automatic_viability",
            "available_date": "automatic_available_date",
            "confidence": "automatic_confidence",
        }
    )
    if replacement.duplicated(keys).any():
        raise ValueError("automatic third profile has duplicate election-slot rows")
    out = base_profile.merge(replacement, on=keys, how="left", validate="one_to_one")
    matched = out["automatic_viability"].notna()
    original_viability = pd.to_numeric(out["viability"], errors="coerce")
    out.loc[matched, "viability"] = pd.to_numeric(
        out.loc[matched, "automatic_viability"], errors="coerce"
    )
    if "notes" in out.columns:
        out.loc[matched, "notes"] = (
            out.loc[matched, "notes"].fillna("").astype(str)
            + "; viability replaced by strictly prior election profile v2"
        )
    audit = out.loc[
        matched,
        keys
        + [
            "candidate_name",
            "automatic_candidate_name",
            "automatic_viability",
            "automatic_available_date",
            "automatic_confidence",
        ],
    ].copy()
    audit["original_viability"] = original_viability.loc[matched].to_numpy()
    audit["non_viability_fields_preserved"] = True
    audit["target_outcome_used"] = False
    out = out.drop(
        columns=[
            "automatic_candidate_name",
            "automatic_viability",
            "automatic_available_date",
            "automatic_confidence",
        ]
    )
    return out, audit


def _latest_family_competitiveness(
    history: pd.DataFrame,
    bloc: str,
    cutoff: pd.Timestamp,
    election_types: frozenset[str],
) -> tuple[float, int, str]:
    """Return a latest-date bloc ratio after region-level bloc aggregation.

    Election families held on the same day are scored separately. This avoids
    double counting their vote shares and prevents the number of raw party rows
    inside a normalized bloc from changing the estimate.
    """

    work = history.loc[
        history["election_type"].astype(str).isin(election_types)
    ].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(election_date), errors="coerce"
    )
    work = work.loc[work["event_date"].lt(cutoff)].copy()
    if work.empty:
        return 0.0, 0, ""
    latest = work["event_date"].max()
    work = work.loc[work["event_date"].eq(latest)].copy()
    work["normalized_bloc"] = work["bloc"].map(normalize_bloc)
    work["vote_share"] = pd.to_numeric(
        work["vote_share"], errors="coerce"
    ).fillna(0.0)
    if "data_quality_weight" in work.columns:
        work["quality"] = pd.to_numeric(
            work["data_quality_weight"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
    else:
        work["quality"] = 1.0

    selected_bloc = normalize_bloc(bloc)
    ratios: list[float] = []
    weights: list[float] = []
    source_ids: list[str] = []
    for election_id, election_rows in work.groupby("election_id", sort=True):
        by_region = (
            election_rows.groupby(
                ["region_id", "normalized_bloc"], as_index=False
            )["vote_share"]
            .sum()
        )
        matrix = by_region.pivot_table(
            index="region_id",
            columns="normalized_bloc",
            values="vote_share",
            aggfunc="sum",
            fill_value=0.0,
        )
        shares = matrix.mean(axis=0)
        strongest = float(shares.max()) if not shares.empty else 0.0
        selected = float(shares.get(selected_bloc, 0.0))
        ratio = selected / max(strongest, 1e-12) if strongest > 0.0 else 0.0
        election_type = str(election_rows["election_type"].iloc[0])
        type_weight = float(LAYER_ELECTION_TYPE_WEIGHTS.get(election_type, 0.0))
        quality = float(election_rows["quality"].mean())
        weight = type_weight * quality
        if weight <= 0.0:
            continue
        ratios.append(float(np.clip(ratio, 0.0, 1.0)))
        weights.append(weight)
        source_ids.append(str(election_id))
    if not weights:
        return 0.0, int(len(work)), ""
    score = float(np.average(np.asarray(ratios), weights=np.asarray(weights)))
    return score, int(len(work)), "|".join(source_ids)


def build_election_derived_third_profile_v2(
    speech_profile: pd.DataFrame,
    candidate_context: pd.DataFrame,
    candidate_history: pd.DataFrame,
    presidential_results: pd.DataFrame,
    bloc_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a prior-only profile without mixing party and candidate ballots."""

    context = candidate_context[
        [
            "election_id",
            "slot",
            "candidate_name",
            "bloc",
            "organization_strength",
            "available_date",
            "confidence",
        ]
    ].copy()
    profile = speech_profile.merge(
        context,
        on=["election_id", "slot", "candidate_name"],
        how="left",
        suffixes=("_speech", "_context"),
        validate="one_to_one",
    )

    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for row in profile.itertuples(index=False):
        cutoff_value = election_date(str(row.election_id))
        if cutoff_value is None:
            continue
        cutoff = pd.Timestamp(cutoff_value)
        prior_presidential, prior_presidential_events = (
            _prior_presidential_competitiveness(
                presidential_results, str(row.candidate_name), cutoff
            )
        )
        direct_party, direct_rows, direct_sources = (
            _latest_family_competitiveness(
                bloc_history,
                str(row.bloc),
                cutoff,
                DIRECT_PARTY_TYPES,
            )
        )
        organization_vote, organization_rows, organization_sources = (
            _latest_family_competitiveness(
                bloc_history,
                str(row.bloc),
                cutoff,
                ORGANIZATION_TYPES,
            )
        )
        organization_strength = float(
            np.clip(float(row.organization_strength), 0.0, 1.0)
        )
        personal_competitiveness = prior_presidential * (
            0.5 + 0.5 * np.sqrt(organization_strength)
        )
        party_structure = (
            direct_party
            if direct_party > 0.0
            else ORGANIZATION_FALLBACK_SCALE * organization_vote
        )
        won_offices = _won_office_count(
            candidate_history,
            str(row.election_id),
            str(row.candidate_name),
            cutoff,
        )
        office_support = (
            (1.0 - np.exp(-float(won_offices))) * party_structure
        )
        electoral_competitiveness = float(
            np.clip(
                max(
                    personal_competitiveness,
                    party_structure,
                    office_support,
                ),
                0.0,
                1.0,
            )
        )
        speech_viability = float(np.clip(float(row.viability), 0.0, 1.0))

        if (
            prior_presidential_events > 0
            and personal_competitiveness >= party_structure
        ):
            # Prior presidential stature and speech prominence describe much of
            # the same candidate recognition; combining them would double count.
            viability = electoral_competitiveness
            conversion_mode = "prior_candidate_stature_no_double_count"
        elif direct_party >= COMPETITIVE_BLOC_RATIO:
            viability = 1.0 - (
                (1.0 - speech_viability) * (1.0 - electoral_competitiveness)
            )
            conversion_mode = "direct_party_plus_speech_complement"
        else:
            viability = electoral_competitiveness
            conversion_mode = "subcompetitive_election_evidence"

        confidence = float(
            np.sqrt(
                np.clip(float(row.confidence_speech), 0.0, 1.0)
                * np.clip(float(row.confidence_context), 0.0, 1.0)
            )
        )
        available_date = max(
            pd.Timestamp(row.available_date_speech),
            pd.Timestamp(row.available_date_context),
        )
        if available_date >= cutoff:
            available_date = cutoff - pd.Timedelta(days=1)
        output = {
            "election_id": str(row.election_id),
            "slot": str(row.slot),
            "candidate_name": str(row.candidate_name),
            "viability": float(np.clip(viability, 0.0, 1.0)),
            "centrist_appeal": float(row.centrist_appeal),
            "anti_major_party_appeal": float(row.anti_major_party_appeal),
            "regional_base_overlap": float(row.regional_base_overlap),
            "available_date": available_date.date().isoformat(),
            "confidence": confidence,
            "notes": (
                "Strictly prior direct-party preference, district organization, "
                "presidential stature, won offices, and Assembly role"
            ),
            "provenance_class": "official_election_and_assembly_derived",
            "derivation_version": "election_derived_third_candidate_profile_v2",
        }
        rows.append(output)
        audit_rows.append(
            {
                **output,
                "speech_viability": speech_viability,
                "prior_presidential_competitiveness": prior_presidential,
                "prior_presidential_events": prior_presidential_events,
                "personal_competitiveness": personal_competitiveness,
                "direct_party_competitiveness": direct_party,
                "direct_party_sources": direct_sources,
                "direct_party_rows": direct_rows,
                "organization_competitiveness": organization_vote,
                "organization_sources": organization_sources,
                "organization_rows": organization_rows,
                "party_structure": party_structure,
                "won_offices": won_offices,
                "office_support": office_support,
                "electoral_competitiveness": electoral_competitiveness,
                "conversion_mode": conversion_mode,
                "target_outcome_used": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)
