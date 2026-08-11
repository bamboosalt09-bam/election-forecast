"""Third-candidate stature from strictly prior election records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import election_date, normalize_bloc


PARTY_SUPPORT_TYPES = frozenset(
    {
        "national_assembly_pr",
        "assembly_pr",
        "metro_council_pr",
        "local_council_pr",
        "assembly_district",
    }
)
COMPETITIVE_BLOC_RATIO = 1.0 / 3.0


def _prior_presidential_competitiveness(
    results: pd.DataFrame,
    candidate_name: str,
    cutoff: pd.Timestamp,
) -> tuple[float, int]:
    work = results.loc[
        results["candidate_name"].astype(str).eq(candidate_name)
        & results["slot"].astype(str).ne("alpha")
    ].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(election_date), errors="coerce"
    )
    work = work.loc[work["event_date"].lt(cutoff)].copy()
    ratios: list[float] = []
    for election_id, candidate in work.groupby("election_id"):
        election = results.loc[
            results["election_id"].astype(str).eq(str(election_id))
            & results["slot"].astype(str).ne("alpha")
        ].copy()
        national = election.groupby("candidate_name")["votes"].sum()
        total = float(national.sum())
        if total <= 0.0 or national.empty:
            continue
        candidate_share = float(candidate["votes"].sum()) / total
        winner_share = float(national.max()) / total
        ratios.append(candidate_share / max(winner_share, 1e-12))
    return (float(max(ratios, default=0.0)), len(ratios))


def _latest_party_competitiveness(
    history: pd.DataFrame,
    bloc: str,
    cutoff: pd.Timestamp,
) -> tuple[float, int, str]:
    work = history.loc[
        history["election_type"].astype(str).isin(PARTY_SUPPORT_TYPES)
    ].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(election_date), errors="coerce"
    )
    work = work.loc[work["event_date"].lt(cutoff)].copy()
    if work.empty:
        return 0.0, 0, ""
    latest = work["event_date"].max()
    latest_rows = work.loc[work["event_date"].eq(latest)].copy()
    latest_rows["normalized_bloc"] = latest_rows["bloc"].map(normalize_bloc)
    latest_rows["vote_share"] = pd.to_numeric(
        latest_rows["vote_share"], errors="coerce"
    ).fillna(0.0)
    shares = latest_rows.groupby("normalized_bloc")["vote_share"].mean()
    strongest = float(shares.max()) if not shares.empty else 0.0
    selected = float(shares.get(normalize_bloc(bloc), 0.0))
    ratio = selected / max(strongest, 1e-12) if strongest > 0.0 else 0.0
    return float(np.clip(ratio, 0.0, 1.0)), int(len(latest_rows)), str(
        latest_rows["election_id"].iloc[0]
    )


def _won_office_count(
    candidate_history: pd.DataFrame,
    election_id: str,
    candidate_name: str,
    cutoff: pd.Timestamp,
) -> int:
    work = candidate_history.loc[
        candidate_history["target_election_id"].astype(str).eq(election_id)
        & candidate_history["target_candidate_name"].astype(str).eq(candidate_name)
    ].copy()
    work["source_date"] = pd.to_datetime(
        work["source_election_date"], errors="coerce"
    )
    work = work.loc[
        work["source_date"].lt(cutoff)
        & work["source_sg_typecode"].astype(str).isin({"2", "3", "4", "7"})
        & work["prior_election_won"].astype(str).str.upper().eq("Y")
    ]
    return int(len(work))


def build_election_derived_third_profile(
    speech_profile: pd.DataFrame,
    candidate_context: pd.DataFrame,
    candidate_history: pd.DataFrame,
    presidential_results: pd.DataFrame,
    bloc_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace speech-only viability with prior-election stature evidence."""

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
    audit: list[dict[str, object]] = []
    for row in profile.itertuples(index=False):
        cutoff = election_date(str(row.election_id))
        if cutoff is None:
            continue
        cutoff = pd.Timestamp(cutoff)
        prior_presidential, prior_presidential_events = (
            _prior_presidential_competitiveness(
                presidential_results, str(row.candidate_name), cutoff
            )
        )
        party_competitiveness, party_rows, party_source = (
            _latest_party_competitiveness(bloc_history, str(row.bloc), cutoff)
        )
        organization = float(
            np.clip(float(row.organization_strength), 0.0, 1.0)
        )
        personal_competitiveness = prior_presidential * (
            0.5 + 0.5 * np.sqrt(organization)
        )
        won_offices = _won_office_count(
            candidate_history,
            str(row.election_id),
            str(row.candidate_name),
            cutoff,
        )
        office_support = (
            (1.0 - np.exp(-float(won_offices))) * party_competitiveness
        )
        electoral_competitiveness = float(
            np.clip(
                max(
                    personal_competitiveness,
                    party_competitiveness,
                    office_support,
                ),
                0.0,
                1.0,
            )
        )
        speech_viability = float(np.clip(float(row.viability), 0.0, 1.0))
        if electoral_competitiveness >= COMPETITIVE_BLOC_RATIO:
            viability = 1.0 - (
                (1.0 - speech_viability) * (1.0 - electoral_competitiveness)
            )
            conversion_mode = "complementary_competitive_evidence"
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
                "Strictly prior presidential competitiveness, party ballot support, "
                "won office history, and speech-derived candidate role"
            ),
            "provenance_class": "official_election_and_assembly_derived",
            "derivation_version": "election_derived_third_candidate_profile_v1",
        }
        rows.append(output)
        audit.append(
            {
                **output,
                "speech_viability": speech_viability,
                "prior_presidential_competitiveness": prior_presidential,
                "prior_presidential_events": prior_presidential_events,
                "party_competitiveness": party_competitiveness,
                "party_source_election": party_source,
                "party_source_rows": party_rows,
                "won_offices": won_offices,
                "office_support": office_support,
                "electoral_competitiveness": electoral_competitiveness,
                "conversion_mode": conversion_mode,
                "target_outcome_used": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit)
