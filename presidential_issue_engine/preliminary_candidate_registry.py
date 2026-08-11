"""Dated preliminary-candidate identity and prior-election profile evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import election_date


def _bounded_concentration(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=0))
    return std / max(mean + std, 1e-12) if mean + std > 0.0 else 0.0


def _bounded_union(*values: float) -> float:
    remainder = 1.0
    for value in values:
        remainder *= 1.0 - float(np.clip(value, 0.0, 1.0))
    return float(np.clip(1.0 - remainder, 0.0, 1.0))


def build_preliminary_candidate_registry(
    coalition_events: pd.DataFrame,
    withdrawn_transfers: pd.DataFrame,
    candidate_landscape: pd.DataFrame,
) -> pd.DataFrame:
    """Build withdrawal candidates without treating final ballot C as the same entity."""

    events = coalition_events.loc[
        coalition_events["event_type"].astype(str).eq("coalition_withdrawal")
    ].copy()
    transfers = withdrawn_transfers.copy()
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        candidate_rows = transfers.loc[
            transfers["election_id"].astype(str).eq(str(event.election_id))
        ].copy()
        if candidate_rows.empty:
            continue
        names = sorted(set(candidate_rows["candidate_name"].dropna().astype(str)))
        if len(names) != 1:
            raise ValueError(
                f"withdrawal event {event.event_id} must resolve exactly one candidate"
            )
        candidate_name = names[0]
        landscape = candidate_landscape.loc[
            candidate_landscape["election_id"].astype(str).eq(str(event.election_id))
            & candidate_landscape["slot"].astype(str).eq(str(event.source_slot))
            & candidate_landscape["candidate_name"].astype(str).str.casefold().eq(
                candidate_name.casefold()
            )
        ].copy()
        rows.append(
            {
                "event_id": str(event.event_id),
                "election_id": str(event.election_id),
                "source_slot": str(event.source_slot),
                "candidate_name": candidate_name,
                "candidate_role": "withdrawn_preliminary",
                "event_date": str(event.event_date),
                "available_date": str(event.available_date),
                "landscape_matched": bool(not landscape.empty),
                "target_outcome_used": False,
            }
        )
    return pd.DataFrame(rows)


def derive_prior_candidate_profile(
    registry: pd.DataFrame,
    prior_profiles: pd.DataFrame,
    presidential_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive preliminary stature from the same person's prior profile and result."""

    profile = prior_profiles.copy()
    profile["profile_date"] = pd.to_datetime(
        profile["election_id"].map(election_date), errors="coerce"
    )
    results = presidential_results.copy()
    results["result_date"] = pd.to_datetime(
        results["election_id"].map(election_date), errors="coerce"
    )
    output_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for candidate in registry.itertuples(index=False):
        cutoff = pd.Timestamp(candidate.event_date)
        prior = profile.loc[
            profile["candidate_name"].astype(str).str.casefold().eq(
                str(candidate.candidate_name).casefold()
            )
            & profile["profile_date"].lt(cutoff)
        ].sort_values("profile_date")
        if prior.empty:
            audit_rows.append(
                {
                    **candidate._asdict(),
                    "prior_election_id": "",
                    "prior_evidence_available": False,
                    "target_outcome_used": False,
                }
            )
            continue
        source = prior.iloc[-1]
        source_election = str(source["election_id"])
        source_slot = str(source["slot"])
        election = results.loc[
            results["election_id"].astype(str).eq(source_election)
            & results["result_date"].lt(cutoff)
            & results["slot"].astype(str).ne("alpha")
        ].copy()
        candidate_result = election.loc[election["slot"].astype(str).eq(source_slot)]
        if election.empty or candidate_result.empty:
            continue
        by_slot = election.groupby("slot")["votes"].sum()
        total = float(by_slot.sum())
        winner_share = float(by_slot.max()) / max(total, 1e-12)
        candidate_share = float(candidate_result["votes"].sum()) / max(total, 1e-12)
        competitiveness = float(
            np.clip(candidate_share / max(winner_share, 1e-12), 0.0, 1.0)
        )
        shares = (
            pd.to_numeric(candidate_result.get("vote_share"), errors="coerce")
            if "vote_share" in candidate_result.columns
            else pd.to_numeric(candidate_result["votes"], errors="coerce")
        )
        concentration = _bounded_concentration(shares)
        prior_overlap = float(
            np.clip(float(source["regional_base_overlap"]), 0.0, 1.0)
        )
        regional_overlap = _bounded_union(
            prior_overlap * competitiveness,
            concentration * competitiveness,
        )
        confidence = float(np.clip(float(source["confidence"]), 0.0, 1.0))
        output = {
            "event_id": str(candidate.event_id),
            "election_id": str(candidate.election_id),
            "slot": str(candidate.source_slot),
            "candidate_name": str(candidate.candidate_name),
            "viability": competitiveness,
            "regional_base_overlap": regional_overlap,
            "available_date": pd.Timestamp(candidate.available_date).date().isoformat(),
            "confidence": confidence,
            "prior_election_id": source_election,
            "prior_slot": source_slot,
            "target_outcome_used": False,
        }
        output_rows.append(output)
        audit_rows.append(
            {
                **candidate._asdict(),
                **output,
                "prior_candidate_share": candidate_share,
                "prior_winner_share": winner_share,
                "prior_competitiveness": competitiveness,
                "prior_profile_overlap": prior_overlap,
                "prior_regional_concentration": concentration,
                "prior_evidence_available": True,
                "target_outcome_used": False,
            }
        )
    return pd.DataFrame(output_rows), pd.DataFrame(audit_rows)


def merge_preliminary_profile(
    active_profile: pd.DataFrame,
    automatic_preliminary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace only fields with prior-person evidence; leave unmatched rows intact."""

    keys = ["election_id", "slot", "candidate_name"]
    replacement = automatic_preliminary[
        keys + ["viability", "regional_base_overlap", "available_date", "confidence"]
    ].rename(
        columns={
            "viability": "automatic_viability",
            "regional_base_overlap": "automatic_regional_base_overlap",
            "available_date": "automatic_available_date",
            "confidence": "automatic_confidence",
        }
    )
    out = active_profile.merge(replacement, on=keys, how="left", validate="one_to_one")
    matched = out["automatic_viability"].notna()
    audit = out.loc[
        matched,
        keys
        + [
            "viability",
            "automatic_viability",
            "regional_base_overlap",
            "automatic_regional_base_overlap",
        ],
    ].copy()
    audit = audit.rename(
        columns={
            "viability": "prior_active_viability",
            "regional_base_overlap": "prior_active_regional_base_overlap",
        }
    )
    audit["target_outcome_used"] = False
    out.loc[matched, "viability"] = out.loc[matched, "automatic_viability"]
    out.loc[matched, "regional_base_overlap"] = out.loc[
        matched, "automatic_regional_base_overlap"
    ]
    out.loc[matched, "available_date"] = out.loc[matched, "automatic_available_date"]
    out.loc[matched, "confidence"] = out.loc[matched, "automatic_confidence"]
    return out.drop(
        columns=[
            "automatic_viability",
            "automatic_regional_base_overlap",
            "automatic_available_date",
            "automatic_confidence",
        ]
    ), audit

