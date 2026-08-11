"""Election-derived third-candidate character traits and stature."""

from __future__ import annotations

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import election_date, normalize_bloc
from presidential_issue_engine.election_derived_third_candidate_profile_v2 import (
    COMPETITIVE_BLOC_RATIO,
    DIRECT_PARTY_TYPES,
    build_election_derived_third_profile_v2,
)
from presidential_issue_engine.electorate_layers import LAYER_ELECTION_TYPE_WEIGHTS


COMPETITIVE_PARTY_CHALLENGE_SCALE = 0.75


def _bounded_union(*values: float) -> float:
    remainder = 1.0
    for value in values:
        remainder *= 1.0 - float(np.clip(value, 0.0, 1.0))
    return float(np.clip(1.0 - remainder, 0.0, 1.0))


def _bounded_concentration(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=0))
    return std / max(mean + std, 1e-12) if mean + std > 0.0 else 0.0


def _latest_direct_party_concentration(
    history: pd.DataFrame,
    bloc: str,
    cutoff: pd.Timestamp,
) -> tuple[float, str]:
    work = history.loc[
        history["election_type"].astype(str).isin(DIRECT_PARTY_TYPES)
    ].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(election_date), errors="coerce"
    )
    work = work.loc[work["event_date"].lt(cutoff)].copy()
    if work.empty:
        return 0.0, ""
    work = work.loc[work["event_date"].eq(work["event_date"].max())].copy()
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
    scores: list[float] = []
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
        values = matrix.get(selected_bloc, pd.Series(0.0, index=matrix.index))
        election_type = str(election_rows["election_type"].iloc[0])
        weight = float(LAYER_ELECTION_TYPE_WEIGHTS.get(election_type, 0.0))
        weight *= float(election_rows["quality"].mean())
        if weight <= 0.0:
            continue
        scores.append(_bounded_concentration(values))
        weights.append(weight)
        source_ids.append(str(election_id))
    if not weights:
        return 0.0, ""
    return float(np.average(scores, weights=weights)), "|".join(source_ids)


def _prior_presidential_concentration(
    results: pd.DataFrame,
    candidate_name: str,
    cutoff: pd.Timestamp,
) -> tuple[float, str]:
    work = results.loc[
        results["candidate_name"].astype(str).eq(candidate_name)
        & results["slot"].astype(str).ne("alpha")
    ].copy()
    work["event_date"] = pd.to_datetime(
        work["election_id"].map(election_date), errors="coerce"
    )
    work = work.loc[work["event_date"].lt(cutoff)].copy()
    if work.empty:
        return 0.0, ""
    latest = work["event_date"].max()
    work = work.loc[work["event_date"].eq(latest)].copy()
    if "vote_share" in work.columns:
        shares = pd.to_numeric(work["vote_share"], errors="coerce").fillna(0.0)
    else:
        shares = pd.to_numeric(work["votes"], errors="coerce").fillna(0.0)
    return _bounded_concentration(shares), str(work["election_id"].iloc[0])


def _candidate_base_mean(
    candidate_base: pd.DataFrame,
    election_id: str,
    slot: str,
    cutoff: pd.Timestamp,
) -> tuple[float, int]:
    work = candidate_base.loc[
        candidate_base["election_id"].astype(str).eq(election_id)
        & candidate_base["slot"].astype(str).eq(slot)
    ].copy()
    work["available_date"] = pd.to_datetime(
        work["available_date"], errors="coerce"
    )
    work = work.loc[work["available_date"].lt(cutoff)].copy()
    if work.empty:
        return 0.0, 0
    values = pd.to_numeric(work["regional_affinity"], errors="coerce").fillna(0.0)
    confidence = pd.to_numeric(work.get("confidence", 1.0), errors="coerce").fillna(0.0)
    if float(confidence.sum()) > 0.0:
        score = float(np.average(values, weights=confidence))
    else:
        score = float(values.mean())
    return float(np.clip(score, 0.0, 1.0)), int(len(work))


def build_election_derived_third_profile_v3(
    speech_profile: pd.DataFrame,
    candidate_context: pd.DataFrame,
    candidate_history: pd.DataFrame,
    presidential_results: pd.DataFrame,
    bloc_history: pd.DataFrame,
    candidate_landscape: pd.DataFrame,
    candidate_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Automate all third-profile fields from dated non-target evidence."""

    profile, stature_audit = build_election_derived_third_profile_v2(
        speech_profile,
        candidate_context,
        candidate_history,
        presidential_results,
        bloc_history,
    )
    audit_lookup = stature_audit.set_index(["election_id", "slot"])
    context_lookup = candidate_context.set_index(["election_id", "slot"])
    landscape_lookup = candidate_landscape.set_index(["election_id", "slot"])

    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for output in profile.to_dict(orient="records"):
        key = (str(output["election_id"]), str(output["slot"]))
        stature = audit_lookup.loc[key]
        context = context_lookup.loc[key]
        cutoff = pd.Timestamp(election_date(key[0]))
        confidence = float(np.clip(float(output["confidence"]), 0.0, 1.0))
        direct_party = float(stature["direct_party_competitiveness"])
        personal = float(stature["personal_competitiveness"])
        outsider = float(np.clip(float(context["outsider_status"]), 0.0, 1.0))

        ideology_balance = 0.0
        if key in landscape_lookup.index:
            landscape = landscape_lookup.loc[key]
            conservative = float(
                pd.to_numeric(landscape["conservative"], errors="coerce")
            )
            liberal = float(pd.to_numeric(landscape["liberal"], errors="coerce"))
            denominator = conservative + liberal
            if denominator > 0.0:
                ideology_balance = 1.0 - abs(conservative - liberal) / denominator
        centrist_evidence = ideology_balance * direct_party * confidence
        centrist_appeal = _bounded_union(
            float(output["centrist_appeal"]), centrist_evidence
        )

        competitive_party_challenge = (
            COMPETITIVE_PARTY_CHALLENGE_SCALE * direct_party
            if direct_party >= COMPETITIVE_BLOC_RATIO
            else 0.0
        )
        established_outsider = outsider * personal * confidence
        anti_major_evidence = max(
            competitive_party_challenge,
            established_outsider,
        )
        anti_major_party_appeal = _bounded_union(
            float(output["anti_major_party_appeal"]), anti_major_evidence
        )

        direct_concentration, direct_concentration_sources = (
            _latest_direct_party_concentration(
                bloc_history, str(context["bloc"]), cutoff
            )
        )
        prior_concentration, prior_concentration_source = (
            _prior_presidential_concentration(
                presidential_results, str(output["candidate_name"]), cutoff
            )
        )
        base_mean, base_rows = _candidate_base_mean(
            candidate_base, key[0], key[1], cutoff
        )
        regional_base_overlap = _bounded_union(
            base_mean,
            direct_concentration * direct_party,
            prior_concentration * personal,
        )

        original_centrist = float(output["centrist_appeal"])
        original_anti_major = float(output["anti_major_party_appeal"])
        original_regional_overlap = float(output["regional_base_overlap"])
        output["centrist_appeal"] = centrist_appeal
        output["anti_major_party_appeal"] = anti_major_party_appeal
        output["regional_base_overlap"] = regional_base_overlap
        output["notes"] = (
            str(output["notes"])
            + "; character traits derived from ideology balance, outsider status, "
            "party concentration, and prior candidate footprint"
        )
        output["derivation_version"] = "election_derived_third_candidate_profile_v3"
        rows.append(output)
        audits.append(
            {
                **output,
                "speech_centrist_appeal": original_centrist,
                "ideology_balance": ideology_balance,
                "centrist_evidence": centrist_evidence,
                "speech_anti_major_party_appeal": original_anti_major,
                "competitive_party_challenge": competitive_party_challenge,
                "established_outsider": established_outsider,
                "anti_major_evidence": anti_major_evidence,
                "speech_regional_base_overlap": original_regional_overlap,
                "candidate_base_mean": base_mean,
                "candidate_base_rows": base_rows,
                "direct_party_concentration": direct_concentration,
                "direct_party_concentration_sources": direct_concentration_sources,
                "prior_presidential_concentration": prior_concentration,
                "prior_presidential_concentration_source": prior_concentration_source,
                "target_outcome_used": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audits)
