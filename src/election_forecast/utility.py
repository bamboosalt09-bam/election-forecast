"""Utility construction for candidate-region-date forecasts."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from election_forecast.candidate_score import compute_candidate_general_premium
from election_forecast.config import ForecastConfig, ModelWeights
from election_forecast.event_effects import compute_event_effects
from election_forecast.enhanced_issues import merge_enhanced_issue_scores
from election_forecast.filters import filter_available_date, latest_by_available_date
from election_forecast.issue_score import compute_candidate_issue_scores, compute_issue_impact
from election_forecast.local_fit import compute_local_policy_fit
from election_forecast.party_base import compute_party_base, compute_party_base_effect
from election_forecast.poll_score import compute_poll_scores


def prepare_forecast_inputs(
    data: Dict[str, pd.DataFrame],
    forecast_date: str | pd.Timestamp,
    exclude_election_id: str | None = None,
    target_election_id: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Apply available-date filtering and latest-row selection.

    ``exclude_election_id`` is used by backtests so the target actual result is
    never available to input feature construction, even if the CSV was filled
    incorrectly.
    """

    filtered = {name: filter_available_date(frame, forecast_date) for name, frame in data.items()}
    if exclude_election_id and "election_results" in filtered:
        filtered["election_results"] = filtered["election_results"].loc[
            filtered["election_results"]["election_id"] != exclude_election_id
        ].copy()
    if target_election_id:
        for name, frame in list(filtered.items()):
            if name in {"election_results", "bloc_history_results"} or frame.empty or "election_id" not in frame.columns:
                continue
            filtered[name] = frame.loc[frame["election_id"].astype(str) == str(target_election_id)].copy()

    filtered["candidates"] = latest_by_available_date(filtered["candidates"], ["candidate_id"])
    candidate_ids = set(filtered["candidates"].get("candidate_id", pd.Series(dtype=str)).astype(str))
    if candidate_ids:
        for name in ["candidate_party_vectors", "polls", "issue_scores"]:
            frame = filtered.get(name)
            if frame is not None and not frame.empty and "candidate_id" in frame.columns:
                filtered[name] = frame.loc[frame["candidate_id"].astype(str).isin(candidate_ids)].copy()
        events = filtered.get("event_effects")
        if events is not None and not events.empty:
            keep = pd.Series(True, index=events.index)
            for column in ["source_candidate_id", "target_candidate_id"]:
                if column in events.columns:
                    values = events[column].fillna("").astype(str)
                    keep &= values.eq("") | values.isin(candidate_ids)
            filtered["event_effects"] = events.loc[keep].copy()
    filtered["candidate_party_vectors"] = latest_by_available_date(
        filtered["candidate_party_vectors"], ["candidate_id"]
    )
    filtered["region_issue_sensitivity"] = latest_by_available_date(
        filtered["region_issue_sensitivity"], ["region_id", "issue_name"]
    )
    filtered["candidate_policy_positions"] = latest_by_available_date(
        filtered["candidate_policy_positions"], ["candidate_id", "issue_name"]
    )
    return filtered


def build_candidate_region_grid(candidates: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Create the base ``candidate x region`` frame."""

    candidate_cols = ["candidate_id", "candidate_name"]
    grid = candidates[candidate_cols].drop_duplicates().merge(
        regions[["region_id", "eligible_voters", "expected_turnout"]].drop_duplicates(),
        how="cross",
    )
    return grid


def compute_turnout_effects(candidates: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for TurnoutGroup x CandidateSupportGroup effects.

    The MVP has no turnout-group support matrix, so the default contribution is
    0.0. This function is deliberately isolated for later group-level modeling.
    """

    grid = build_candidate_region_grid(candidates, regions)
    grid["turnout_effect"] = 0.0
    return grid[["candidate_id", "region_id", "turnout_effect"]]


def compute_interaction_effects(
    base_features: pd.DataFrame,
    candidate_issue_scores: pd.DataFrame,
    region_issue_sensitivity: pd.DataFrame,
    candidate_policy_positions: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Compute future-model interaction effects.

    MVP coefficients default to zero, but the function keeps required expansion
    points:
    CandidatePremium x RegionAffinity,
    PolicyPosition x RegionSensitivity x CandidateCredibility x IssueSalience,
    PartyBase x CandidateFit,
    IssueSalience x CandidateLink x RegionSensitivity,
    TurnoutGroup x CandidateSupportGroup.
    """

    weights = config.interaction_weights
    result = base_features[["candidate_id", "region_id"]].copy()
    result["candidate_local_interaction"] = (
        weights.a_candidate_local
        * base_features.get("candidate_general_premium", 0.0)
        * base_features.get("region_affinity", 1.0)
    )
    result["party_candidate_fit_interaction"] = (
        weights.a_party_candidate_fit
        * base_features.get("party_base_effect", 0.0)
        * base_features.get("candidate_fit", 1.0)
    )
    result["turnout_group_interaction"] = (
        weights.a_turnout_group
        * base_features.get("turnout_group", 0.0)
        * base_features.get("candidate_support_group", 0.0)
    )

    result["policy_region_interaction"] = 0.0
    if (
        weights.a_policy_region
        and not candidate_issue_scores.empty
        and not region_issue_sensitivity.empty
        and not candidate_policy_positions.empty
    ):
        policy_joined = candidate_policy_positions.merge(
            region_issue_sensitivity, on="issue_name", how="inner"
        ).merge(
            candidate_issue_scores[["candidate_id", "issue_name", "issue_salience"]],
            on=["candidate_id", "issue_name"],
            how="left",
        )
        for col in ["policy_direction", "sensitivity_score", "candidate_credibility", "issue_salience"]:
            policy_joined[col] = pd.to_numeric(policy_joined[col], errors="coerce").fillna(0.0)
        policy_joined["component"] = (
            weights.a_policy_region
            * policy_joined["policy_direction"]
            * policy_joined["sensitivity_score"]
            * policy_joined["candidate_credibility"]
            * policy_joined["issue_salience"]
        )
        policy_component = policy_joined.groupby(["candidate_id", "region_id"], as_index=False)[
            "component"
        ].sum()
        result = result.merge(policy_component, on=["candidate_id", "region_id"], how="left")
        result["policy_region_interaction"] = result["component"].fillna(0.0)
        result = result.drop(columns=["component"])

    result["issue_region_interaction"] = 0.0
    if weights.a_issue_region and not candidate_issue_scores.empty and not region_issue_sensitivity.empty:
        issue_joined = candidate_issue_scores.merge(region_issue_sensitivity, on="issue_name", how="inner")
        for col in ["issue_salience", "candidate_link_score", "sensitivity_score"]:
            issue_joined[col] = pd.to_numeric(issue_joined[col], errors="coerce").fillna(0.0)
        issue_joined["component"] = (
            weights.a_issue_region
            * issue_joined["issue_salience"]
            * issue_joined["candidate_link_score"]
            * issue_joined["sensitivity_score"]
        )
        issue_component = issue_joined.groupby(["candidate_id", "region_id"], as_index=False)[
            "component"
        ].sum()
        result = result.merge(issue_component, on=["candidate_id", "region_id"], how="left")
        result["issue_region_interaction"] = result["component"].fillna(0.0)
        result = result.drop(columns=["component"])

    interaction_cols = [
        "candidate_local_interaction",
        "policy_region_interaction",
        "party_candidate_fit_interaction",
        "issue_region_interaction",
        "turnout_group_interaction",
    ]
    result["interaction_effect"] = result[interaction_cols].sum(axis=1)
    return result[["candidate_id", "region_id", "interaction_effect", *interaction_cols]]


def compute_utility_frame(
    data: Dict[str, pd.DataFrame],
    forecast_date: str | pd.Timestamp,
    config: ForecastConfig,
    weights: ModelWeights,
    target_election_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full feature frame and temporary linear Utility.

    The MVP Utility is a linear sum. This is explicitly temporary; the
    interaction function above is the expansion point for the full model.
    """

    candidates = data["candidates"]
    regions = data["regions"]
    grid = build_candidate_region_grid(candidates, regions)

    if target_election_id is None and "election_id" in candidates.columns and not candidates.empty:
        election_ids = candidates["election_id"].dropna().astype(str).unique()
        if len(election_ids) == 1:
            target_election_id = election_ids[0]
    party_base = compute_party_base(
        data["election_results"],
        forecast_date,
        config,
        bloc_history=data.get("bloc_history_results", pd.DataFrame()),
        target_election_id=target_election_id,
    )
    party_base_effect = compute_party_base_effect(party_base, data["candidate_party_vectors"], config)
    poll_scores = compute_poll_scores(data["polls"], forecast_date, config.poll_half_life_days)
    candidate_premium = compute_candidate_general_premium(candidates)
    local_policy_fit = compute_local_policy_fit(
        data["region_issue_sensitivity"], data["candidate_policy_positions"]
    )
    candidate_issue_scores = compute_candidate_issue_scores(
        merge_enhanced_issue_scores(data, forecast_date),
        forecast_date,
        config.issue_half_life_days,
    )
    issue_impact = compute_issue_impact(
        candidate_issue_scores,
        data["region_issue_sensitivity"],
        data.get("issue_scope_weights", pd.DataFrame()),
    )
    events = compute_event_effects(data["event_effects"], candidates, regions, forecast_date)
    turnout = compute_turnout_effects(candidates, regions)

    feature_frame = (
        grid.merge(party_base_effect, on=["candidate_id", "region_id"], how="left")
        .merge(poll_scores, on="candidate_id", how="left")
        .merge(candidate_premium, on="candidate_id", how="left")
        .merge(local_policy_fit, on=["candidate_id", "region_id"], how="left")
        .merge(issue_impact, on=["candidate_id", "region_id"], how="left")
        .merge(events, on=["candidate_id", "region_id"], how="left")
        .merge(turnout, on=["candidate_id", "region_id"], how="left")
    )

    feature_cols = [
        "party_base_effect",
        "poll_score",
        "candidate_general_premium",
        "local_policy_fit",
        "issue_impact",
        "issue_impact_national",
        "issue_impact_local",
        "event_effect",
        "turnout_effect",
    ]
    feature_frame[feature_cols] = feature_frame[feature_cols].fillna(0.0)
    feature_frame["region_affinity"] = 1.0
    feature_frame["candidate_fit"] = 1.0
    feature_frame["turnout_group"] = 0.0
    feature_frame["candidate_support_group"] = 0.0

    interactions = compute_interaction_effects(
        feature_frame,
        candidate_issue_scores,
        data["region_issue_sensitivity"],
        data["candidate_policy_positions"],
        config,
    )
    feature_frame = feature_frame.merge(interactions, on=["candidate_id", "region_id"], how="left")
    feature_frame["interaction_effect"] = feature_frame["interaction_effect"].fillna(0.0)

    feature_frame["utility"] = (
        weights.w_party_base * feature_frame["party_base_effect"]
        + weights.w_poll * feature_frame["poll_score"]
        + weights.w_candidate * feature_frame["candidate_general_premium"]
        + weights.w_local_policy * feature_frame["local_policy_fit"]
        + weights.w_issue * feature_frame["issue_impact"]
        + weights.w_event * feature_frame["event_effect"]
        + weights.w_turnout * feature_frame["turnout_effect"]
        + feature_frame["interaction_effect"]
    )
    return feature_frame, party_base
