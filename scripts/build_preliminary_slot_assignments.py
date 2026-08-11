"""Build strict rolling preliminary A/B/C assignments through 2022.

Target-election outcomes are never passed to the preliminary model or slot
compiler. The generator uses a fixed ridge specification, six slot-free
predictors, prior-election regional vote volume, and coefficient uncertainty.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.preliminary_slots import (  # noqa: E402
    DEFAULT_SLOT_FREE_PREDICTORS,
    PreliminarySlotConfig,
    attenuate_withdrawn_endorsement_transfer,
    assign_preliminary_slots,
    assign_role_aware_slots,
    latent_withdrawn_candidate_weight,
    redistribute_withdrawn_vote_mass,
)
from presidential_issue_engine import issue_vote_engine as engine  # noqa: E402
from presidential_issue_engine import electorate_layers as electorate  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "preliminary_slot_assignment"
ELECTIONS = tuple(engine.ORDER)
WARMUP_ELECTIONS = ("pres_1997",)
PREDICTORS = DEFAULT_SLOT_FREE_PREDICTORS
RIDGE_ALPHA = engine.DEFAULT_RIDGE_ALPHA
MONTE_CARLO_DRAWS = 4000
RANDOM_SEED = 20260718
CANDIDATE_STRENGTH_EXPONENT = 2.0
CANDIDATE_STRENGTH_FLOOR = 0.08
WITHDRAWAL_EVENT_PROFILES = ROOT / "data" / "raw" / "withdrawal_event_profiles.csv"


def _withdrawal_profiles() -> pd.DataFrame:
    """Build point-in-time latent-candidate profiles from existing raw inputs."""

    registry_value = getattr(engine, "WITHDRAWAL_TRANSFER_REGISTRY", "")
    registry_path = Path(registry_value) if registry_value else None
    if registry_path is not None and not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    if registry_path is not None and registry_path.exists():
        registry = pd.read_csv(registry_path, encoding="utf-8-sig")
        if "use_in_preliminary_layer" in registry.columns:
            enabled = (
                registry["use_in_preliminary_layer"]
                .astype(str)
                .str.lower()
                .isin(["1", "true", "yes", "y"])
            )
            registry = registry.loc[enabled].copy()
        if "preliminary_transfer_rate" in registry.columns:
            registry["transfer_rate"] = registry["preliminary_transfer_rate"]
        if "preliminary_voter_compliance" in registry.columns:
            registry["voter_compliance"] = registry[
                "preliminary_voter_compliance"
            ]
        registry["available_date"] = pd.to_datetime(
            registry["available_date"], errors="coerce"
        )
        registry["election_cutoff"] = pd.to_datetime(
            registry["election_id"].map(engine.ELECTION_DATES), errors="coerce"
        )
        registry = registry.loc[
            registry["available_date"].notna()
            & registry["election_cutoff"].notna()
            & registry["available_date"].le(registry["election_cutoff"])
        ].copy()
        rows: list[dict[str, object]] = []
        for (election_id, candidate_name), candidate_rows in registry.groupby(
            ["election_id", "candidate_name"], sort=False
        ):
            base = candidate_rows.sort_values("available_date").iloc[-1]
            target_fractions = {
                str(row.target_slot): float(row.transfer_rate)
                * float(row.voter_compliance)
                for row in candidate_rows.itertuples(index=False)
            }
            viability = float(base.viability)
            centrist = float(base.centrist_appeal)
            anti_major = float(base.anti_major_party_appeal)
            overlap = float(base.regional_base_overlap)
            confidence = float(base.profile_confidence)
            rows.append(
                {
                    "election_id": str(election_id),
                    "source_slot": str(base.source_slot),
                    "candidate_name": str(candidate_name),
                    "viability": viability,
                    "centrist_appeal": centrist,
                    "anti_major_party_appeal": anti_major,
                    "regional_base_overlap": overlap,
                    "confidence": confidence,
                    "latent_candidate_weight": latent_withdrawn_candidate_weight(
                        viability=viability,
                        centrist_appeal=centrist,
                        anti_major_party_appeal=anti_major,
                        regional_base_overlap=overlap,
                        confidence=confidence,
                    ),
                    "target_fractions": target_fractions,
                    "available_date": candidate_rows["available_date"].max().date().isoformat(),
                    "support_retention": float(base.get("support_retention", 1.0)),
                }
            )
        return pd.DataFrame(rows)

    events = pd.read_csv(ROOT / engine.COALITION_EVENTS)
    transfers = pd.read_csv(ROOT / engine.WITHDRAWN_CANDIDATE_TRANSFERS)
    landscape = pd.read_csv(ROOT / engine.CANDIDATE_POLITICAL_LANDSCAPE)
    third = pd.read_csv(ROOT / engine.THIRD_CANDIDATE_PROFILE)
    events = events.loc[events["event_type"].eq("coalition_withdrawal")].copy()
    rows: list[dict[str, object]] = []
    for (election_id, candidate_name), candidate_transfers in transfers.groupby(
        ["election_id", "candidate_name"], sort=False
    ):
        election_events = events.loc[events["election_id"].eq(election_id)]
        if election_events.empty:
            continue
        event = election_events.iloc[0]
        available = pd.to_datetime(candidate_transfers["available_date"], errors="coerce")
        cutoff = pd.Timestamp(engine.ELECTION_DATES[str(election_id)])
        candidate_transfers = candidate_transfers.loc[available.le(cutoff)].copy()
        if candidate_transfers.empty:
            continue
        candidate_landscape = landscape.loc[
            landscape["election_id"].eq(election_id)
            & landscape["candidate_name"].astype(str).str.casefold().eq(str(candidate_name).casefold())
        ]
        if candidate_landscape.empty:
            continue
        landscape_row = candidate_landscape.iloc[0]
        candidate_third = third.loc[
            third["election_id"].eq(election_id)
            & third["candidate_name"].astype(str).str.casefold().eq(str(candidate_name).casefold())
        ]
        overlap = (
            float(candidate_third["regional_base_overlap"].iloc[0])
            if not candidate_third.empty
            else 0.0
        )
        viability = float(pd.to_numeric(candidate_transfers["viability"], errors="coerce").max())
        confidence = min(
            float(pd.to_numeric(landscape_row["confidence"], errors="coerce")),
            float(pd.to_numeric(candidate_transfers["confidence"], errors="coerce").max()),
        )
        centrist = float(pd.to_numeric(landscape_row["centrist"], errors="coerce"))
        anti_major = float(
            pd.to_numeric(landscape_row["anti_establishment"], errors="coerce")
        )
        target_fractions = {
            str(row.target_slot): float(row.transfer_rate) * float(row.voter_compliance)
            for row in candidate_transfers.itertuples(index=False)
        }
        rows.append(
            {
                "election_id": str(election_id),
                "source_slot": str(event["source_slot"]),
                "candidate_name": str(candidate_name),
                "viability": viability,
                "centrist_appeal": centrist,
                "anti_major_party_appeal": anti_major,
                "regional_base_overlap": overlap,
                "confidence": confidence,
                "latent_candidate_weight": latent_withdrawn_candidate_weight(
                    viability=viability,
                    centrist_appeal=centrist,
                    anti_major_party_appeal=anti_major,
                    regional_base_overlap=overlap,
                    confidence=confidence,
                ),
                "target_fractions": target_fractions,
                "available_date": str(event["available_date"]),
            }
        )
    legacy = pd.DataFrame(rows)
    if not WITHDRAWAL_EVENT_PROFILES.exists():
        return legacy

    event_profiles = pd.read_csv(WITHDRAWAL_EVENT_PROFILES)
    event_profiles["available_date"] = pd.to_datetime(
        event_profiles["available_date"], errors="coerce"
    )
    event_profiles["election_cutoff"] = pd.to_datetime(
        event_profiles["election_id"].map(engine.ELECTION_DATES), errors="coerce"
    )
    event_profiles = event_profiles.loc[
        event_profiles["available_date"].notna()
        & event_profiles["election_cutoff"].notna()
        & event_profiles["available_date"].le(event_profiles["election_cutoff"])
    ].copy()
    profile_rows: list[dict[str, object]] = []
    for (election_id, candidate_name), candidate_events in event_profiles.groupby(
        ["election_id", "candidate_name"], sort=False
    ):
        initial = candidate_events.loc[
            candidate_events["event_type"].eq("coalition_withdrawal")
        ].sort_values("available_date")
        if initial.empty:
            continue
        base = initial.iloc[0]
        target_fractions = {
            str(row.target_slot): float(row.transfer_rate) * float(row.voter_compliance)
            for row in initial.itertuples(index=False)
        }
        retention_factors: list[float] = []
        reversals = candidate_events.loc[
            candidate_events["event_type"].eq("coalition_support_withdrawal")
        ].sort_values("available_date")
        for reversal in reversals.itertuples(index=False):
            days = (
                pd.Timestamp(reversal.election_cutoff) - pd.Timestamp(reversal.available_date)
            ).total_seconds() / 86400.0
            target_fractions, retention = attenuate_withdrawn_endorsement_transfer(
                target_fractions,
                target_slot=str(reversal.target_slot),
                event_strength=float(reversal.event_strength),
                voter_reach=float(reversal.voter_reach),
                days_to_election=days,
            )
            retention_factors.append(retention)
        viability = float(base.viability)
        centrist = float(base.centrist_appeal)
        anti_major = float(base.anti_major_party_appeal)
        overlap = float(base.regional_base_overlap)
        confidence = float(base.confidence)
        profile_rows.append(
            {
                "election_id": str(election_id),
                "source_slot": str(base.source_slot),
                "candidate_name": str(candidate_name),
                "viability": viability,
                "centrist_appeal": centrist,
                "anti_major_party_appeal": anti_major,
                "regional_base_overlap": overlap,
                "confidence": confidence,
                "latent_candidate_weight": latent_withdrawn_candidate_weight(
                    viability=viability,
                    centrist_appeal=centrist,
                    anti_major_party_appeal=anti_major,
                    regional_base_overlap=overlap,
                    confidence=confidence,
                ),
                "target_fractions": target_fractions,
                "available_date": pd.Timestamp(candidate_events["available_date"].max()).date().isoformat(),
                "support_retention": float(np.prod(retention_factors)) if retention_factors else 1.0,
            }
        )
    added = pd.DataFrame(profile_rows)
    if added.empty:
        return legacy
    legacy = legacy.loc[~legacy["election_id"].isin(added["election_id"])] if not legacy.empty else legacy
    return pd.concat([legacy, added], ignore_index=True, sort=False)


def _all_ballot_rows(withdrawal_profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble A/B/C ballot rows without old model-scope exclusions.

    The standardized source marks some genuine minor ballot candidates as
    inactive because the old evaluation normalized over major slots. For slot
    discovery they must remain in the candidate denominator. Coalition events
    are disabled here because withdrawn candidates are not ballot candidates;
    transfer reservoirs remain a separate downstream mechanism.
    """

    results_path = ROOT / engine.RESULTS
    results = pd.read_csv(results_path)
    results.loc[results["slot"].ne("alpha"), "is_active_slot"] = True
    coalition_headers = pd.DataFrame(
        columns=[
            "election_id",
            "available_date",
            "event_type",
            "source_slot",
            "target_slot",
            "transfer_rate",
            "voter_compliance",
            "source_viability_after_event",
            "exclude_source_from_evaluation",
        ]
    )
    withdrawn_headers = pd.DataFrame(
        columns=[
            "election_id",
            "candidate_name",
            "target_slot",
            "viability",
            "transfer_rate",
            "voter_compliance",
            "available_date",
            "confidence",
        ]
    )
    original_results = engine.RESULTS
    original_coalitions = engine.COALITION_EVENTS
    original_withdrawn = engine.WITHDRAWN_CANDIDATE_TRANSFERS
    original_registry = getattr(engine, "WITHDRAWAL_TRANSFER_REGISTRY", "")
    with tempfile.TemporaryDirectory(prefix="slot_assignment_") as temp_dir:
        temp = Path(temp_dir)
        result_copy = temp / "presidential_results_standardized.csv"
        coalition_copy = temp / "coalition_events.csv"
        withdrawn_copy = temp / "withdrawn_candidate_transfers.csv"
        results.to_csv(result_copy, index=False, encoding="utf-8-sig")
        coalition_headers.to_csv(coalition_copy, index=False, encoding="utf-8-sig")
        if original_registry:
            withdrawn_headers.to_csv(
                withdrawn_copy, index=False, encoding="utf-8-sig"
            )
        engine.RESULTS = str(result_copy)
        engine.COALITION_EVENTS = str(coalition_copy)
        if original_registry:
            engine.WITHDRAWN_CANDIDATE_TRANSFERS = str(withdrawn_copy)
        engine.WITHDRAWAL_TRANSFER_REGISTRY = ""
        try:
            frame = engine.assemble()
        finally:
            engine.RESULTS = original_results
            engine.COALITION_EVENTS = original_coalitions
            engine.WITHDRAWN_CANDIDATE_TRANSFERS = original_withdrawn
            engine.WITHDRAWAL_TRANSFER_REGISTRY = original_registry
    profiles = _withdrawal_profiles() if withdrawal_profiles is None else withdrawal_profiles
    for profile in profiles.itertuples(index=False):
        mask = frame["election_id"].eq(profile.election_id) & frame["slot"].eq(
            profile.source_slot
        )
        frame.loc[mask, "candidate_name"] = profile.candidate_name
        frame.loc[mask, "candidate_weight"] = profile.latent_candidate_weight
        frame.loc[mask, "latent_withdrawn_candidate"] = 1.0
    if "latent_withdrawn_candidate" not in frame.columns:
        frame["latent_withdrawn_candidate"] = 0.0
    else:
        frame["latent_withdrawn_candidate"] = pd.to_numeric(
            frame["latent_withdrawn_candidate"], errors="coerce"
        ).fillna(0.0)
    return frame


def _normalize_by_region(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    return engine.normalize_vote_share_predictions(frame, raw)


def _normalize_draws_by_region(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    result = np.zeros_like(raw, dtype=float)
    for indices in frame.groupby(["election_id", "region_id"], sort=False).indices.values():
        idx = np.asarray(indices, dtype=int)
        clipped = np.clip(raw[:, idx], 0.0, None)
        totals = clipped.sum(axis=1, keepdims=True)
        normalized = np.divide(
            clipped,
            totals,
            out=np.full_like(clipped, 1.0 / max(len(idx), 1)),
            where=totals > 1e-12,
        )
        result[:, idx] = normalized
    return result


def _apply_candidate_strength(
    frame: pd.DataFrame,
    shares: np.ndarray,
) -> np.ndarray:
    """Combine contextual shares with pre-election vote-conversion capacity.

    Candidate weight is assembly-derived and outcome-free. Squaring represents
    two multiplicative stages: political presence and conversion into a ballot
    choice. The small floor retains genuinely minor ballot candidates.
    """

    candidate_weight = pd.to_numeric(
        frame["candidate_weight"], errors="coerce"
    ).fillna(0.0).clip(lower=CANDIDATE_STRENGTH_FLOOR, upper=1.0).to_numpy(float)
    factor = candidate_weight**CANDIDATE_STRENGTH_EXPONENT
    if shares.ndim == 1:
        adjusted = shares * factor
        return _normalize_by_region(frame, adjusted)
    adjusted = shares * factor[None, :]
    return _normalize_draws_by_region(frame, adjusted)


def _stable_coefficient_draws(
    beta: np.ndarray,
    covariance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, 0.0, None)
    stable = (eigenvectors * clipped) @ eigenvectors.T
    return rng.multivariate_normal(beta, stable, size=MONTE_CARLO_DRAWS)


def _prior_region_weights(
    target: str,
    test: pd.DataFrame,
    history: pd.DataFrame,
    order: tuple[str, ...],
) -> pd.Series:
    target_index = order.index(target)
    if target_index == 0:
        return pd.Series(1.0, index=test.index)
    prior_id = order[target_index - 1]
    prior = history.loc[history["election_id"].eq(prior_id)]
    if prior.empty or "votes" not in prior.columns:
        return pd.Series(1.0, index=test.index)
    region_volume = pd.to_numeric(prior["votes"], errors="coerce").fillna(0.0).groupby(
        prior["region_id"]
    ).sum()
    positive = region_volume.loc[region_volume.gt(0.0)]
    # Warmup shares do not contain turnout counts, so equal region weights are
    # more honest than pretending those shares are population volumes.
    if positive.empty or float(positive.max() / positive.min()) < 1.5:
        return pd.Series(1.0, index=test.index)
    fallback = float(positive.min())
    mapped = test["region_id"].map(region_volume).fillna(fallback).clip(lower=1.0)
    return mapped / max(float(mapped.mean()), 1e-12)


def _aggregate_candidate_draws(
    test: pd.DataFrame,
    point: np.ndarray,
    draws: np.ndarray,
    weights: pd.Series,
) -> pd.DataFrame:
    candidate_columns = [
        "election_id",
        "region_id",
        "slot",
        "candidate_name",
        "candidate_weight",
    ]
    candidate_columns.extend(
        column
        for column in ["bloc", "major_party_core_eligible", "third_viability"]
        if column in test.columns
    )
    work = test[candidate_columns].copy()
    work["_point"] = point
    work["_weight"] = weights.to_numpy(float)
    rows: list[dict[str, object]] = []
    for (source_slot, candidate_name), group in work.groupby(
        ["slot", "candidate_name"], sort=True
    ):
        idx = group.index.to_numpy(int)
        candidate_weights = group["_weight"].to_numpy(float)
        candidate_weights = candidate_weights / max(float(candidate_weights.sum()), 1e-12)
        point_share = float(np.dot(group["_point"].to_numpy(float), candidate_weights))
        national_draws = draws[:, idx] @ candidate_weights
        if "major_party_core_eligible" in group.columns:
            major_party_core_eligible = bool(
                group["major_party_core_eligible"]
                .fillna(False)
                .astype(bool)
                .mean()
                >= 0.5
            )
        else:
            source_bloc = group.get(
                "bloc", pd.Series("", index=group.index)
            ).astype(str).str.strip()
            major_party_core_eligible = bool(
                source_bloc.isin(electorate.MAJOR_PARTY_CORE_BLOCS).mean() >= 0.5
            )
        rows.append(
            {
                "source_slot": str(source_slot),
                "candidate_name": str(candidate_name),
                "structural_candidate_weight": float(
                    pd.to_numeric(group["candidate_weight"], errors="coerce").fillna(0.0).mean()
                ),
                "preliminary_mean_share": point_share,
                "preliminary_std": float(np.std(national_draws, ddof=1)),
                "lower_90": float(np.quantile(national_draws, 0.05)),
                "upper_90": float(np.quantile(national_draws, 0.95)),
                "third_viability_input": float(np.mean(national_draws >= 0.05)),
                "major_party_core_eligible": major_party_core_eligible,
                "automatic_third_viability": float(
                    pd.to_numeric(
                        group.get(
                            "third_viability",
                            pd.Series(0.0, index=group.index),
                        ),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    total = float(result["preliminary_mean_share"].sum())
    if total > 1e-12:
        result["preliminary_mean_share"] /= total
    return result


def _apply_withdrawal_redistribution(
    candidate: pd.DataFrame,
    withdrawal_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Convert pre-event candidate shares into post-event active vote shares."""

    out = candidate.copy()
    out["pre_withdrawal_mean_share"] = out["preliminary_mean_share"]
    out["post_withdrawal_mean_share"] = out["preliminary_mean_share"]
    out["withdrawal_received_share"] = 0.0
    out["withdrawal_unconverted_share"] = 0.0
    out["withdrawal_event_applied"] = False
    for profile in withdrawal_profiles.itertuples(index=False):
        if str(profile.election_id) not in set(out["election_id"].astype(str)):
            continue
        election_mask = out["election_id"].astype(str).eq(str(profile.election_id))
        election = out.loc[election_mask]
        shares = {
            str(row.source_slot): float(row.pre_withdrawal_mean_share)
            for row in election.itertuples(index=False)
        }
        if str(profile.source_slot) not in shares:
            continue
        post, unconverted = redistribute_withdrawn_vote_mass(
            shares,
            source_slot=str(profile.source_slot),
            target_fractions=dict(profile.target_fractions),
        )
        for index in election.index:
            slot = str(out.at[index, "source_slot"])
            out.at[index, "withdrawal_event_applied"] = True
            out.at[index, "post_withdrawal_mean_share"] = post.get(slot, 0.0)
            if slot == str(profile.source_slot):
                out.at[index, "candidate_status"] = "withdrawn"
                out.at[index, "withdrawal_unconverted_share"] = unconverted
                out.at[index, "available_date"] = str(profile.available_date)
                continue
            before = float(out.at[index, "preliminary_mean_share"])
            after = float(post.get(slot, before))
            out.at[index, "withdrawal_received_share"] = (
                shares[str(profile.source_slot)]
                * float(dict(profile.target_fractions).get(slot, 0.0))
            )
            out.at[index, "preliminary_mean_share"] = after
            delta = after - before
            out.at[index, "lower_90"] = float(np.clip(out.at[index, "lower_90"] + delta, 0.0, 1.0))
            out.at[index, "upper_90"] = float(np.clip(out.at[index, "upper_90"] + delta, 0.0, 1.0))
    return out


def build(*, role_aware: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    withdrawal_profiles = _withdrawal_profiles()
    ballot = _all_ballot_rows(withdrawal_profiles).reset_index(drop=True)
    warmup = engine.historical_presidential_warmup_frame().copy()
    warmup = warmup.loc[warmup["election_id"].isin(WARMUP_ELECTIONS)]
    full = pd.concat([warmup, ballot], ignore_index=True, sort=False).copy()
    order = (*WARMUP_ELECTIONS, *ELECTIONS)
    order_lookup = {election_id: index for index, election_id in enumerate(order)}
    full["_order"] = full["election_id"].map(order_lookup)
    for predictor in PREDICTORS:
        full[predictor] = pd.to_numeric(full[predictor], errors="coerce").fillna(0.0)

    assignments: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for target in ELECTIONS:
        target_order = order_lookup[target]
        train = full.loc[full["_order"] < target_order].copy().reset_index(drop=True)
        test = full.loc[full["election_id"].eq(target)].copy().reset_index(drop=True)
        if train.empty or test.empty:
            raise RuntimeError(f"missing strict rolling rows for {target}")
        train["_target"] = engine.normalized_vote_share_target(train)
        x_train = train[list(PREDICTORS)].to_numpy(float)
        x_test = test[list(PREDICTORS)].to_numpy(float)
        beta, _, covariance, residuals, means, scales = engine.ridge_fit(
            x_train,
            train["_target"].to_numpy(float),
            alpha=RIDGE_ALPHA,
            sample_weight=engine.election_epoch_sample_weight(train),
        )
        point_raw = engine.ridge_predict(beta, x_test, means, scales)
        point = _apply_candidate_strength(test, _normalize_by_region(test, point_raw))
        x_scaled = (x_test - means) / scales
        design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        beta_draws = _stable_coefficient_draws(beta, covariance, rng)
        raw_draws = beta_draws @ design.T
        normalized_draws = _apply_candidate_strength(
            test, _normalize_draws_by_region(test, raw_draws)
        )
        weights = _prior_region_weights(target, test, full, order)
        candidate = _aggregate_candidate_draws(test, point, normalized_draws, weights)
        cutoff = date.fromisoformat(engine.ELECTION_DATES[target]) - timedelta(days=1)
        candidate.insert(0, "election_id", target)
        candidate["candidate_id"] = candidate.apply(
            lambda row: f"{target}:{row['candidate_name']}", axis=1
        )
        candidate["candidate_status"] = "active_ballot"
        candidate["forecast_date"] = cutoff.isoformat()
        candidate["available_date"] = cutoff.isoformat()
        candidate = _apply_withdrawal_redistribution(candidate, withdrawal_profiles)
        assigned = assign_preliminary_slots(candidate, PreliminarySlotConfig())
        if role_aware:
            assigned = assign_role_aware_slots(assigned)
        assignments.append(assigned)
        audit_rows.append(
            {
                "target_election": target,
                "training_elections": "|".join(
                    election_id for election_id in order if order_lookup[election_id] < target_order
                ),
                "train_rows": len(train),
                "test_rows": len(test),
                "candidate_count": int(candidate["candidate_id"].nunique()),
                "target_rows_used_for_fit": 0,
                "target_outcome_columns_passed_to_assignment": False,
                "residual_sigma": float(np.std(residuals, ddof=1)),
                "region_weight_source": "immediately_prior_presidential_votes_or_equal_warmup",
            }
        )

    combined = pd.concat(assignments, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    summary = {
        "scope": {
            "warmup_elections": list(WARMUP_ELECTIONS),
            "assigned_elections": list(ELECTIONS),
            "post_2022_outcomes_used": False,
            "target_election_outcomes_used_for_assignment": False,
        },
        "predictors": list(PREDICTORS),
        "excluded_slot_predictors": sorted(
            set(engine.PREDICTORS).difference(PREDICTORS)
        ),
        "ridge_alpha": RIDGE_ALPHA,
        "candidate_strength": {
            "source": "assembly-derived candidate_weight",
            "exponent": CANDIDATE_STRENGTH_EXPONENT,
            "floor": CANDIDATE_STRENGTH_FLOOR,
            "target_outcomes_used": False,
        },
        "monte_carlo_draws": MONTE_CARLO_DRAWS,
        "uncertainty_scope": "ridge coefficient uncertainty; not final forecast interval",
        "candidate_scope": "standardized A/B/C ballot candidates; alpha aggregate excluded",
        "withdrawn_candidate_policy": "separate transfer reservoir; not assigned C",
        "withdrawn_candidate_share_policy": {
            "pre_event_share": "same strict preliminary Ridge plus latent candidate weight",
            "target_fraction": "transfer_rate times voter_compliance",
            "support_withdrawal": "common strength-times-reach time decay attenuates the prior target transfer without restoring C",
            "event_profile_source": (
                str(getattr(engine, "WITHDRAWAL_TRANSFER_REGISTRY", ""))
                or str(WITHDRAWAL_EVENT_PROFILES.relative_to(ROOT))
            ),
            "unconverted_mass": "removed before final valid-vote normalization",
            "target_outcomes_used": False,
        },
        "status": "experimental; not yet wired into the production forecast",
        "role_aware_assignment": bool(role_aware),
    }
    return combined, audit, summary


def main() -> None:
    assignments, audit, summary = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        OUTPUT_DIR / "candidate_slot_assignments_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(OUTPUT_DIR / "fold_audit.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    display = assignments[
        [
            "election_id",
            "candidate_name",
            "preliminary_mean_share",
            "lower_90",
            "upper_90",
            "preliminary_rank",
            "assigned_slot",
            "competition_role",
            "third_viability",
        ]
    ].copy()
    for column in ["preliminary_mean_share", "lower_90", "upper_90", "third_viability"]:
        display[column] = display[column].astype(float).round(4)
    print(display.to_string(index=False))
    print("\n" + audit.to_string(index=False))


if __name__ == "__main__":
    main()
