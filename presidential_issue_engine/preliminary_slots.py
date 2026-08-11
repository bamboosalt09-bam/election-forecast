"""Outcome-blind assignment of candidate competition slots.

The compiler consumes preliminary expected shares. It never consumes actual
target-election votes or realized candidate ranks. A and B are the two highest
preliminary candidates; the highest remaining active candidate is C. The C
label is descriptive only. Its weight is a continuous probability of clearing
a reference share.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from statistics import NormalDist

import numpy as np
import pandas as pd


SLOT_DERIVED_PREDICTORS = frozenset(
    {"slot_A", "slot_B", "slotA_prior", "slotB_prior"}
)

DEFAULT_SLOT_FREE_PREDICTORS = (
    "issue_advantage",
    "rif",
    "partisan_prior",
    "landscape_bloc_alignment",
    "landscape_centrist",
    "landscape_inferred_prior",
)

FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    {
        "actual",
        "actual_share",
        "actual_vote_share",
        "candidate_votes",
        "contest_votes",
        "vote_share",
        "votes",
        "realized_rank",
        "winner",
    }
)

ACTIVE_STATUSES = frozenset({"active", "active_ballot", "ballot", "registered"})
WITHDRAWN_STATUSES = frozenset({"withdrawn", "unified", "endorsed_other"})
INACTIVE_STATUSES = frozenset({"inactive", "disqualified", "not_on_ballot"})


@dataclass(frozen=True)
class PreliminarySlotConfig:
    """Configuration for continuous third-candidate classification."""

    third_reference_share: float = 0.05
    major_third_probability: float = 0.50
    fallback_viability_scale: float = 0.02
    rule_version: str = "preliminary_expected_share_v1"


def classify_competition_regime(
    first_share: float,
    second_share: float,
    third_share: float,
    third_viability: float,
) -> str:
    """Classify the Korean presidential field without allowing three equals."""

    first = max(float(first_share), 0.0)
    second = max(float(second_share), 1e-12)
    third = max(float(third_share), 0.0)
    third_to_second = third / second
    first_to_second = first / second
    if float(third_viability) < 0.5 or third_to_second < 0.20:
        return "two_strong_one_weak"
    if first_to_second >= 1.25 and third_to_second >= 0.45:
        return "one_strong_two_medium"
    return "two_strong_one_medium"


def latent_withdrawn_candidate_weight(
    *,
    viability: float,
    centrist_appeal: float,
    anti_major_party_appeal: float,
    regional_base_overlap: float,
    confidence: float,
) -> float:
    """Convert a pre-withdrawal profile into the standard candidate weight.

    The preliminary model later squares candidate weight. Returning the square
    root of attention times conversion therefore preserves the two structural
    stages used by the active-candidate path without using vote outcomes.
    """

    viability = float(np.clip(viability, 0.0, 1.0))
    centrist = float(np.clip(centrist_appeal, 0.0, 1.0))
    anti_major = float(np.clip(anti_major_party_appeal, 0.0, 1.0))
    overlap = float(np.clip(regional_base_overlap, 0.0, 1.0))
    confidence = float(np.clip(confidence, 0.0, 1.0))
    attention = viability * (0.55 + 0.25 * centrist + 0.20 * anti_major) * confidence
    conversion = float(
        np.clip(0.20 + 0.35 * viability + 0.25 * overlap + 0.20 * centrist, 0.0, 1.0)
    )
    return float(np.sqrt(max(attention * conversion, 0.0)))


def redistribute_withdrawn_vote_mass(
    shares_by_slot: dict[str, float],
    *,
    source_slot: str,
    target_fractions: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Redistribute a latent candidate's expected share and renormalize votes.

    Fractions are unconditional voter-compliance shares. Any source mass not
    transferred to a target is treated as abstention or an unscored minor-vote
    reservoir and is removed before final valid-vote normalization.
    """

    source = str(source_slot)
    pre = {str(slot): max(float(value), 0.0) for slot, value in shares_by_slot.items()}
    source_mass = pre.get(source, 0.0)
    fractions = {
        str(slot): max(float(value), 0.0)
        for slot, value in target_fractions.items()
        if str(slot) != source
    }
    fraction_total = sum(fractions.values())
    if fraction_total > 1.0:
        fractions = {slot: value / fraction_total for slot, value in fractions.items()}
        fraction_total = 1.0
    post = dict(pre)
    post[source] = 0.0
    for slot, fraction in fractions.items():
        post[slot] = post.get(slot, 0.0) + source_mass * fraction
    active_total = sum(value for slot, value in post.items() if slot != source)
    if active_total > 1e-12:
        post = {
            slot: (0.0 if slot == source else value / active_total)
            for slot, value in post.items()
        }
    unconverted = source_mass * max(1.0 - fraction_total, 0.0)
    return post, float(unconverted)


def attenuate_withdrawn_endorsement_transfer(
    target_fractions: dict[str, float],
    *,
    target_slot: str,
    event_strength: float,
    voter_reach: float,
    days_to_election: float,
    decay_days: float = 14.0,
    maximum_fraction_loss: float = 0.50,
) -> tuple[dict[str, float], float]:
    """Attenuate a prior unification transfer after support is withdrawn.

    The candidate remains withdrawn. Only the expected transfer to the former
    endorsed target is reduced. The common maximum-loss and time-decay rule is
    deliberately conservative and contains no election-result input.
    """

    strength = float(np.clip(event_strength, 0.0, 1.0))
    reach = float(np.clip(voter_reach, 0.0, 1.0))
    days = max(float(days_to_election), 0.0)
    decay = max(float(decay_days), 1e-9)
    max_loss = float(np.clip(maximum_fraction_loss, 0.0, 1.0))
    recency = exp(-days / decay)
    retention = float(np.clip(1.0 - max_loss * strength * reach * recency, 0.0, 1.0))
    target = str(target_slot)
    adjusted = {
        str(slot): max(float(value), 0.0) * (retention if str(slot) == target else 1.0)
        for slot, value in target_fractions.items()
    }
    return adjusted, retention


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-9, 1.0 - 1e-9))
    return float(np.log(clipped / (1.0 - clipped)))


def _expit(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-min(value, 700.0))))
    exponent = float(np.exp(max(value, -700.0)))
    return exponent / (1.0 + exponent)


def apply_hierarchical_third_constraint(
    frame: pd.DataFrame,
    predictions: np.ndarray | pd.Series,
    *,
    region_weights: np.ndarray | pd.Series | None = None,
    prior_log_odds_weight: float = 0.25,
    absolute_third_cap: float = 0.30,
    third_to_second_cap: float = 0.95,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Pool C with its preliminary prior while preserving the A:B ratio.

    The constraint weakly regularizes a symmetric three-way field. It does not
    force an A/B winner: their relative prediction remains unchanged. C can
    remain a meaningful medium candidate and can approach, but not exceed, B.
    """

    required = {"election_id", "region_id", "slot", "preliminary_mean_share"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing hierarchy columns: {missing}")
    values = np.asarray(predictions, dtype=float).copy()
    if len(values) != len(frame):
        raise ValueError("prediction length must match hierarchy frame")
    blend = float(np.clip(prior_log_odds_weight, 0.0, 1.0))
    absolute_cap = float(np.clip(absolute_third_cap, 0.0, 1.0))
    relative_cap = max(float(third_to_second_cap), 0.0)
    weights = (
        np.ones(len(frame), dtype=float)
        if region_weights is None
        else np.asarray(region_weights, dtype=float)
    )
    if len(weights) != len(frame):
        raise ValueError("region_weights length must match hierarchy frame")
    diagnostics: list[dict[str, object]] = []
    for election_id, indices in frame.groupby("election_id", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        group = frame.loc[idx].copy().reset_index(drop=True)
        slots = group["slot"].astype(str)
        if not {"A", "B", "C"}.issubset(set(slots)):
            continue
        local_values = values[idx].copy()
        local_weights = np.clip(weights[idx], 0.0, None)
        region_groups = list(group.groupby("region_id", sort=False).indices.values())
        for region_index in region_groups:
            region_index = np.asarray(region_index, dtype=int)
            clipped = np.clip(local_values[region_index], 0.0, None)
            total = float(clipped.sum())
            local_values[region_index] = (
                clipped / total
                if total > 1e-12
                else np.repeat(1.0 / len(region_index), len(region_index))
            )

        def national_share(candidate_slot: str, candidate_values: np.ndarray) -> float:
            candidate_index = np.flatnonzero(slots.eq(candidate_slot).to_numpy())
            candidate_weights = local_weights[candidate_index]
            if float(candidate_weights.sum()) <= 1e-12:
                candidate_weights = np.ones(len(candidate_index), dtype=float)
            return float(np.average(candidate_values[candidate_index], weights=candidate_weights))

        c_pred = national_share("C", local_values)
        a_pred = national_share("A", local_values)
        b_pred = national_share("B", local_values)
        candidate_prior = (
            group.groupby("slot")["preliminary_mean_share"]
            .mean()
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
        prior_total = float(candidate_prior.sum())
        c_prior = (
            float(candidate_prior.get("C", 0.0) / prior_total)
            if prior_total > 1e-12
            else c_pred
        )
        pooled = _expit((1.0 - blend) * _logit(c_pred) + blend * _logit(c_prior))
        ab_total = a_pred + b_pred
        if ab_total <= 1e-12:
            continue
        b_within_ab = b_pred / ab_total
        relative_share_cap = relative_cap * b_within_ab / (1.0 + relative_cap * b_within_ab)
        target_c = min(pooled, absolute_cap, relative_share_cap)
        c_mask = slots.eq("C").to_numpy()

        def apply_multiplier(multiplier: float) -> tuple[np.ndarray, float]:
            candidate = local_values.copy()
            candidate[c_mask] *= multiplier
            for region_index in region_groups:
                region_index = np.asarray(region_index, dtype=int)
                total = float(candidate[region_index].sum())
                if total > 1e-12:
                    candidate[region_index] /= total
            return candidate, national_share("C", candidate)

        low, high = 0.0, 1.0
        _, high_share = apply_multiplier(high)
        while high_share < target_c and high < 1e6:
            high *= 2.0
            _, high_share = apply_multiplier(high)
        adjusted, achieved = local_values.copy(), c_pred
        for _ in range(80):
            midpoint = (low + high) / 2.0
            candidate, current = apply_multiplier(midpoint)
            adjusted, achieved = candidate, current
            if abs(current - target_c) < 1e-12:
                break
            if current < target_c:
                low = midpoint
            else:
                high = midpoint
        values[idx] = adjusted
        diagnostics.append(
            {
                "election_id": election_id,
                "third_prediction_before": c_pred,
                "third_preliminary_prior": c_prior,
                "third_log_odds_pooled": pooled,
                "third_prediction_after": achieved,
                "absolute_cap": absolute_cap,
                "third_to_second_cap": relative_cap,
            }
        )
    return values, pd.DataFrame(diagnostics)


def slot_free_predictors(predictors: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Remove predictors whose meaning depends on a realized A/B/C assignment."""

    return tuple(value for value in predictors if value not in SLOT_DERIVED_PREDICTORS)


def _logistic_probability(mean: float, threshold: float, scale: float) -> float:
    z = (mean - threshold) / max(scale, 1e-9)
    if z >= 0:
        return float(1.0 / (1.0 + exp(-min(z, 700.0))))
    ez = exp(max(z, -700.0))
    return float(ez / (1.0 + ez))


def third_viability_probability(
    mean_share: float,
    *,
    standard_deviation: float | None = None,
    reference_share: float = 0.05,
    fallback_scale: float = 0.02,
) -> float:
    """Return P(candidate share >= reference share)."""

    mean = float(mean_share)
    if standard_deviation is not None and np.isfinite(standard_deviation):
        std = float(standard_deviation)
        if std > 1e-12:
            z = (float(reference_share) - mean) / std
            return float(np.clip(1.0 - NormalDist().cdf(z), 0.0, 1.0))
    return _logistic_probability(mean, float(reference_share), float(fallback_scale))


def _validate_input(frame: pd.DataFrame) -> None:
    required = {
        "election_id",
        "candidate_id",
        "candidate_name",
        "preliminary_mean_share",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing preliminary slot columns: {missing}")
    forbidden = sorted(FORBIDDEN_OUTCOME_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise ValueError(
            "target outcome columns are forbidden in slot assignment input: "
            + ", ".join(forbidden)
        )
    duplicates = frame.duplicated(["election_id", "candidate_id"], keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, ["election_id", "candidate_id"]].drop_duplicates()
        raise ValueError(f"candidate rows must be unique: {keys.to_dict('records')}")


def assign_preliminary_slots(
    frame: pd.DataFrame,
    config: PreliminarySlotConfig | None = None,
) -> pd.DataFrame:
    """Assign A/B/C from preliminary shares without changing the denominator."""

    config = config or PreliminarySlotConfig()
    work = frame.copy()
    _validate_input(work)
    if work.empty:
        return work.assign(
            preliminary_rank=pd.Series(dtype="Int64"),
            assigned_slot=pd.Series(dtype="string"),
            competition_role=pd.Series(dtype="string"),
            third_viability=pd.Series(dtype=float),
            assignment_rule_version=pd.Series(dtype="string"),
        )

    work["preliminary_mean_share"] = pd.to_numeric(
        work["preliminary_mean_share"], errors="raise"
    ).clip(lower=0.0)
    if "preliminary_std" not in work.columns:
        work["preliminary_std"] = np.nan
    else:
        work["preliminary_std"] = pd.to_numeric(
            work["preliminary_std"], errors="coerce"
        ).clip(lower=0.0)
    if "candidate_status" not in work.columns:
        work["candidate_status"] = "active_ballot"
    work["candidate_status"] = (
        work["candidate_status"].fillna("active_ballot").astype(str).str.lower()
    )

    if "forecast_date" in work.columns and "available_date" in work.columns:
        forecast = pd.to_datetime(work["forecast_date"], errors="raise")
        available = pd.to_datetime(work["available_date"], errors="raise")
        if (available > forecast).any():
            bad = work.loc[
                available > forecast,
                ["election_id", "candidate_id", "available_date", "forecast_date"],
            ]
            raise ValueError(
                "post-cutoff candidate input detected: " + str(bad.to_dict("records"))
            )

    work["preliminary_rank"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
    work["assigned_slot"] = "inactive"
    work["competition_role"] = "inactive"
    work["third_viability"] = 0.0
    work["competition_regime"] = "inactive"

    for _, indices in work.groupby("election_id", sort=False).groups.items():
        group_index = list(indices)
        status = work.loc[group_index, "candidate_status"]
        withdrawn = status.isin(WITHDRAWN_STATUSES)
        inactive = status.isin(INACTIVE_STATUSES)
        unknown = ~(status.isin(ACTIVE_STATUSES) | withdrawn | inactive)
        if unknown.any():
            unknown_values = sorted(status.loc[unknown].unique())
            raise ValueError(f"unknown candidate_status values: {unknown_values}")

        work.loc[status.index[withdrawn], "assigned_slot"] = "withdrawn"
        work.loc[status.index[withdrawn], "competition_role"] = "transfer_reservoir"

        active_index = status.index[status.isin(ACTIVE_STATUSES)]
        if len(active_index) == 0:
            continue
        active = work.loc[active_index].sort_values(
            ["preliminary_mean_share", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        active_order = list(active.index)
        work.loc[active_order, "preliminary_rank"] = np.arange(1, len(active_order) + 1)
        work.loc[active_order, "assigned_slot"] = "alpha"
        work.loc[active_order, "competition_role"] = "minor_active"

        if active_order:
            work.loc[active_order[0], ["assigned_slot", "competition_role"]] = [
                "A",
                "major_candidate",
            ]
        if len(active_order) >= 2:
            work.loc[active_order[1], ["assigned_slot", "competition_role"]] = [
                "B",
                "major_candidate",
            ]
        if len(active_order) >= 3:
            third_index = active_order[2]
            row = work.loc[third_index]
            supplied = row.get("third_viability_input", np.nan)
            if pd.notna(supplied):
                viability = float(np.clip(float(supplied), 0.0, 1.0))
            else:
                std = row["preliminary_std"]
                viability = third_viability_probability(
                    float(row["preliminary_mean_share"]),
                    standard_deviation=None if pd.isna(std) else float(std),
                    reference_share=config.third_reference_share,
                    fallback_scale=config.fallback_viability_scale,
                )
            role = (
                "major_third"
                if viability >= config.major_third_probability
                else "minor_third"
            )
            work.loc[third_index, ["assigned_slot", "competition_role"]] = ["C", role]
            work.loc[third_index, "third_viability"] = viability
            regime = classify_competition_regime(
                float(work.loc[active_order[0], "preliminary_mean_share"]),
                float(work.loc[active_order[1], "preliminary_mean_share"]),
                float(work.loc[third_index, "preliminary_mean_share"]),
                viability,
            )
            work.loc[group_index, "competition_regime"] = regime
        elif len(active_order) == 2:
            work.loc[group_index, "competition_regime"] = "two_candidate"

    work["assignment_rule_version"] = config.rule_version
    return work


def assign_role_aware_slots(
    frame: pd.DataFrame,
    *,
    major_eligibility_column: str = "major_party_core_eligible",
    automatic_viability_column: str = "automatic_third_viability",
) -> pd.DataFrame:
    """Separate forecast rank from major-party and third-candidate roles.

    The input is the output of :func:`assign_preliminary_slots`. Overall
    preliminary rank is retained for diagnostics, while A and B are assigned
    among candidates from the two major-party lineages. The strongest active
    non-major candidate becomes C. If the lineage evidence is incomplete, the
    original rank assignment is preserved.
    """

    required = {
        "election_id",
        "candidate_id",
        "candidate_status",
        "preliminary_mean_share",
        "assigned_slot",
        "competition_role",
        major_eligibility_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"role-aware assignment missing columns: {missing}")

    out = frame.copy()
    out["rank_slot"] = out["assigned_slot"].astype(str)
    out["political_role"] = "inactive"
    out["role_assignment_applied"] = False
    out["role_assignment_reason"] = "insufficient_major_party_lineage"
    eligibility = out[major_eligibility_column].fillna(False).astype(bool)
    status = out["candidate_status"].fillna("active_ballot").astype(str).str.lower()
    active = status.isin(ACTIVE_STATUSES)
    out.loc[active & eligibility, "political_role"] = "major_party_candidate"
    out.loc[active & ~eligibility, "political_role"] = "nonmajor_candidate"
    out.loc[status.isin(WITHDRAWN_STATUSES), "political_role"] = "withdrawn"

    for _, indices in out.groupby("election_id", sort=False).groups.items():
        group_index = pd.Index(indices)
        active_index = group_index[active.loc[group_index]]
        major_index = active_index[eligibility.loc[active_index]]
        nonmajor_index = active_index[~eligibility.loc[active_index]]
        if len(major_index) < 2:
            continue

        major_order = list(
            out.loc[major_index]
            .sort_values(
                ["preliminary_mean_share", "candidate_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            .index
        )
        nonmajor_order = list(
            out.loc[nonmajor_index]
            .sort_values(
                ["preliminary_mean_share", "candidate_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            .index
        )

        out.loc[active_index, "assigned_slot"] = "alpha"
        out.loc[active_index, "competition_role"] = "minor_active"
        out.loc[major_order[0], ["assigned_slot", "competition_role"]] = [
            "A",
            "major_candidate",
        ]
        out.loc[major_order[1], ["assigned_slot", "competition_role"]] = [
            "B",
            "major_candidate",
        ]

        if nonmajor_order:
            third_index = nonmajor_order[0]
            automatic_viability = pd.to_numeric(
                pd.Series([out.at[third_index, automatic_viability_column]])
                if automatic_viability_column in out.columns
                else pd.Series([np.nan]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(automatic_viability):
                viability = float(np.clip(automatic_viability, 0.0, 1.0))
            else:
                viability = float(
                    np.clip(out.at[third_index, "third_viability"], 0.0, 1.0)
                )
            role = "major_third" if viability >= 0.50 else "minor_third"
            out.loc[third_index, ["assigned_slot", "competition_role"]] = [
                "C",
                role,
            ]
            out.loc[third_index, "third_viability"] = viability
            regime = classify_competition_regime(
                float(out.at[major_order[0], "preliminary_mean_share"]),
                float(out.at[major_order[1], "preliminary_mean_share"]),
                float(out.at[third_index, "preliminary_mean_share"]),
                viability,
            )
        else:
            regime = "two_candidate"

        out.loc[group_index, "competition_regime"] = regime
        out.loc[group_index, "role_assignment_applied"] = True
        out.loc[group_index, "role_assignment_reason"] = (
            "major_party_lineage_then_preliminary_share"
        )

    out["assignment_rule_version"] = np.where(
        out["role_assignment_applied"],
        "political_role_then_preliminary_share_v1",
        out["assignment_rule_version"],
    )
    return out
