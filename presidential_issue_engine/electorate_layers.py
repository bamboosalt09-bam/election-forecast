"""Latent electorate layers for regional presidential vote forecasts.

This module is deliberately independent from the fitted Ridge engine.  It
decomposes the already available regional party terrain into durable core,
critical-support, and swing voting mass, then lets issue character affect each
mass with a different elasticity.  Every historical input is filtered to a
date strictly before the target election.

The layers are ecological latent quantities, not observed individual voters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from election_forecast.features.region_bloc_prior import (
    CONSERVATIVE_BLOC,
    INDEPENDENT_BLOC,
    LIBERAL_BLOC,
    PROGRESSIVE_BLOC,
    THIRD_BLOC,
    election_date,
    normalize_bloc,
)


ISSUE_CLASSES = (
    "economy",
    "housing",
    "integrity_candidate",
    "regime",
    "social",
    "security",
    "coalition",
    "regional",
)

CAMP_CONSERVATIVE = "camp_conservative"
CAMP_LIBERAL = "camp_liberal"
CAMP_PROGRESSIVE = "camp_progressive"
CAMP_CENTRIST = "camp_centrist"
CAMP_INDEPENDENT = "camp_independent"

ISSUE_CLASS_BY_NAME = {
    "economy_growth": "economy",
    "jobs_labor": "economy",
    "inflation_livelihood": "economy",
    "housing": "housing",
    "corruption_integrity": "integrity_candidate",
    "family_legal_risk": "integrity_candidate",
    "candidate_competence": "integrity_candidate",
    "gaffe_event": "integrity_candidate",
    "regime_change": "regime",
    "external_shock": "regime",
    "welfare_pension": "social",
    "education": "social",
    "gender_generation": "social",
    "security_nk": "security",
    "foreign_policy": "security",
    "unification_event": "coalition",
    "withdrawal_event": "coalition",
    "endorsement_event": "coalition",
    "regional_dev": "regional",
}

# Direct party ballots dominate durable-base estimation. Candidate ballots are
# retained at low weight so that 2002 still has pre-election evidence.
LAYER_ELECTION_TYPE_WEIGHTS = {
    "national_assembly_pr": 1.00,
    "assembly_pr": 1.00,
    "metro_council_pr": 0.80,
    "local_council_pr": 0.55,
    "presidential": 0.35,
    "national_assembly_district": 0.25,
    "assembly_district": 0.25,
    "metro_council_district": 0.18,
    "local_council_district": 0.10,
    "metro_governor": 0.06,
    "local_governor": 0.04,
    "education_superintendent": 0.0,
    "education_council": 0.0,
}

# These ballots ask voters to choose a party directly. They therefore provide
# the cleanest evidence for durable party support. Candidate ballots remain a
# fallback and stabilizer, especially for early targets such as 2002 where the
# direct-party history is short.
DIRECT_PARTY_ELECTION_TYPES = frozenset(
    {
        "national_assembly_pr",
        "assembly_pr",
        "metro_council_pr",
        "local_council_pr",
    }
)

DIRECT_PARTY_EVIDENCE_PRIOR = 2.0
DIRECT_PARTY_CORE_LCB_Z = 1.5
MAJOR_PARTY_CORE_BLOCS = frozenset({CONSERVATIVE_BLOC, LIBERAL_BLOC})
# Raw shifts are capped below the 3pp final-mass policy because regional
# compositional normalization can amplify a raw shift slightly.
MAX_LAYER_RECLASSIFICATION = 0.025

REGIONALIST_PARTY_LABELS = frozenset(
    {
        "자유민주연합",
        "자민련",
        "국민중심당",
        "자유선진당",
        "충청의미래당",
        "민주평화당",
        "대안신당",
    }
)
REFORM_PARTY_LABELS = frozenset(
    {
        "국민의당",
        "바른미래당",
        "개혁신당",
        "국민생각",
        "새정치개혁당",
        "새로운미래",
    }
)


PREFERENCE_SENSITIVITY = {
    "core": {
        "economy": 0.15,
        "housing": 0.15,
        "integrity_candidate": 0.20,
        "regime": 0.30,
        "social": 0.15,
        "security": 0.20,
        "coalition": 0.10,
        "regional": 0.20,
    },
    "critical": {
        "economy": 0.60,
        "housing": 0.65,
        "integrity_candidate": 0.75,
        "regime": 0.85,
        "social": 0.55,
        "security": 0.55,
        "coalition": 0.90,
        "regional": 0.65,
    },
    "swing": {
        "economy": 1.00,
        "housing": 0.95,
        "integrity_candidate": 1.00,
        "regime": 1.05,
        "social": 0.80,
        "security": 0.75,
        "coalition": 1.10,
        "regional": 0.85,
    },
}

TURNOUT_SENSITIVITY = {
    "core": {
        "economy": 0.12,
        "housing": 0.12,
        "integrity_candidate": 0.25,
        "regime": 0.50,
        "social": 0.18,
        "security": 0.25,
        "coalition": 0.18,
        "regional": 0.25,
    },
    "critical": {
        "economy": 0.35,
        "housing": 0.40,
        "integrity_candidate": 0.55,
        "regime": 0.80,
        "social": 0.40,
        "security": 0.45,
        "coalition": 0.65,
        "regional": 0.50,
    },
    "swing": {
        "economy": 0.55,
        "housing": 0.60,
        "integrity_candidate": 0.70,
        "regime": 0.95,
        "social": 0.55,
        "security": 0.55,
        "coalition": 0.85,
        "regional": 0.60,
    },
    "nonvoter": {
        "economy": 0.60,
        "housing": 0.65,
        "integrity_candidate": 0.75,
        "regime": 1.10,
        "social": 0.65,
        "security": 0.65,
        "coalition": 0.80,
        "regional": 0.70,
    },
}


@dataclass(frozen=True)
class ElectorateLayerConfig:
    """Low-dimensional gains applied to fixed, theory-constrained templates."""

    terrain_anchor_gain: float = 0.0
    camp_core_anchor_gain: float = 0.0
    camp_regional_lean_gain: float = 0.0
    camp_composition_gain: float = 0.0
    regional_accent_gain: float = 0.0
    regional_accent_signal_width: float = 0.10
    preference_gain: float = 0.0
    layer_separation: float = 0.0
    layer_response_profile: str = "combined"
    mass_profile: str = "legacy"
    turnout_gain: float = 0.0
    nonvoter_gain: float = 0.0

    @property
    def complexity(self) -> tuple[int, float]:
        values = (
            self.terrain_anchor_gain,
            self.camp_core_anchor_gain,
            self.camp_regional_lean_gain,
            self.camp_composition_gain,
            self.regional_accent_gain,
            self.preference_gain,
            self.layer_separation,
            self.turnout_gain,
            self.nonvoter_gain,
        )
        return sum(value > 0.0 for value in values), float(sum(values))


NEUTRAL_LAYER_CONFIG = ElectorateLayerConfig()


def _candidate_camp_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    """Infer candidate camps and bounded within-camp claims from prior metadata."""

    out = candidates.copy()
    normalized_bloc = out["bloc"].map(normalize_bloc)
    def numeric_series(column: str) -> pd.Series:
        value = pd.to_numeric(out.get(column, 0.0), errors="coerce")
        if not isinstance(value, pd.Series):
            value = pd.Series(float(value), index=out.index)
        return value.fillna(0.0).clip(0.0, 1.0)

    conservative = numeric_series("landscape_axis_conservative")
    liberal = numeric_series("landscape_axis_liberal")
    progressive = numeric_series("landscape_axis_progressive")
    centrist = numeric_series("landscape_axis_centrist")
    confidence = numeric_series("landscape_confidence")

    camps: list[str] = []
    origin_weights: list[float] = []
    for index, bloc in normalized_bloc.items():
        if bloc == "국민의힘":
            camps.append(CAMP_CONSERVATIVE)
            origin_weights.append(1.0)
            continue
        if bloc == "더불어민주당":
            camps.append(CAMP_LIBERAL)
            origin_weights.append(1.0)
            continue
        if bloc == "진보정당계":
            camps.append(CAMP_PROGRESSIVE)
            origin_weights.append(1.0)
            continue

        scores = {
            CAMP_CONSERVATIVE: float(conservative.loc[index]),
            CAMP_LIBERAL: float(liberal.loc[index]),
            CAMP_PROGRESSIVE: float(progressive.loc[index]),
            CAMP_CENTRIST: float(centrist.loc[index]),
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_camp, best_score = ranked[0]
        margin = best_score - ranked[1][1]
        if best_camp != CAMP_CENTRIST and margin >= 0.08:
            camps.append(best_camp)
            evidence = np.clip(margin / 0.25, 0.0, 1.0) * float(confidence.loc[index])
            origin_weights.append(float(0.15 + 0.50 * evidence))
        elif bloc == THIRD_BLOC or best_score > 0.0:
            camps.append(CAMP_CENTRIST)
            origin_weights.append(1.0)
        else:
            camps.append(CAMP_INDEPENDENT)
            origin_weights.append(1.0)

    out["candidate_camp"] = camps
    out["candidate_camp_origin_weight"] = np.asarray(origin_weights, dtype=float)
    claim_total = out.groupby(
        ["election_id", "region_id", "candidate_camp"]
    )["candidate_camp_origin_weight"].transform("sum")
    out["candidate_camp_claim"] = (
        out["candidate_camp_origin_weight"] / claim_total.replace(0.0, np.nan)
    ).fillna(0.0)
    source_camp = normalized_bloc.map(
        {
            "국민의힘": CAMP_CONSERVATIVE,
            "더불어민주당": CAMP_LIBERAL,
            "진보정당계": CAMP_PROGRESSIVE,
            THIRD_BLOC: CAMP_CENTRIST,
        }
    ).fillna("")
    out["candidate_source_camp"] = source_camp
    axis_by_camp = {
        CAMP_CONSERVATIVE: conservative,
        CAMP_LIBERAL: liberal,
        CAMP_PROGRESSIVE: progressive,
        CAMP_CENTRIST: centrist,
    }
    for camp, axis in axis_by_camp.items():
        weights = pd.Series(0.0, index=out.index)
        official = source_camp.eq(camp)
        weights.loc[official] = 1.0
        nonmainstream = source_camp.eq("") | source_camp.eq(CAMP_CENTRIST)
        cross_claim = (0.10 + 0.50 * axis * confidence).clip(0.0, 0.65)
        weights.loc[nonmainstream & ~official] = cross_claim.loc[
            nonmainstream & ~official
        ]
        inferred_primary = out["candidate_camp"].eq(camp) & nonmainstream
        weights.loc[inferred_primary] = np.maximum(
            weights.loc[inferred_primary],
            out.loc[inferred_primary, "candidate_camp_origin_weight"],
        )
        total = weights.groupby(
            [out["election_id"], out["region_id"]]
        ).transform("sum")
        out[f"candidate_claim_{camp}"] = (
            weights / total.replace(0.0, np.nan)
        ).fillna(0.0)
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(weights.sum())
    index = int(np.searchsorted(np.cumsum(weights), cutoff, side="left"))
    return float(values[min(index, len(values) - 1)])


def _history_before_target(
    history: pd.DataFrame,
    target_election_id: str,
    date_resolver: Callable[[str], pd.Timestamp | None] = election_date,
    half_life_years: float = 12.0,
) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    target_date = date_resolver(str(target_election_id))
    if target_date is None:
        return history.iloc[0:0].copy()
    target_date = pd.Timestamp(target_date)
    out = history.copy()
    out["source_date"] = pd.to_datetime(
        out["election_id"].astype(str).map(date_resolver), errors="coerce"
    )
    out = out.loc[out["source_date"].notna() & (out["source_date"] < target_date)].copy()
    if out.empty:
        return out
    out["vote_share"] = pd.to_numeric(out["vote_share"], errors="coerce").fillna(0.0)
    source_bloc = out["bloc"].astype(str).str.strip()
    out["major_party_vote_share"] = out["vote_share"].where(
        source_bloc.isin(MAJOR_PARTY_CORE_BLOCS), 0.0
    )
    out["regionalist_vote_share"] = out["vote_share"].where(
        source_bloc.isin(REGIONALIST_PARTY_LABELS), 0.0
    )
    out["reform_vote_share"] = out["vote_share"].where(
        source_bloc.isin(REFORM_PARTY_LABELS), 0.0
    )
    out["bloc"] = source_bloc.map(normalize_bloc)
    out["data_quality_weight"] = pd.to_numeric(
        out.get("data_quality_weight", 1.0), errors="coerce"
    ).fillna(1.0)
    # Multiple parties can map to one historical bloc. Sum them before taking
    # a lower quantile; otherwise party fragmentation masquerades as volatility.
    out = out.groupby(
        ["election_id", "election_type", "source_date", "region_id", "bloc"],
        as_index=False,
    ).agg(
        vote_share=("vote_share", "sum"),
        major_party_vote_share=("major_party_vote_share", "sum"),
        regionalist_vote_share=("regionalist_vote_share", "sum"),
        reform_vote_share=("reform_vote_share", "sum"),
        data_quality_weight=("data_quality_weight", "mean"),
    )
    age_years = (target_date - out["source_date"]).dt.days.clip(lower=0) / 365.25
    out["type_weight"] = out["election_type"].map(LAYER_ELECTION_TYPE_WEIGHTS).fillna(0.0)
    out["time_weight"] = np.exp(-np.log(2.0) * age_years / max(half_life_years, 0.1))
    out["weight"] = out["type_weight"] * out["time_weight"] * out["data_quality_weight"]
    return out.loc[out["weight"] > 0.0].copy()


def _regional_accent_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Build PIT-safe, multi-axis regional composition and trend evidence."""

    axes = {
        "conservative": CONSERVATIVE_BLOC,
        "liberal": LIBERAL_BLOC,
        "progressive": PROGRESSIVE_BLOC,
        "centrist": THIRD_BLOC,
    }
    columns = ["region_id"]
    for axis in (*axes, "regionalist", "reform"):
        columns.extend(
            [
                f"regional_accent_{axis}_share",
                f"regional_accent_{axis}_trend",
                f"regional_accent_{axis}_volatility",
                f"regional_accent_{axis}_reliability",
            ]
        )
    if history.empty:
        return pd.DataFrame(columns=columns)

    direct = history.loc[
        history["election_type"].isin(DIRECT_PARTY_ELECTION_TYPES)
    ].copy()
    evidence = direct if not direct.empty else history.copy()
    election_region = evidence[
        ["election_id", "source_date", "region_id", "weight"]
    ].drop_duplicates(["election_id", "region_id"])
    for axis, bloc in axes.items():
        values = (
            evidence.loc[evidence["bloc"].eq(bloc)]
            .groupby(["election_id", "region_id"])["vote_share"]
            .sum()
            .rename(axis)
        )
        election_region = election_region.merge(
            values,
            on=["election_id", "region_id"],
            how="left",
        )
        election_region[axis] = election_region[axis].fillna(0.0)
    for axis, source in (
        ("regionalist", "regionalist_vote_share"),
        ("reform", "reform_vote_share"),
    ):
        values = (
            evidence.groupby(["election_id", "region_id"])[source]
            .sum()
            .rename(axis)
        )
        election_region = election_region.merge(
            values,
            on=["election_id", "region_id"],
            how="left",
        )
        election_region[axis] = election_region[axis].fillna(0.0)

    rows: list[dict[str, object]] = []
    for region_id, group in election_region.groupby("region_id", sort=False):
        group = group.sort_values("source_date")
        weights = group["weight"].to_numpy(float)
        total_weight = float(weights.sum())
        row: dict[str, object] = {"region_id": str(region_id)}
        for axis in (*axes, "regionalist", "reform"):
            values = group[axis].to_numpy(float)
            mean = float(np.average(values, weights=weights)) if total_weight > 0.0 else 0.0
            mad = (
                float(np.average(np.abs(values - mean), weights=weights))
                if total_weight > 0.0
                else 0.0
            )
            n_eff = total_weight**2 / max(float(np.square(weights).sum()), 1e-12)
            recent = float(values[-1]) if len(values) else mean
            reliability = n_eff / (n_eff + DIRECT_PARTY_EVIDENCE_PRIOR)
            reliability *= float(np.clip(1.0 - mad / max(mean + 0.05, 0.05), 0.20, 1.0))
            row[f"regional_accent_{axis}_share"] = float(np.clip(mean, 0.0, 1.0))
            row[f"regional_accent_{axis}_trend"] = float(
                np.clip(recent - mean, -0.35, 0.35)
            )
            row[f"regional_accent_{axis}_volatility"] = mad
            row[f"regional_accent_{axis}_reliability"] = float(
                np.clip(reliability, 0.0, 1.0)
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _attach_candidate_regional_accent(frame: pd.DataFrame) -> pd.DataFrame:
    """Match fine regional composition to candidate ideology without a national bonus."""

    out = frame.copy()
    axes = (
        "conservative",
        "liberal",
        "progressive",
        "centrist",
        "regionalist",
        "reform",
    )
    candidate_axes: dict[str, pd.Series] = {}
    for axis in axes:
        source = pd.to_numeric(
            out.get(f"landscape_axis_{axis}", pd.Series(0.0, index=out.index)),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0)
        candidate_axes[axis] = source
    normalized_bloc = out["bloc"].map(normalize_bloc)
    official = {
        "conservative": normalized_bloc.eq(CONSERVATIVE_BLOC).astype(float),
        "liberal": normalized_bloc.eq(LIBERAL_BLOC).astype(float),
        "progressive": normalized_bloc.eq(PROGRESSIVE_BLOC).astype(float),
        "centrist": normalized_bloc.eq(THIRD_BLOC).astype(float),
        "regionalist": pd.Series(0.0, index=out.index),
        "reform": pd.Series(0.0, index=out.index),
    }
    axis_total = sum(candidate_axes.values())
    score = pd.Series(0.0, index=out.index)
    reliability_score = pd.Series(0.0, index=out.index)
    volatility_score = pd.Series(0.0, index=out.index)
    weight_total = pd.Series(0.0, index=out.index)
    for axis in axes:
        ideological = candidate_axes[axis] / axis_total.replace(0.0, np.nan)
        ideological = ideological.fillna(0.0)
        weight = (0.65 * official[axis] + 0.35 * ideological).clip(0.0, 1.0)
        share = pd.to_numeric(
            out.get(f"regional_accent_{axis}_share", 0.0), errors="coerce"
        ).fillna(0.0)
        trend = pd.to_numeric(
            out.get(f"regional_accent_{axis}_trend", 0.0), errors="coerce"
        ).fillna(0.0)
        reliability = pd.to_numeric(
            out.get(f"regional_accent_{axis}_reliability", 0.0), errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        volatility = pd.to_numeric(
            out.get(f"regional_accent_{axis}_volatility", 0.0), errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        score += weight * (share + 0.50 * trend) * reliability
        reliability_score += weight * reliability
        volatility_score += weight * volatility
        weight_total += weight
    score = (score / weight_total.replace(0.0, np.nan)).fillna(0.0)
    reliability_score = (
        reliability_score / weight_total.replace(0.0, np.nan)
    ).fillna(0.0)
    volatility_score = (
        volatility_score / weight_total.replace(0.0, np.nan)
    ).fillna(0.0)
    candidate_mean = score.groupby([out["election_id"], out["slot"]]).transform("mean")
    candidate_centered = score - candidate_mean
    region_mean = candidate_centered.groupby(
        [out["election_id"], out["region_id"]]
    ).transform("mean")
    out["regional_accent_signal"] = (candidate_centered - region_mean).clip(-1.0, 1.0)
    out["regional_accent_reliability"] = reliability_score.clip(0.0, 1.0)
    out["regional_accent_volatility"] = volatility_score.clip(0.0, 1.0)
    return out


def _bloc_layer_summary(
    history: pd.DataFrame,
    mass_profile: str = "legacy",
) -> pd.DataFrame:
    columns = [
        "region_id",
        "bloc",
        "durable_core_raw",
        "recent_bloc_base",
        "critical_support_raw",
        "bloc_vote_volatility",
        "layer_effective_elections",
        "direct_party_core_raw",
        "candidate_ballot_core_raw",
        "direct_party_recent_base",
        "candidate_ballot_recent_base",
        "direct_party_effective_elections",
        "candidate_ballot_effective_elections",
        "direct_party_reliability",
        "candidate_personal_vote_raw",
        "candidate_conversion_gap_raw",
    ]
    profiles = {
        "legacy",
        "durable_floor",
        "broad_critical",
        "durable_floor_broad_critical",
        "direct_party_layers",
    }
    if mass_profile not in profiles:
        raise ValueError(f"unknown electorate mass profile: {mass_profile}")
    if history.empty:
        return pd.DataFrame(columns=columns)

    durable_floor = mass_profile in {"durable_floor", "durable_floor_broad_critical"}
    broad_critical = mass_profile in {"broad_critical", "durable_floor_broad_critical"}
    direct_party_layers = mass_profile == "direct_party_layers"
    core_quantile = 0.10 if durable_floor else 0.25

    def channel_summary(
        group: pd.DataFrame,
        *,
        direct_party: bool,
    ) -> dict[str, float] | None:
        if group.empty:
            return None
        values = group["vote_share"].to_numpy(float)
        core_values = pd.to_numeric(
            group.get("major_party_vote_share", 0.0), errors="coerce"
        )
        if not isinstance(core_values, pd.Series):
            core_values = pd.Series(float(core_values), index=group.index)
        core_values = core_values.fillna(0.0).clip(lower=0.0).to_numpy(float)
        weights = group["weight"].to_numpy(float)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            return None
        mean = float(np.average(values, weights=weights))
        mad = float(np.average(np.abs(values - mean), weights=weights))
        n_eff = total_weight**2 / max(float(np.square(weights).sum()), 1e-12)
        stable_floor = min(
            mean,
            _weighted_quantile(values, weights, core_quantile),
        )
        if direct_party and not durable_floor:
            stable_floor = min(
                mean,
                max(
                    stable_floor,
                    mean - DIRECT_PARTY_CORE_LCB_Z * mad / max(np.sqrt(n_eff), 1.0),
                ),
            )
        core_mean = float(np.average(core_values, weights=weights))
        core_mad = float(
            np.average(np.abs(core_values - core_mean), weights=weights)
        )
        lower_tail = min(
            core_mean,
            _weighted_quantile(core_values, weights, core_quantile),
        )
        core = lower_tail
        if direct_party and not durable_floor:
            # With repeated direct-party ballots, use an outcome-blind lower
            # confidence bound as well as the empirical lower quartile. More
            # evidence narrows uncertainty; a single election cannot promote
            # its mean into "concrete" support.
            lower_confidence_bound = core_mean - (
                DIRECT_PARTY_CORE_LCB_Z * core_mad / max(np.sqrt(n_eff), 1.0)
            )
            core = min(core_mean, max(lower_tail, lower_confidence_bound))
        return {
            "mean": float(np.clip(mean, 0.0, 1.0)),
            "core": float(np.clip(core, 0.0, 1.0)),
            "stable_floor": float(np.clip(stable_floor, 0.0, 1.0)),
            "mad": max(mad, 0.0),
            "n_eff": max(n_eff, 0.0),
        }

    def adjusted_core_and_persistence(
        mean: float,
        core: float,
        mad: float,
        n_eff: float,
        bloc: str,
    ) -> tuple[float, float]:
        evidence = n_eff / (n_eff + 2.0)
        persistence = np.clip(1.0 - mad / max(mean + 0.05, 0.05), 0.15, 0.90)
        persistence = float(0.25 * (1.0 - evidence) + persistence * evidence)
        if bloc not in MAJOR_PARTY_CORE_BLOCS:
            if bloc == INDEPENDENT_BLOC:
                persistence *= 0.25
            elif bloc == THIRD_BLOC:
                persistence *= 0.70
            return 0.0, persistence
        return core, persistence

    rows: list[dict[str, object]] = []
    for (region_id, bloc), group in history.groupby(["region_id", "bloc"], sort=False):
        direct = channel_summary(
            group.loc[group["election_type"].isin(DIRECT_PARTY_ELECTION_TYPES)],
            direct_party=True,
        )
        candidate = channel_summary(
            group.loc[~group["election_type"].isin(DIRECT_PARTY_ELECTION_TYPES)],
            direct_party=False,
        )
        if direct is None and candidate is None:
            continue
        legacy_reference: tuple[float, float, float, float, float] | None = None
        if direct is not None and candidate is not None:
            direct_reliability = direct["n_eff"] / (
                direct["n_eff"] + DIRECT_PARTY_EVIDENCE_PRIOR
            )
            legacy_mean = (
                direct_reliability * direct["mean"]
                + (1.0 - direct_reliability) * candidate["mean"]
            )
            legacy_core = (
                direct_reliability * direct["core"]
                + (1.0 - direct_reliability) * candidate["core"]
            )
            legacy_stable_floor = (
                direct_reliability * direct["stable_floor"]
                + (1.0 - direct_reliability) * candidate["stable_floor"]
            )
            legacy_mad = (
                direct_reliability * direct["mad"]
                + (1.0 - direct_reliability) * candidate["mad"]
            )
            legacy_n_eff = direct["n_eff"] + candidate["n_eff"]
            legacy_reference = (
                legacy_mean,
                legacy_core,
                legacy_stable_floor,
                legacy_mad,
                legacy_n_eff,
            )
            if direct_party_layers:
                # Party ballots define party attachment. Candidate ballots only
                # stabilize the uncertain lower floor; they never raise the
                # party base or turn candidate popularity into party loyalty.
                fallback_floor = min(direct["core"], candidate["core"])
                separated_core = (
                    direct_reliability * direct["core"]
                    + (1.0 - direct_reliability) * fallback_floor
                )
                mean = legacy_mean + float(
                    np.clip(
                        direct["mean"] - legacy_mean,
                        -MAX_LAYER_RECLASSIFICATION,
                        MAX_LAYER_RECLASSIFICATION,
                    )
                )
                core = legacy_core + float(
                    np.clip(
                        separated_core - legacy_core,
                        -MAX_LAYER_RECLASSIFICATION,
                        MAX_LAYER_RECLASSIFICATION,
                    )
                )
                stable_floor = (
                    direct_reliability * direct["stable_floor"]
                    + (1.0 - direct_reliability) * min(
                        direct["stable_floor"], candidate["stable_floor"]
                    )
                )
                mad = legacy_mad
                n_eff = legacy_n_eff
            else:
                mean, core, stable_floor, mad, n_eff = legacy_reference
        elif direct is not None:
            direct_reliability = 1.0
            mean, core, stable_floor, mad, n_eff = (
                direct["mean"],
                direct["core"],
                direct["stable_floor"],
                direct["mad"],
                direct["n_eff"],
            )
        else:
            direct_reliability = 0.0
            mean, core, stable_floor, mad, n_eff = (
                candidate["mean"],
                candidate["core"],
                candidate["stable_floor"],
                candidate["mad"],
                candidate["n_eff"],
            )
        core, persistence = adjusted_core_and_persistence(mean, core, mad, n_eff, bloc)
        if bloc in MAJOR_PARTY_CORE_BLOCS:
            critical_gap = max(mean - core, 0.0)
            critical = critical_gap if broad_critical else critical_gap * persistence
        else:
            critical_gap = max(mean - stable_floor, 0.0)
            critical = stable_floor + (
                critical_gap if broad_critical else critical_gap * persistence
            )
        if direct_party_layers and direct is not None and legacy_reference is not None:
            (
                legacy_mean,
                legacy_core,
                legacy_stable_floor,
                legacy_mad,
                legacy_n_eff,
            ) = legacy_reference
            legacy_core, legacy_persistence = adjusted_core_and_persistence(
                legacy_mean,
                legacy_core,
                legacy_mad,
                legacy_n_eff,
                bloc,
            )
            core = legacy_core + float(
                np.clip(
                    core - legacy_core,
                    -MAX_LAYER_RECLASSIFICATION,
                    MAX_LAYER_RECLASSIFICATION,
                )
            )
            if bloc in MAJOR_PARTY_CORE_BLOCS:
                critical_gap = max(mean - core, 0.0)
                critical_target = critical_gap
                legacy_critical = (
                    max(legacy_mean - legacy_core, 0.0) * legacy_persistence
                )
            else:
                critical_gap = max(mean - stable_floor, 0.0)
                critical_target = stable_floor + critical_gap * persistence
                legacy_critical_gap = max(
                    legacy_mean - legacy_stable_floor, 0.0
                )
                legacy_critical = (
                    legacy_stable_floor
                    + legacy_critical_gap * legacy_persistence
                )
            critical = legacy_critical + float(
                np.clip(
                    critical_target - legacy_critical,
                    -MAX_LAYER_RECLASSIFICATION,
                    MAX_LAYER_RECLASSIFICATION,
                )
            )
        candidate_conversion_gap = (
            0.0
            if direct is None or candidate is None
            else candidate["mean"] - direct["mean"]
        )
        rows.append(
            {
                "region_id": str(region_id),
                "bloc": str(bloc),
                "durable_core_raw": float(np.clip(core, 0.0, 1.0)),
                "recent_bloc_base": float(np.clip(mean, 0.0, 1.0)),
                "critical_support_raw": float(np.clip(critical, 0.0, 1.0)),
                "bloc_vote_volatility": mad,
                "layer_effective_elections": n_eff,
                "direct_party_core_raw": 0.0 if direct is None else direct["core"],
                "candidate_ballot_core_raw": 0.0 if candidate is None else candidate["core"],
                "direct_party_recent_base": 0.0 if direct is None else direct["mean"],
                "candidate_ballot_recent_base": 0.0 if candidate is None else candidate["mean"],
                "direct_party_effective_elections": 0.0 if direct is None else direct["n_eff"],
                "candidate_ballot_effective_elections": (
                    0.0 if candidate is None else candidate["n_eff"]
                ),
                "direct_party_reliability": float(np.clip(direct_reliability, 0.0, 1.0)),
                "candidate_personal_vote_raw": float(max(candidate_conversion_gap, 0.0)),
                "candidate_conversion_gap_raw": float(candidate_conversion_gap),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def estimate_electorate_layers(
    candidate_frame: pd.DataFrame,
    history: pd.DataFrame,
    date_resolver: Callable[[str], pd.Timestamp | None] = election_date,
    mass_profile: str = "legacy",
) -> pd.DataFrame:
    """Attach PIT-safe candidate-aligned core/critical/swing voting mass."""

    required = {"election_id", "region_id", "slot", "bloc"}
    missing = required - set(candidate_frame.columns)
    if missing:
        raise ValueError(f"candidate frame missing layer keys: {sorted(missing)}")
    outputs: list[pd.DataFrame] = []
    for election_id, candidates in candidate_frame.groupby("election_id", sort=False):
        candidates = candidates.copy()
        source_bloc = candidates["bloc"].astype(str).str.strip()
        candidates["major_party_core_eligible"] = source_bloc.isin(
            MAJOR_PARTY_CORE_BLOCS
        )
        candidates["bloc"] = candidates["bloc"].map(normalize_bloc)
        eligible = _history_before_target(history, str(election_id), date_resolver=date_resolver)
        summary = _bloc_layer_summary(eligible, mass_profile=mass_profile)
        merged = candidates.merge(summary, on=["region_id", "bloc"], how="left")
        accent = _regional_accent_summary(eligible)
        merged = merged.merge(accent, on="region_id", how="left")
        numeric = [
            "durable_core_raw",
            "recent_bloc_base",
            "critical_support_raw",
            "bloc_vote_volatility",
            "layer_effective_elections",
            "direct_party_core_raw",
            "candidate_ballot_core_raw",
            "direct_party_recent_base",
            "candidate_ballot_recent_base",
            "direct_party_effective_elections",
            "candidate_ballot_effective_elections",
            "direct_party_reliability",
            "candidate_personal_vote_raw",
            "candidate_conversion_gap_raw",
        ]
        for column in numeric:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
        ineligible_core = ~merged["major_party_core_eligible"].fillna(False).astype(bool)
        # Stable support outside the two major-party lineages remains part of
        # the competitive electorate, but it is not protected as concrete.
        merged.loc[ineligible_core, "critical_support_raw"] = (
            merged.loc[ineligible_core, "critical_support_raw"]
            + merged.loc[ineligible_core, "durable_core_raw"]
        ).clip(upper=merged.loc[ineligible_core, "recent_bloc_base"])
        for column in (
            "durable_core_raw",
            "direct_party_core_raw",
            "candidate_ballot_core_raw",
        ):
            merged.loc[ineligible_core, column] = 0.0
        anchored = merged["durable_core_raw"] + merged["critical_support_raw"]
        anchored_sum = anchored.groupby([merged["election_id"], merged["region_id"]]).transform("sum")
        scale = np.minimum(1.0, 0.90 / anchored_sum.replace(0.0, np.nan)).fillna(1.0)
        merged["core_voting_mass"] = merged["durable_core_raw"] * scale
        merged["critical_voting_mass"] = merged["critical_support_raw"] * scale
        group_anchor = (merged["core_voting_mass"] + merged["critical_voting_mass"]).groupby(
            [merged["election_id"], merged["region_id"]]
        ).transform("sum")
        merged["swing_voting_mass"] = (1.0 - group_anchor).clip(0.0, 1.0)
        merged = _candidate_camp_frame(merged)
        merged = _attach_candidate_regional_accent(merged)
        merged["camp_core_voting_mass"] = 0.0
        merged["camp_critical_voting_mass"] = 0.0
        merged["camp_core_total"] = 0.0
        merged["camp_critical_total"] = 0.0
        for camp in (
            CAMP_CONSERVATIVE,
            CAMP_LIBERAL,
            CAMP_PROGRESSIVE,
            CAMP_CENTRIST,
        ):
            source_mask = merged["candidate_source_camp"].eq(camp)
            source_core = merged["core_voting_mass"].where(source_mask, 0.0)
            source_critical = merged["critical_voting_mass"].where(source_mask, 0.0)
            group_keys = [merged["election_id"], merged["region_id"]]
            camp_core = source_core.groupby(group_keys).transform("max")
            camp_critical = source_critical.groupby(group_keys).transform("max")
            claim = merged[f"candidate_claim_{camp}"]
            eligible_claim = claim.where(
                merged["major_party_core_eligible"].fillna(False).astype(bool),
                0.0,
            )
            eligible_total = eligible_claim.groupby(group_keys).transform("sum")
            core_claim = (
                eligible_claim / eligible_total.replace(0.0, np.nan)
            ).fillna(0.0)
            merged["camp_core_voting_mass"] += camp_core * core_claim
            merged["camp_critical_voting_mass"] += camp_critical * claim
            merged["camp_core_total"] += camp_core
            merged["camp_critical_total"] += camp_critical
        ineligible_core = ~merged["major_party_core_eligible"].fillna(False).astype(bool)
        merged.loc[ineligible_core, "camp_critical_voting_mass"] = (
            merged.loc[ineligible_core, "camp_critical_voting_mass"]
            + merged.loc[ineligible_core, "camp_core_voting_mass"]
        )
        merged.loc[ineligible_core, "camp_core_voting_mass"] = 0.0
        camp_anchor_sum = (
            merged["camp_core_voting_mass"] + merged["camp_critical_voting_mass"]
        ).groupby([merged["election_id"], merged["region_id"]]).transform("sum")
        merged["camp_swing_voting_mass"] = (1.0 - camp_anchor_sum).clip(0.0, 1.0)
        merged["camp_core_regional_mean"] = merged.groupby(
            ["election_id", "slot"]
        )["camp_core_voting_mass"].transform("mean")
        merged["camp_core_regional_lean"] = (
            merged["camp_core_voting_mass"] - merged["camp_core_regional_mean"]
        )
        outputs.append(merged)
    if not outputs:
        return candidate_frame.copy()
    return pd.concat(outputs, ignore_index=True, sort=False)


def compile_issue_class_signals(
    candidate_frame: pd.DataFrame,
    salience: pd.DataFrame,
    candidate_link: pd.DataFrame,
    character_overlay: pd.DataFrame,
    regional_sensitivity: pd.DataFrame | None = None,
    candidate_stance: pd.DataFrame | None = None,
    election_dates: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compile continuous directional and attention signals by issue class.

    Direction comes from the speech-character overlay. Neutral/informational
    sentences contribute to attention, but do not become positive or negative
    votes by fiat.
    """

    keys = ["election_id", "region_id", "slot"]
    base = candidate_frame[keys].drop_duplicates().copy()
    if base.empty:
        return base
    sal = salience.copy()
    link = candidate_link.copy()
    overlay = character_overlay.copy()
    for source in (sal, link):
        source["available_date"] = pd.to_datetime(source.get("available_date"), errors="coerce")
    date_map = dict(election_dates or {})
    if not date_map:
        date_map = {
            election_id: str(election_date(str(election_id)).date())
            for election_id in base["election_id"].unique()
            if election_date(str(election_id)) is not None
        }
    sal["target_date"] = pd.to_datetime(sal["election_id"].map(date_map), errors="coerce")
    link["target_date"] = pd.to_datetime(link["election_id"].map(date_map), errors="coerce")
    sal = sal.loc[
        sal["available_date"].notna()
        & sal["target_date"].notna()
        & (sal["available_date"] < sal["target_date"])
    ].copy()
    link = link.loc[
        link["available_date"].notna()
        & link["target_date"].notna()
        & (link["available_date"] < link["target_date"])
    ].copy()
    if sal.empty or link.empty or overlay.empty:
        for issue_class in ISSUE_CLASSES:
            base[f"issue_pref_{issue_class}"] = 0.0
            base[f"issue_attention_{issue_class}"] = 0.0
        base["issue_preference_strength"] = 0.0
        base["issue_attention_strength"] = 0.0
        return base

    sal["salience_score"] = pd.to_numeric(sal["salience_score"], errors="coerce").fillna(0.0)
    age = (sal["target_date"] - sal["available_date"]).dt.days.clip(lower=0)
    sal["temporal_weight"] = 0.20 + 0.80 * np.exp(-np.log(2.0) * age / 45.0)
    sal["weighted_salience"] = sal["salience_score"] * sal["temporal_weight"]
    issue_salience = sal.groupby(["election_id", "issue_name"], as_index=False).agg(
        weighted_salience=("weighted_salience", "sum"),
        temporal_weight=("temporal_weight", "sum"),
    )
    issue_salience["salience"] = issue_salience["weighted_salience"] / issue_salience[
        "temporal_weight"
    ].replace(0.0, np.nan)

    link["emphasis_within"] = pd.to_numeric(link["emphasis_within"], errors="coerce").fillna(0.0)
    link = link.groupby(["election_id", "slot", "issue_name"], as_index=False)[
        "emphasis_within"
    ].mean()
    overlay_columns = [
        "election_id",
        "slot",
        "issue_name",
        "character_score",
        "character_intensity",
        "informational_score",
        "directional_balance",
        "issue_confidence_quality",
    ]
    for column in overlay_columns:
        if column not in overlay.columns:
            overlay[column] = 0.0
    overlay = overlay[overlay_columns].drop_duplicates(["election_id", "slot", "issue_name"])
    for column in overlay_columns[3:]:
        overlay[column] = pd.to_numeric(overlay[column], errors="coerce").fillna(0.0)

    issue_rows = link.merge(
        issue_salience[["election_id", "issue_name", "salience"]],
        on=["election_id", "issue_name"],
        how="left",
    ).merge(overlay, on=["election_id", "slot", "issue_name"], how="left")
    issue_rows["issue_class"] = issue_rows["issue_name"].map(ISSUE_CLASS_BY_NAME).fillna("social")
    for column in [
        "salience",
        "character_score",
        "character_intensity",
        "informational_score",
        "issue_confidence_quality",
    ]:
        issue_rows[column] = pd.to_numeric(issue_rows[column], errors="coerce").fillna(0.0)

    expanded = base.merge(issue_rows, on=["election_id", "slot"], how="left")
    if candidate_stance is not None and not candidate_stance.empty:
        stance = candidate_stance.copy()
        stance["available_date"] = pd.to_datetime(stance.get("available_date"), errors="coerce")
        stance["target_date"] = pd.to_datetime(stance["election_id"].map(date_map), errors="coerce")
        stance = stance.loc[
            stance["available_date"].notna()
            & stance["target_date"].notna()
            & (stance["available_date"] < stance["target_date"])
        ].copy()
        stance["candidate_stance_direction"] = pd.to_numeric(
            stance.get("party_stance_signal_centered", 0.0), errors="coerce"
        ).fillna(0.0)
        stance["candidate_stance_confidence"] = pd.to_numeric(
            stance.get("confidence", 0.0), errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        stance = stance[
            ["election_id", "slot", "candidate_stance_direction", "candidate_stance_confidence"]
        ].drop_duplicates(["election_id", "slot"], keep="last")
        expanded = expanded.merge(stance, on=["election_id", "slot"], how="left")
    else:
        expanded["candidate_stance_direction"] = 0.0
        expanded["candidate_stance_confidence"] = 0.0
    expanded["candidate_stance_direction"] = pd.to_numeric(
        expanded["candidate_stance_direction"], errors="coerce"
    ).fillna(0.0).clip(-1.0, 1.0)
    expanded["candidate_stance_confidence"] = pd.to_numeric(
        expanded["candidate_stance_confidence"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    if regional_sensitivity is not None and not regional_sensitivity.empty:
        sensitivity = regional_sensitivity.copy()
        if "sensitivity_score" in sensitivity.columns and "sensitivity" not in sensitivity.columns:
            sensitivity = sensitivity.rename(columns={"sensitivity_score": "sensitivity"})
        sensitivity = sensitivity[["issue_name", "region_id", "sensitivity"]].copy()
        sensitivity["sensitivity"] = pd.to_numeric(sensitivity["sensitivity"], errors="coerce")
        expanded = expanded.merge(sensitivity, on=["issue_name", "region_id"], how="left")
    else:
        expanded["sensitivity"] = 0.30
    expanded["sensitivity"] = expanded["sensitivity"].fillna(0.30).clip(0.0, 1.0)
    locality = 0.70 + 0.30 * expanded["sensitivity"]
    confidence = expanded["issue_confidence_quality"].clip(0.0, 1.0)
    common = expanded["salience"] * expanded["emphasis_within"] * confidence * locality
    expanded["attention_component"] = common * (
        0.50 * expanded["character_intensity"].clip(0.0, 1.0)
        + 0.50 * expanded["informational_score"].clip(0.0, 1.0)
    )
    # Sentence polarity describes the speech, not who benefits from it. The
    # candidate-facing sign therefore comes from own-party support/defense
    # versus cross-party attack posture. Issue character controls magnitude.
    expanded["preference_component"] = (
        expanded["attention_component"]
        * expanded["candidate_stance_direction"]
        * expanded["candidate_stance_confidence"]
    )
    grouped = expanded.groupby(keys + ["issue_class"], as_index=False).agg(
        preference_component=("preference_component", "sum"),
        attention_component=("attention_component", "sum"),
    )
    pref = grouped.pivot_table(index=keys, columns="issue_class", values="preference_component", fill_value=0.0)
    attention = grouped.pivot_table(index=keys, columns="issue_class", values="attention_component", fill_value=0.0)
    out = base.set_index(keys)
    centered_by_class: dict[str, pd.Series] = {}
    attention_by_class: dict[str, pd.Series] = {}
    for issue_class in ISSUE_CLASSES:
        raw_pref = pref.get(issue_class, pd.Series(0.0, index=pref.index)).reindex(out.index).fillna(0.0)
        raw_attention = attention.get(
            issue_class, pd.Series(0.0, index=attention.index)
        ).reindex(out.index).fillna(0.0)
        region_mean = raw_pref.groupby(level=[0, 1]).transform("mean")
        centered_by_class[issue_class] = raw_pref - region_mean
        attention_by_class[issue_class] = raw_attention
    # Preserve the relative salience of issue classes. Normalizing each class
    # independently would turn every tiny issue into a unit shock and allow up
    # to eight unit shocks to accumulate in one candidate utility.
    preference_total = sum((value.abs() for value in centered_by_class.values()), start=pd.Series(0.0, index=out.index))
    attention_total = sum(attention_by_class.values(), start=pd.Series(0.0, index=out.index))
    attention_scale = attention_total.groupby(level=0).transform("max").replace(0.0, np.nan)
    for issue_class in ISSUE_CLASSES:
        # Use the attention scale, not the maximum preference signal. Dividing
        # by the latter forced every election's strongest candidate contrast
        # to one and erased the magnitude of stance/confidence differences.
        out[f"issue_pref_{issue_class}"] = (
            centered_by_class[issue_class] / attention_scale
        ).fillna(0.0).clip(-1.0, 1.0)
        out[f"issue_attention_{issue_class}"] = (
            attention_by_class[issue_class] / attention_scale
        ).fillna(0.0).clip(0.0, 1.0)
    out["issue_preference_strength"] = sum(
        (out[f"issue_pref_{issue_class}"].abs() for issue_class in ISSUE_CLASSES),
        start=pd.Series(0.0, index=out.index),
    )
    out["issue_attention_strength"] = sum(
        (out[f"issue_attention_{issue_class}"] for issue_class in ISSUE_CLASSES),
        start=pd.Series(0.0, index=out.index),
    )
    return out.reset_index()


def _normalize_group(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-9, None)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(len(values), 1.0 / max(len(values), 1))
    return values / total


def _preference_layer_scales(
    pref_signal: np.ndarray,
    layer_separation: float,
    layer_response_profile: str,
) -> tuple[float, np.ndarray, float]:
    """Return core, critical, and swing response scales.

    The profile separates the three theory-driven mechanisms for ablation.
    ``combined`` preserves the original one-parameter experiment.
    """

    separation = float(np.clip(layer_separation, 0.0, 1.0))
    profiles = {
        "combined",
        "core_rigidity",
        "critical_defection",
        "swing_mobility",
        "critical_swing",
    }
    if layer_response_profile not in profiles:
        raise ValueError(f"unknown electorate layer response profile: {layer_response_profile}")
    core_active = layer_response_profile in {"combined", "core_rigidity"}
    critical_active = layer_response_profile in {
        "combined",
        "critical_defection",
        "critical_swing",
    }
    swing_active = layer_response_profile in {
        "combined",
        "swing_mobility",
        "critical_swing",
    }
    core_scale = 1.0 - 0.50 * separation if core_active else 1.0
    if critical_active:
        positive_scale = (
            1.0 + 0.15 * separation
            if layer_response_profile in {"combined", "critical_swing"}
            else 1.0
        )
        critical_scale = np.where(
            np.asarray(pref_signal, dtype=float) < 0.0,
            1.0 + separation,
            positive_scale,
        )
    else:
        critical_scale = np.ones_like(np.asarray(pref_signal, dtype=float))
    swing_scale = 1.0 + 0.50 * separation if swing_active else 1.0
    return core_scale, critical_scale, swing_scale


def apply_electorate_layer_response(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
    config: ElectorateLayerConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply terrain anchoring and heterogeneous issue elasticities.

    A neutral config is an exact identity after compositional normalization.
    The optional non-voter reservoir is active only when the caller supplies a
    non-negative ``nonvoter_reservoir`` column and a non-zero gain.
    """

    required = {
        "election_id",
        "region_id",
        "slot",
        "core_voting_mass",
        "critical_voting_mass",
        "swing_voting_mass",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"layer response frame missing columns: {sorted(missing)}")
    out = frame.copy().reset_index(drop=True)
    out["baseline_pred"] = np.asarray(pred, dtype=float)
    out["national_baseline_pred"] = out.groupby(
        ["election_id", "slot"]
    )["baseline_pred"].transform("mean")
    camp_anchor_gain = float(np.clip(config.camp_core_anchor_gain, 0.0, 1.0))
    composition_gain = float(np.clip(config.camp_composition_gain, 0.0, 1.0))
    use_camp_masses = (
        (camp_anchor_gain > 0.0 or composition_gain > 0.0)
        and {
            "camp_core_voting_mass",
            "camp_critical_voting_mass",
        }.issubset(out.columns)
    )
    core_column = "camp_core_voting_mass" if use_camp_masses else "core_voting_mass"
    critical_column = (
        "camp_critical_voting_mass" if use_camp_masses else "critical_voting_mass"
    )
    out["national_core_voting_mass"] = out.groupby(
        ["election_id", "slot"]
    )[core_column].transform("mean")
    predictions = np.zeros(len(out), dtype=float)
    diagnostic_rows: list[pd.DataFrame] = []
    for _, group in out.groupby(["election_id", "region_id"], sort=False):
        idx = group.index.to_numpy()
        baseline = _normalize_group(group["baseline_pred"].to_numpy(float))
        core_raw = group[core_column].to_numpy(float)
        critical_raw = group[critical_column].to_numpy(float)
        accent_signal = pd.to_numeric(
            group.get("regional_accent_signal", pd.Series(0.0, index=group.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)
        accent_reliability = pd.to_numeric(
            group.get(
                "regional_accent_reliability",
                pd.Series(0.0, index=group.index),
            ),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(float)
        accent_volatility = pd.to_numeric(
            group.get(
                "regional_accent_volatility",
                pd.Series(0.0, index=group.index),
            ),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(float)
        accent_gain = float(np.clip(config.regional_accent_gain, 0.0, 0.35))
        accent_width = max(float(config.regional_accent_signal_width), 0.01)
        scaled_accent_signal = np.clip(accent_signal / accent_width, -1.0, 1.0)
        noncore_mobility = np.clip(
            1.0 - core_raw / np.maximum(baseline, 1e-6),
            0.0,
            1.0,
        )
        accent_log_shift = (
            accent_gain
            * scaled_accent_signal
            * accent_reliability
            * (1.0 - 0.50 * accent_volatility)
            * noncore_mobility
        )
        accent_anchored_pred = _normalize_group(
            baseline * np.exp(np.clip(accent_log_shift, -0.35, 0.35))
        )
        regional_lean = pd.to_numeric(
            group.get("camp_core_regional_lean", pd.Series(0.0, index=group.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)
        regional_lean_gain = float(np.clip(config.camp_regional_lean_gain, 0.0, 1.0))
        regional_anchored_pred = _normalize_group(
            np.clip(
                accent_anchored_pred + regional_lean_gain * regional_lean,
                1e-9,
                None,
            )
        )
        camp_floor = np.clip(core_raw, 0.0, None)
        floor_sum = float(camp_floor.sum())
        if floor_sum > 0.95:
            camp_floor *= 0.95 / floor_sum
            floor_sum = 0.95
        flexible = np.clip(regional_anchored_pred - camp_floor, 0.0, None)
        remaining = max(1.0 - floor_sum, 0.0)
        if float(flexible.sum()) > 1e-12:
            camp_terrain = camp_floor + remaining * flexible / float(flexible.sum())
        else:
            camp_terrain = camp_floor + remaining * regional_anchored_pred
        camp_terrain = _normalize_group(camp_terrain)
        national_contestable = np.clip(
            group["national_baseline_pred"].to_numpy(float)
            - group["national_core_voting_mass"].to_numpy(float),
            1e-9,
            None,
        )
        national_profile = _normalize_group(national_contestable)
        composition_residual = max(1.0 - floor_sum, 0.0)
        camp_composition = _normalize_group(
            camp_floor + composition_residual * national_profile
        )
        floor_anchored_pred = _normalize_group(
            (1.0 - camp_anchor_gain) * regional_anchored_pred
            + camp_anchor_gain * camp_terrain
        )
        camp_anchored_pred = _normalize_group(
            (1.0 - composition_gain) * floor_anchored_pred
            + composition_gain * camp_composition
        )
        anchored_raw = np.clip(core_raw + critical_raw, 0.0, None)
        residual = max(1.0 - float(anchored_raw.sum()), 0.0)
        terrain = _normalize_group(anchored_raw + residual * camp_anchored_pred)
        anchor_gain = float(np.clip(config.terrain_anchor_gain, 0.0, 1.0))
        anchored_pred = _normalize_group(
            (1.0 - anchor_gain) * camp_anchored_pred + anchor_gain * terrain
        )

        # Fit latent masses inside the anchored prediction so that zero issue
        # response preserves it exactly.
        cap = 0.95 * anchored_pred
        raw_total = core_raw + critical_raw
        scale = np.minimum(1.0, cap / np.maximum(raw_total, 1e-12))
        core = core_raw * scale
        critical = critical_raw * scale
        swing = np.clip(anchored_pred - core - critical, 0.0, None)
        preference_delta = np.zeros(len(group), dtype=float)
        core_preference_delta = np.zeros(len(group), dtype=float)
        critical_preference_delta = np.zeros(len(group), dtype=float)
        swing_preference_delta = np.zeros(len(group), dtype=float)
        turnout_delta = np.zeros(len(group), dtype=float)
        nonvoter_reservoir = pd.to_numeric(
            group.get("nonvoter_reservoir", pd.Series(0.0, index=group.index)), errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        for issue_class in ISSUE_CLASSES:
            pref_signal = pd.to_numeric(
                group.get(f"issue_pref_{issue_class}", 0.0), errors="coerce"
            )
            if not isinstance(pref_signal, pd.Series):
                pref_signal = pd.Series(float(pref_signal), index=group.index)
            pref_signal_array = pref_signal.fillna(0.0).to_numpy(float)
            attention = pd.to_numeric(
                group.get(f"issue_attention_{issue_class}", 0.0), errors="coerce"
            )
            if not isinstance(attention, pd.Series):
                attention = pd.Series(float(attention), index=group.index)
            attention_array = attention.fillna(0.0).to_numpy(float)
            core_scale, critical_scale, swing_scale = _preference_layer_scales(
                pref_signal_array,
                config.layer_separation,
                config.layer_response_profile,
            )
            denominator = np.maximum(anchored_pred, 1e-6)
            gain = float(max(config.preference_gain, 0.0))
            core_preference_delta += (
                gain
                * pref_signal_array
                * core
                * PREFERENCE_SENSITIVITY["core"][issue_class]
                * core_scale
                / denominator
            )
            critical_preference_delta += (
                gain
                * pref_signal_array
                * critical
                * PREFERENCE_SENSITIVITY["critical"][issue_class]
                * critical_scale
                / denominator
            )
            swing_preference_delta += (
                gain
                * pref_signal_array
                * swing
                * PREFERENCE_SENSITIVITY["swing"][issue_class]
                * swing_scale
                / denominator
            )
            turnout_mass = (
                core * TURNOUT_SENSITIVITY["core"][issue_class]
                + critical * TURNOUT_SENSITIVITY["critical"][issue_class]
                + swing * TURNOUT_SENSITIVITY["swing"][issue_class]
            )
            defensive_or_positive = 0.25 + 0.75 * np.clip(pref_signal_array, 0.0, 1.0)
            turnout_delta += (
                float(max(config.turnout_gain, 0.0))
                * attention_array
                * defensive_or_positive
                * turnout_mass
                / np.maximum(anchored_pred, 1e-6)
            )
            turnout_delta += (
                float(max(config.nonvoter_gain, 0.0))
                * attention_array
                * defensive_or_positive
                * nonvoter_reservoir
                * TURNOUT_SENSITIVITY["nonvoter"][issue_class]
            )
        preference_delta = (
            core_preference_delta
            + critical_preference_delta
            + swing_preference_delta
        )
        total_delta = np.clip(preference_delta + turnout_delta, -1.25, 1.25)
        predicted = _normalize_group(anchored_pred * np.exp(total_delta))
        predictions[idx] = predicted
        diagnostics = group[["election_id", "region_id", "slot"]].copy()
        diagnostics["camp_regional_anchored_pred"] = regional_anchored_pred
        diagnostics["regional_accent_signal"] = accent_signal
        diagnostics["regional_accent_signal_scaled"] = scaled_accent_signal
        diagnostics["regional_accent_reliability"] = accent_reliability
        diagnostics["regional_accent_log_shift"] = accent_log_shift
        diagnostics["regional_accent_anchored_pred"] = accent_anchored_pred
        diagnostics["camp_terrain_pred"] = camp_terrain
        diagnostics["camp_floor_anchored_pred"] = floor_anchored_pred
        diagnostics["national_contestable_profile"] = national_profile
        diagnostics["camp_composition_pred"] = camp_composition
        diagnostics["camp_anchored_pred"] = camp_anchored_pred
        diagnostics["terrain_pred"] = terrain
        diagnostics["anchored_pred"] = anchored_pred
        diagnostics["core_voting_mass_effective"] = core
        diagnostics["critical_voting_mass_effective"] = critical
        diagnostics["swing_voting_mass_effective"] = swing
        diagnostics["core_preference_log_shift"] = core_preference_delta
        diagnostics["critical_preference_log_shift"] = critical_preference_delta
        diagnostics["swing_preference_log_shift"] = swing_preference_delta
        diagnostics["preference_log_shift"] = preference_delta
        diagnostics["turnout_log_shift"] = turnout_delta
        diagnostics["layer_pred"] = predicted
        diagnostic_rows.append(diagnostics)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True) if diagnostic_rows else pd.DataFrame()
    return predictions, diagnostics


def apply_electorate_layer_response_draws(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    config: ElectorateLayerConfig,
) -> np.ndarray:
    """Vectorized electorate-layer response for Monte Carlo prediction draws."""

    draws = np.asarray(predictions, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != len(frame):
        raise ValueError("prediction draws must have shape (n_draws, len(frame))")
    out = np.zeros_like(draws)
    working = frame.reset_index(drop=True)
    camp_anchor_gain = float(np.clip(config.camp_core_anchor_gain, 0.0, 1.0))
    composition_gain = float(np.clip(config.camp_composition_gain, 0.0, 1.0))
    use_camp_masses = (
        (camp_anchor_gain > 0.0 or composition_gain > 0.0)
        and {
            "camp_core_voting_mass",
            "camp_critical_voting_mass",
        }.issubset(working.columns)
    )
    core_column = "camp_core_voting_mass" if use_camp_masses else "core_voting_mass"
    critical_column = (
        "camp_critical_voting_mass" if use_camp_masses else "critical_voting_mass"
    )
    national_draws = np.zeros_like(draws)
    national_core = np.zeros(len(working), dtype=float)
    for (_, slot), indices in working.groupby(["election_id", "slot"], sort=False).indices.items():
        slot_idx = np.asarray(indices, dtype=int)
        national_draws[:, slot_idx] = draws[:, slot_idx].mean(axis=1, keepdims=True)
        national_core[slot_idx] = float(
            pd.to_numeric(
                working.iloc[slot_idx][core_column], errors="coerce"
            ).fillna(0.0).mean()
        )
    for indices in working.groupby(["election_id", "region_id"], sort=False).indices.values():
        idx = np.asarray(indices, dtype=int)
        group = working.iloc[idx]
        baseline = np.clip(draws[:, idx], 1e-9, None)
        baseline /= np.maximum(baseline.sum(axis=1, keepdims=True), 1e-12)
        core_raw = pd.to_numeric(group[core_column], errors="coerce").fillna(0.0).to_numpy(float)
        critical_raw = pd.to_numeric(
            group[critical_column], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        accent_signal = pd.to_numeric(
            group.get("regional_accent_signal", pd.Series(0.0, index=group.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)[None, :]
        accent_reliability = pd.to_numeric(
            group.get(
                "regional_accent_reliability",
                pd.Series(0.0, index=group.index),
            ),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(float)[None, :]
        accent_volatility = pd.to_numeric(
            group.get(
                "regional_accent_volatility",
                pd.Series(0.0, index=group.index),
            ),
            errors="coerce",
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(float)[None, :]
        noncore_mobility = np.clip(
            1.0 - core_raw[None, :] / np.maximum(baseline, 1e-6),
            0.0,
            1.0,
        )
        accent_gain = float(np.clip(config.regional_accent_gain, 0.0, 0.35))
        accent_width = max(float(config.regional_accent_signal_width), 0.01)
        scaled_accent_signal = np.clip(
            accent_signal / accent_width,
            -1.0,
            1.0,
        )
        accent_shift = (
            accent_gain
            * scaled_accent_signal
            * accent_reliability
            * (1.0 - 0.50 * accent_volatility)
            * noncore_mobility
        )
        accent_anchored = baseline * np.exp(np.clip(accent_shift, -0.35, 0.35))
        accent_anchored /= np.maximum(
            accent_anchored.sum(axis=1, keepdims=True), 1e-12
        )
        regional_lean = pd.to_numeric(
            group.get("camp_core_regional_lean", pd.Series(0.0, index=group.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)[None, :]
        regional_lean_gain = float(np.clip(config.camp_regional_lean_gain, 0.0, 1.0))
        regional_anchored = np.clip(
            accent_anchored + regional_lean_gain * regional_lean,
            1e-9,
            None,
        )
        regional_anchored /= np.maximum(
            regional_anchored.sum(axis=1, keepdims=True), 1e-12
        )
        camp_floor = np.clip(core_raw, 0.0, None)
        floor_sum = float(camp_floor.sum())
        if floor_sum > 0.95:
            camp_floor *= 0.95 / floor_sum
            floor_sum = 0.95
        flexible = np.clip(regional_anchored - camp_floor[None, :], 0.0, None)
        flexible_sum = flexible.sum(axis=1, keepdims=True)
        remaining = max(1.0 - floor_sum, 0.0)
        normalized_flexible = np.divide(
            flexible,
            np.maximum(flexible_sum, 1e-12),
        )
        fallback = regional_anchored
        allocation = np.where(flexible_sum > 1e-12, normalized_flexible, fallback)
        camp_terrain = camp_floor[None, :] + remaining * allocation
        camp_terrain /= np.maximum(camp_terrain.sum(axis=1, keepdims=True), 1e-12)
        national_profile = np.clip(
            national_draws[:, idx] - national_core[idx][None, :],
            1e-9,
            None,
        )
        national_profile /= np.maximum(
            national_profile.sum(axis=1, keepdims=True), 1e-12
        )
        composition_residual = max(1.0 - floor_sum, 0.0)
        camp_composition = (
            camp_floor[None, :] + composition_residual * national_profile
        )
        camp_composition /= np.maximum(
            camp_composition.sum(axis=1, keepdims=True), 1e-12
        )
        floor_anchored = (
            (1.0 - camp_anchor_gain) * regional_anchored
            + camp_anchor_gain * camp_terrain
        )
        floor_anchored /= np.maximum(
            floor_anchored.sum(axis=1, keepdims=True), 1e-12
        )
        camp_anchored = (
            (1.0 - composition_gain) * floor_anchored
            + composition_gain * camp_composition
        )
        camp_anchored /= np.maximum(camp_anchored.sum(axis=1, keepdims=True), 1e-12)
        anchored_raw = np.clip(core_raw + critical_raw, 0.0, None)
        residual = max(1.0 - float(anchored_raw.sum()), 0.0)
        terrain = anchored_raw[None, :] + residual * camp_anchored
        terrain /= np.maximum(terrain.sum(axis=1, keepdims=True), 1e-12)
        anchor_gain = float(np.clip(config.terrain_anchor_gain, 0.0, 1.0))
        anchored = (1.0 - anchor_gain) * camp_anchored + anchor_gain * terrain
        anchored /= np.maximum(anchored.sum(axis=1, keepdims=True), 1e-12)
        raw_total = core_raw + critical_raw
        scale = np.minimum(
            1.0,
            0.95 * anchored / np.maximum(raw_total[None, :], 1e-12),
        )
        core = core_raw[None, :] * scale
        critical = critical_raw[None, :] * scale
        swing = np.clip(anchored - core - critical, 0.0, None)
        core_preference_delta = np.zeros_like(anchored)
        critical_preference_delta = np.zeros_like(anchored)
        swing_preference_delta = np.zeros_like(anchored)
        turnout_delta = np.zeros_like(anchored)
        nonvoter = pd.to_numeric(
            group.get("nonvoter_reservoir", pd.Series(0.0, index=group.index)), errors="coerce"
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)[None, :]
        for issue_class in ISSUE_CLASSES:
            pref_signal = pd.to_numeric(
                group.get(f"issue_pref_{issue_class}", 0.0), errors="coerce"
            )
            if not isinstance(pref_signal, pd.Series):
                pref_signal = pd.Series(float(pref_signal), index=group.index)
            pref_signal_array = pref_signal.fillna(0.0).to_numpy(float)[None, :]
            attention = pd.to_numeric(
                group.get(f"issue_attention_{issue_class}", 0.0), errors="coerce"
            )
            if not isinstance(attention, pd.Series):
                attention = pd.Series(float(attention), index=group.index)
            attention_array = attention.fillna(0.0).to_numpy(float)[None, :]
            core_scale, critical_scale, swing_scale = _preference_layer_scales(
                pref_signal_array,
                config.layer_separation,
                config.layer_response_profile,
            )
            denominator = np.maximum(anchored, 1e-6)
            gain = float(max(config.preference_gain, 0.0))
            core_preference_delta += (
                gain
                * pref_signal_array
                * core
                * PREFERENCE_SENSITIVITY["core"][issue_class]
                * core_scale
                / denominator
            )
            critical_preference_delta += (
                gain
                * pref_signal_array
                * critical
                * PREFERENCE_SENSITIVITY["critical"][issue_class]
                * critical_scale
                / denominator
            )
            swing_preference_delta += (
                gain
                * pref_signal_array
                * swing
                * PREFERENCE_SENSITIVITY["swing"][issue_class]
                * swing_scale
                / denominator
            )
            turnout_mass = (
                core * TURNOUT_SENSITIVITY["core"][issue_class]
                + critical * TURNOUT_SENSITIVITY["critical"][issue_class]
                + swing * TURNOUT_SENSITIVITY["swing"][issue_class]
            )
            defensive_or_positive = 0.25 + 0.75 * np.clip(pref_signal_array, 0.0, 1.0)
            turnout_delta += (
                float(max(config.turnout_gain, 0.0))
                * attention_array
                * defensive_or_positive
                * turnout_mass
                / np.maximum(anchored, 1e-6)
            )
            turnout_delta += (
                float(max(config.nonvoter_gain, 0.0))
                * attention_array
                * defensive_or_positive
                * nonvoter
                * TURNOUT_SENSITIVITY["nonvoter"][issue_class]
            )
        preference_delta = (
            core_preference_delta
            + critical_preference_delta
            + swing_preference_delta
        )
        delta = np.clip(preference_delta + turnout_delta, -1.25, 1.25)
        predicted = anchored * np.exp(delta)
        predicted /= np.maximum(predicted.sum(axis=1, keepdims=True), 1e-12)
        out[:, idx] = predicted
    return out
