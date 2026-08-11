"""Issue vote engine with history-derived regional bloc priors.

The module keeps the original lightweight OLS workflow, but replaces
hand-coded regional partisanship with a prior estimated from previous
presidential and proportional-election bloc results.
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.shared_schema.election import RegionResolution  # noqa: E402,F401
from news_collector.sources.member_party import party_bloc  # noqa: E402
from presidential_issue_engine.point_in_time import (  # noqa: E402
    filter_available_by_election,
    forecast_cutoff,
)
from presidential_issue_engine.electorate_layers import (  # noqa: E402
    ElectorateLayerConfig,
    apply_electorate_layer_response,
    apply_electorate_layer_response_draws,
    compile_issue_class_signals,
    estimate_electorate_layers,
)
from presidential_issue_engine.election_scope import (  # noqa: E402
    ELECTION_DATES as CENTRAL_ELECTION_DATES,
    ROLLING_WARMUP_ELECTIONS,
    SCORED_ELECTIONS,
    WARMUP_ELECTIONS,
)

try:
    from presidential_issue_engine.region_bloc_prior import (  # noqa: E402
        DISTRICT_TERRAIN_TYPE_WEIGHTS,
        attach_bloc_prior,
        compute_bloc_prior,
        load_bloc_history,
        normalize_bloc,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from region_bloc_prior import (  # type: ignore  # noqa: E402
        DISTRICT_TERRAIN_TYPE_WEIGHTS,
        attach_bloc_prior,
        compute_bloc_prior,
        load_bloc_history,
        normalize_bloc,
    )


RESULTS = "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
SALIENCE = "data/issue_salience_assembly.csv"
LINK = "data/candidate_issue_link.csv"
REGION_ISSUE_SENSITIVITY = "presidential_issue_engine/fixed_dataset/region_issue_sensitivity_curated.csv"
ISSUE_IMPORTANCE = "presidential_issue_engine/fixed_dataset/issue_importance.csv"
STRICT_UNDATED_CURATED_INPUTS_ENV = "POLL_PROJECT_STRICT_UNDATED_CURATED_INPUTS"
BLOC_HISTORY = "presidential_issue_engine/fixed_dataset/bloc_history_results.csv"
COALITION_EVENTS = "presidential_issue_engine/fixed_dataset/coalition_events.csv"
SCORED_CONTEST_SCOPE = "presidential_issue_engine/fixed_dataset/scored_contest_scope.csv"
ECONOMIC_INDICATORS = "presidential_issue_engine/fixed_dataset/economic_indicators.csv"
ECONOMIC_SLOT_ALIGNMENT = "presidential_issue_engine/fixed_dataset/economic_slot_alignment.csv"
HOUSING_PRICE_INDEX_SIDO = "presidential_issue_engine/fixed_dataset/housing_price_index_sido.csv"
HOUSING_PRICE_INDEX_SGG = "presidential_issue_engine/fixed_dataset/housing_price_index_sgg.csv"
HOUSING_SLOT_ALIGNMENT = "presidential_issue_engine/fixed_dataset/housing_slot_alignment.csv"
KOSPI_DAILY = "presidential_issue_engine/fixed_dataset/kospi_daily.csv"
INTEREST_RATE_INDICATORS = "presidential_issue_engine/fixed_dataset/interest_rate_indicators.csv"
ENHANCED_CANDIDATE_ISSUE_PROFILE = "data/raw/candidate_issue_profile.csv"
ENHANCED_MEGA_ISSUE_AXIS = "data/raw/mega_issue_axis.csv"
ENHANCED_MEGA_ISSUE_INTENSITY = "data/raw/mega_issue_intensity.csv"
ENHANCED_MEGA_ISSUE_ATTRIBUTION = "data/raw/mega_issue_attribution.csv"
AUTO_CANDIDATE_ISSUE_PROFILE = "data/raw/auto_issue_seed/candidate_issue_profile.csv"
AUTO_MEGA_ISSUE_AXIS = "data/raw/auto_issue_seed/mega_issue_axis.csv"
AUTO_MEGA_ISSUE_ATTRIBUTION = "data/raw/auto_issue_seed/mega_issue_attribution.csv"
MEGA_ISSUE_TAXONOMY = "data/raw/mega_issue_taxonomy.csv"
ENHANCED_ISSUE_SCOPE_WEIGHTS = "data/raw/issue_scope_weights.csv"
ASSEMBLY_DERIVED_ISSUE_SCOPE_WEIGHTS = "data/raw/issue_scope_weights_assembly_derived.csv"
ISSUE_EPOCH_IMPORTANCE = "data/raw/issue_epoch_importance.csv"
ISSUE_TEMPORAL_CONVERSION = "data/raw/issue_temporal_conversion.csv"
THIRD_CANDIDATE_PROFILE = "data/raw/third_candidate_profile.csv"
THIRD_CANDIDATE_PRESSURE = "data/raw/third_candidate_pressure.csv"
CANDIDATE_REGIONAL_BASE = "data/raw/candidate_regional_base.csv"
WITHDRAWN_CANDIDATE_TRANSFERS = "data/raw/withdrawn_candidate_transfers.csv"
# Optional compiled registry. Active V23 patches this path so coalition and
# withdrawn-vote features are derived from one source. Empty preserves legacy.
WITHDRAWAL_TRANSFER_REGISTRY = ""
CANDIDATE_POLITICAL_LANDSCAPE = "data/raw/candidate_political_landscape.csv"
ASSEMBLY15_CANDIDATE_LEGACY_LANDSCAPE = "data/raw/assembly15_candidate_legacy_landscape.csv"
CANDIDATE_PARTY_SPEECH_CONTEXT = "data/raw/candidate_party_speech_context.csv"
CANDIDATE_PARTY_TONE_GAP = "data/raw/candidate_party_tone_gap.csv"
CANDIDATE_PUBLIC_TREATMENT = "data/raw/candidate_public_treatment.csv"
CANDIDATE_VOTE_CONVERSION_CONTEXT = "data/raw/candidate_vote_conversion_context.csv"
CANDIDATE_NEUTRAL_ISSUE_CONTEXT = "data/raw/assembly_neutral_issue_context.csv"
ASSEMBLY_ISSUE_CHARACTER_OVERLAY = "data/raw/assembly_issue_character_overlay.csv"
CANDIDATE_GENERATION_PROFILE = "data/raw/candidate_generation_profile.csv"
ELECTION_GENERATION_WEIGHTS = "data/raw/election_generation_weights.csv"
ORDER = list(SCORED_ELECTIONS)
WEIGHT_SELECTION_ELECTIONS = tuple(ORDER)
WARMUP_ORDER = list(WARMUP_ELECTIONS)
ROLLING_WARMUP_ORDER = list(ROLLING_WARMUP_ELECTIONS)
REGIONAL_BASE_ORDER = ["pres_1992", "pres_1997", *ORDER]
ELECTION_DATES = dict(CENTRAL_ELECTION_DATES)
PREDICTORS = [
    "slot_A",
    "slot_B",
    "issue_advantage",
    "rif",
    "partisan_prior",
    "slotA_prior",
    "slotB_prior",
    "landscape_bloc_alignment",
    "landscape_centrist",
    "landscape_inferred_prior",
]
DEFAULT_RIDGE_ALPHA = 0.30
RIDGE_ALPHA = DEFAULT_RIDGE_ALPHA
REGION_RESIDUAL_MAX_PRIOR_ELECTIONS = 1
ELECTION_EPOCH_SAMPLE_WEIGHTS = {
    "pres_1997": 0.40,
    "pres_2002": 0.60,
    "pres_2007": 0.80,
    "pres_2012": 0.90,
    "pres_2017": 1.00,
    "pres_2022": 1.00,
}
THROUGH_2022_TUNING_BASELINE = True
THROUGH_2022_REDERIVED_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "config"
    / "through2022_rederived_layers.json"
)
ELECTORATE_LAYER_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "config"
    / "electorate_layers.json"
)
MAJOR_SLOT_ISSUE_WEIGHT = 0.30
MINOR_SLOT_BASE_ISSUE_WEIGHT = 0.20
MINOR_SLOT_HISTORY_BONUS = 0.30
MINOR_SLOT_HISTORY_SHARE_CAP = 0.15
CONCRETE_PRIOR_CAP = 0.18
GENERAL_PRIOR_SHRINK = 1.00
PRIOR_MODERATION_CORE_MULTIPLIER = 0.50
PRIOR_MODERATION_GENERAL_MULTIPLIER = 1.00
PRIOR_MODERATION_MIN_DEVIATION = 0.04
# Party context describes retention inside an existing camp, not broad public
# approval. Core voters therefore move very little while critical supporters
# can return to the contestable pool when elite support is weak or fragmented.
PARTY_CONTEXT_CORE_DEFECTION_CAP = 0.02
PARTY_CONTEXT_CRITICAL_DEFECTION_CAP = 0.15
SAME_PARTY_TONE_ADJUSTMENT_SCALE = -0.04
CROSS_PARTY_POSITIVE_ADJUSTMENT_SCALE = -0.06
CROSS_PARTY_ADVERSE_ADJUSTMENT_SCALE = 0.00
SAME_ORIENTATION_DISPERSION_ADJUSTMENT_SCALE = 0.00
SAME_ORIENTATION_ANCHOR_ADJUSTMENT_SCALE = 0.03
PUBLIC_TREATMENT_SUPPORT_ADJUSTMENT_SCALE = 0.00
PUBLIC_TREATMENT_SERIOUS_ADJUSTMENT_SCALE = 0.08
GENERATION_ALIGNMENT_ADJUSTMENT_SCALE = 0.08
GENERATION_YOUTH_NICHE_PENALTY_SCALE = 0.04
CANDIDATE_CONVERSION_CONTEXT_ADJUSTMENT_SCALE = 0.035
CANDIDATE_REGIONALISM_ADJUSTMENT_SCALE = 0.20
CANDIDATE_REGIONAL_ANCHOR_STRENGTH = 3.50
CANDIDATE_REGIONAL_ANCHOR_CAP = 3.50
THIRD_COMPETITIVENESS_REFERENCE = 0.25
THIRD_COMPETITIVENESS_MULTIPLIER_FLOOR = 0.35
THIRD_COMPETITIVENESS_MULTIPLIER_CAP = 1.25
NEUTRAL_ISSUE_CONTEXT_SCALE = 0.60
PARTY_STANCE_PROXY_ADJUSTMENT_SCALE = 0.02
WARMUP_PRESIDENTIAL_SLOTS = {
    ("pres_1992", "국민의힘"): ("A", "김영삼"),
    ("pres_1992", "더불어민주당"): ("B", "김대중"),
    ("pres_1992", "제3지대"): ("C", "정주영"),
    ("pres_1997", "더불어민주당"): ("A", "김대중"),
    ("pres_1997", "국민의힘"): ("B", "이회창"),
    ("pres_1997", "제3지대"): ("C", "이인제"),
}
ISSUE_TEMPORAL_HALF_LIFE_DAYS = 28.0
ISSUE_TEMPORAL_WEIGHT_FLOOR = 0.35
ISSUE_TEMPORAL_MOMENTUM_DAYS = 28.0
ISSUE_TEMPORAL_MOMENTUM_STRENGTH = 0.20
ISSUE_TEMPORAL_MOMENTUM_CAP = 0.50
ISSUE_RESIDUAL_BASE_HALF_LIFE_DAYS = 35.0
ISSUE_RESIDUAL_BASE_STRENGTH = 0.08
ISSUE_RESIDUAL_MAX_STRENGTH = 0.25
MEGA_SIGNED_ATTRIBUTION_SCALE = 0.04
MEGA_SIGNED_ATTRIBUTION_CAP = 0.05
MEGA_EVENT_TYPE_MULTIPLIERS = {
    "institutional_crisis": 1.15,
    "state_capture_scandal": 1.10,
    "accountability_scandal": 1.05,
    "candidate_scandal": 1.00,
    "coalition_realignment": 0.95,
    "incumbent_assessment": 0.95,
    "distributional_policy": 0.90,
    "political_realignment": 0.90,
}
LANDSCAPE_VECTOR_COLUMNS = [
    "conservative",
    "liberal",
    "progressive",
    "centrist",
    "anti_establishment",
    "reform",
    "regionalist",
]
LANDSCAPE_FEATURE_COLUMNS = [
    "landscape_confidence",
    *[f"landscape_axis_{column}" for column in LANDSCAPE_VECTOR_COLUMNS],
    "landscape_bloc_alignment",
    "landscape_left_right",
    "landscape_centrist",
    "landscape_reform_anti_establishment",
    "landscape_regionalist",
    "landscape_inferred_prior",
    "landscape_legacy_confidence",
    "landscape_legacy_blend",
]


def _load_through2022_rederived_config() -> dict[str, object]:
    """Load layer strengths whose provenance is restricted through 2022."""

    neutral = {
        "ridge_alpha": 0.30,
        "residual_enabled": False,
        "residual_scale": 0.0,
        "residual_shrinkage": 8.0,
        "neutral_context_scale": 0.0,
        "overlay_gain": 0.0,
        "conversion_scale": 0.0,
        "district_terrain_scale": 0.0,
        "within_bloc_transfer_scale": 0.0,
        "within_bloc_reservoir_gain": 1.0,
        "within_bloc_stronghold_gain": 0.0,
        "regionalism_scale": 0.0,
        "regional_anchor_strength": 1.0,
        "third_competitiveness_gate_enabled": False,
        "third_character_multiplier_enabled": False,
        "manual_issue_seed_enabled": False,
        "automatic_issue_seed_enabled": True,
    }
    if not THROUGH_2022_REDERIVED_CONFIG_PATH.exists():
        return neutral
    payload = json.loads(THROUGH_2022_REDERIVED_CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("provenance") != "rederived only from rolling elections through 2022":
        raise RuntimeError("Invalid through-2022 layer-config provenance")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Invalid through-2022 layer config")
    return {**neutral, **config}


THROUGH_2022_REDERIVED_LAYER_CONFIG = _load_through2022_rederived_config()


def _load_electorate_layer_config() -> tuple[bool, ElectorateLayerConfig]:
    if not ELECTORATE_LAYER_CONFIG_PATH.exists():
        return False, ElectorateLayerConfig()
    payload = json.loads(ELECTORATE_LAYER_CONFIG_PATH.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    return bool(payload.get("enabled", False)), ElectorateLayerConfig(
        terrain_anchor_gain=float(config.get("terrain_anchor_gain", 0.0)),
        camp_core_anchor_gain=float(config.get("camp_core_anchor_gain", 0.0)),
        camp_regional_lean_gain=float(config.get("camp_regional_lean_gain", 0.0)),
        camp_composition_gain=float(config.get("camp_composition_gain", 0.0)),
        preference_gain=float(config.get("preference_gain", 0.0)),
        layer_separation=float(config.get("layer_separation", 0.0)),
        layer_response_profile=str(config.get("layer_response_profile", "combined")),
        mass_profile=str(config.get("mass_profile", "legacy")),
        turnout_gain=float(config.get("turnout_gain", 0.0)),
        nonvoter_gain=float(config.get("nonvoter_gain", 0.0)),
    )


ELECTORATE_LAYER_ENABLED, ELECTORATE_LAYER_CONFIG = _load_electorate_layer_config()


def _load_through2022_layer_registry() -> dict[str, object]:
    if not THROUGH_2022_REDERIVED_CONFIG_PATH.exists():
        return {}
    payload = json.loads(THROUGH_2022_REDERIVED_CONFIG_PATH.read_text(encoding="utf-8"))
    registry = payload.get("registered_layers", {})
    return registry if isinstance(registry, dict) else {}


THROUGH_2022_LAYER_REGISTRY = _load_through2022_layer_registry()


def _rederived_float(name: str, default: float = 0.0) -> float:
    value = pd.to_numeric(THROUGH_2022_REDERIVED_LAYER_CONFIG.get(name), errors="coerce")
    return default if pd.isna(value) else float(value)


def _rederived_bool(name: str, default: bool = False) -> bool:
    value = THROUGH_2022_REDERIVED_LAYER_CONFIG.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


RIDGE_ALPHA = _rederived_float("ridge_alpha", RIDGE_ALPHA)
LANDSCAPE_BLOC_AXIS_WEIGHTS = {
    "국민의힘": {
        "conservative": 1.0,
        "liberal": 0.0,
        "progressive": 0.0,
        "centrist": 0.20,
        "anti_establishment": 0.10,
        "reform": 0.10,
        "regionalist": 0.15,
    },
    "더불어민주당": {
        "conservative": 0.0,
        "liberal": 0.85,
        "progressive": 0.45,
        "centrist": 0.25,
        "anti_establishment": 0.25,
        "reform": 0.45,
        "regionalist": 0.10,
    },
    "진보정당계": {
        "conservative": 0.0,
        "liberal": 0.55,
        "progressive": 1.0,
        "centrist": 0.10,
        "anti_establishment": 0.30,
        "reform": 0.55,
        "regionalist": 0.05,
    },
    "제3지대": {
        "conservative": 0.25,
        "liberal": 0.25,
        "progressive": 0.10,
        "centrist": 1.0,
        "anti_establishment": 0.65,
        "reform": 0.75,
        "regionalist": 0.10,
    },
    "무소속": {
        "conservative": 0.30,
        "liberal": 0.20,
        "progressive": 0.10,
        "centrist": 0.55,
        "anti_establishment": 0.45,
        "reform": 0.35,
        "regionalist": 0.45,
    },
}


def _read_csv_if_exists(path: str) -> pd.DataFrame:
    """Read a CSV when available, otherwise return an empty frame."""

    if not str(path).strip():
        return pd.DataFrame()
    csv_path = Path(path)
    if not csv_path.exists() or not csv_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _registered_issue_seed_path(manual_path: str, automatic_path: str) -> str:
    if _rederived_bool("automatic_issue_seed_enabled", True):
        return automatic_path
    if _rederived_bool("manual_issue_seed_enabled", False):
        return manual_path
    return ""


def _read_registered_issue_seed(manual_path: str, automatic_path: str) -> pd.DataFrame:
    path = _registered_issue_seed_path(manual_path, automatic_path)
    return _read_csv_if_exists(path) if path else pd.DataFrame()


def _issue_importance() -> dict[str, float]:
    """Load issue-level importance weights with a neutral fallback."""

    if os.getenv(STRICT_UNDATED_CURATED_INPUTS_ENV, "0") == "1":
        return {}
    frame = _read_csv_if_exists(ISSUE_IMPORTANCE)
    if frame.empty or not {"issue_name", "importance"}.issubset(frame.columns):
        return {}
    frame = frame[["issue_name", "importance"]].copy()
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0.5)
    return dict(zip(frame["issue_name"], frame["importance"]))


def _region_issue_sensitivity() -> pd.DataFrame:
    """Load curated sensitivity unless strict PIT mode requires neutral defaults."""

    if os.getenv(STRICT_UNDATED_CURATED_INPUTS_ENV, "0") == "1":
        return pd.DataFrame(columns=["issue_name", "region_id", "sensitivity"])
    return _read_csv_if_exists(REGION_ISSUE_SENSITIVITY)


def _issue_temporal_salience(
    salience: pd.DataFrame,
    macro_reinforcement: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate issue salience with recency and late-campaign momentum."""

    columns = [
        "election_id",
        "issue_name",
        "salience",
        "salience_temporal_mean",
        "salience_late_momentum",
        "salience_residual_stock",
    ]
    required = {"election_id", "issue_name", "period", "salience_score", "available_date"}
    if salience.empty or not required.issubset(salience.columns):
        return pd.DataFrame(columns=columns)

    frame = filter_available_by_election(
        salience,
        ELECTION_DATES,
        source_name="issue_salience_assembly",
    )[["election_id", "issue_name", "period", "salience_score"]].copy()
    frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
    frame["salience_score"] = pd.to_numeric(frame["salience_score"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["period", "election_id", "issue_name"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    # Availability alone is insufficient when a malformed source row labels a
    # post-election period as available early. Both the observation period and
    # the source availability must be no later than D-1.
    frame["_forecast_cutoff"] = frame["election_id"].map(
        {election_id: forecast_cutoff(election_id, ELECTION_DATES) for election_id in ELECTION_DATES}
    )
    in_period = frame["_forecast_cutoff"].isna() | frame["period"].le(frame["_forecast_cutoff"])
    frame = frame.loc[in_period].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _apply_macro_speech_strength(frame, macro_reinforcement)

    if os.getenv("POLL_PROJECT_DISABLE_ISSUE_TEMPORAL_WEIGHTING", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        out = (
            frame.groupby(["election_id", "issue_name"], as_index=False)["salience_score"]
            .mean()
            .rename(columns={"salience_score": "salience"})
        )
        out["salience_temporal_mean"] = out["salience"]
        out["salience_late_momentum"] = 0.0
        out["salience_residual_stock"] = 0.0
        return out[columns]

    observed_end = frame.groupby("election_id")["period"].transform("max")
    persistence_disabled = os.getenv("POLL_PROJECT_DISABLE_ISSUE_RESIDUAL_PERSISTENCE", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if persistence_disabled:
        frame["_forecast_cutoff"] = observed_end
        frame["effective_end"] = observed_end
        frame["residual_terminal_contribution"] = 0.0
    else:
        frame["effective_end"] = frame["_forecast_cutoff"].fillna(observed_end)
        frame = _apply_issue_residual_persistence(frame)
    age_days = (frame["effective_end"] - frame["period"]).dt.days.clip(lower=0)
    half_life = max(_numeric(os.getenv("POLL_PROJECT_ISSUE_TEMPORAL_HALF_LIFE_DAYS", ISSUE_TEMPORAL_HALF_LIFE_DAYS), ISSUE_TEMPORAL_HALF_LIFE_DAYS), 1.0)
    floor = min(
        max(
            _numeric(
                os.getenv("POLL_PROJECT_ISSUE_TEMPORAL_WEIGHT_FLOOR", ISSUE_TEMPORAL_WEIGHT_FLOOR),
                ISSUE_TEMPORAL_WEIGHT_FLOOR,
            ),
            0.0,
        ),
        1.0,
    )
    recency = np.exp(-age_days.to_numpy(float) / half_life)
    frame["temporal_weight"] = floor + (1.0 - floor) * recency

    grouped = frame.groupby(["election_id", "issue_name"], as_index=False).agg(
        weighted_salience=("salience_score", lambda values: float(np.nan)),
        salience_mean=("salience_score", "mean"),
    )
    weighted = (
        frame.assign(weighted_component=frame["salience_score"] * frame["temporal_weight"])
        .groupby(["election_id", "issue_name"], as_index=False)
        .agg(weighted_component=("weighted_component", "sum"), temporal_weight=("temporal_weight", "sum"))
    )
    grouped = grouped.drop(columns=["weighted_salience"]).merge(
        weighted,
        on=["election_id", "issue_name"],
        how="left",
    )
    grouped["salience_temporal_mean"] = grouped["weighted_component"] / grouped["temporal_weight"].replace(0, pd.NA)
    grouped["salience_temporal_mean"] = grouped["salience_temporal_mean"].fillna(grouped["salience_mean"])
    residual = (
        frame.groupby(["election_id", "issue_name"], as_index=False)["residual_terminal_contribution"]
        .max()
        .rename(columns={"residual_terminal_contribution": "salience_residual_stock"})
    )
    grouped = grouped.merge(residual, on=["election_id", "issue_name"], how="left")
    grouped["salience_residual_stock"] = grouped["salience_residual_stock"].fillna(0.0)
    grouped["salience_temporal_mean"] = grouped["salience_temporal_mean"] + grouped["salience_residual_stock"]

    momentum_days = max(
        _numeric(
            os.getenv("POLL_PROJECT_ISSUE_TEMPORAL_MOMENTUM_DAYS", ISSUE_TEMPORAL_MOMENTUM_DAYS),
            ISSUE_TEMPORAL_MOMENTUM_DAYS,
        ),
        1.0,
    )
    recent = frame.loc[age_days <= momentum_days].copy()
    recent_mean = (
        recent.groupby(["election_id", "issue_name"], as_index=False)["salience_score"]
        .mean()
        .rename(columns={"salience_score": "recent_salience"})
    )
    grouped = grouped.merge(recent_mean, on=["election_id", "issue_name"], how="left")
    grouped["recent_salience"] = grouped["recent_salience"].fillna(grouped["salience_mean"])
    denom = grouped["salience_mean"].abs().replace(0, pd.NA)
    ratio = (grouped["recent_salience"] / denom).fillna(1.0)
    momentum_strength = min(
        max(
            _numeric(
                os.getenv("POLL_PROJECT_ISSUE_TEMPORAL_MOMENTUM_STRENGTH", ISSUE_TEMPORAL_MOMENTUM_STRENGTH),
                ISSUE_TEMPORAL_MOMENTUM_STRENGTH,
            ),
            0.0,
        ),
        1.0,
    )
    momentum_cap = max(
        _numeric(
            os.getenv("POLL_PROJECT_ISSUE_TEMPORAL_MOMENTUM_CAP", ISSUE_TEMPORAL_MOMENTUM_CAP),
            ISSUE_TEMPORAL_MOMENTUM_CAP,
        ),
        0.0,
    )
    grouped["salience_late_momentum"] = ((ratio - 1.0) * momentum_strength).clip(
        -momentum_cap,
        momentum_cap,
    )
    grouped["salience"] = grouped["salience_temporal_mean"] * (1.0 + grouped["salience_late_momentum"])
    grouped["salience"] = grouped["salience"].clip(lower=0.0)
    return grouped[columns]


def _issue_features(
    salience: pd.DataFrame,
    link: pd.DataFrame,
    regions: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build national issue advantage and region-adjusted issue fit."""

    macro_reinforcement = _macro_issue_reinforcement_table()
    natsal = _issue_temporal_salience(salience, macro_reinforcement)
    eligible_link = filter_available_by_election(
        link,
        ELECTION_DATES,
        source_name="candidate_issue_link",
    )
    adv = eligible_link.merge(natsal, on=["election_id", "issue_name"], how="left").fillna({"salience": 0.0})
    adv["emphasis_within"] = pd.to_numeric(adv["emphasis_within"], errors="coerce").fillna(0.0)
    adv["salience"] = pd.to_numeric(adv["salience"], errors="coerce").fillna(0.0)
    adv = _apply_mega_axis_salience_boost(adv)
    adv = _apply_issue_epoch_importance(adv)
    adv = _apply_issue_temporal_conversion(adv)
    adv = _apply_macro_issue_reinforcement(adv, macro_reinforcement)
    adv = _apply_macro_phrase_bonus(adv, macro_reinforcement)
    adv = _attach_mega_signed_attribution(adv)
    adv = _apply_stance_issue_overlay(adv)
    adv["importance"] = adv["issue_name"].map(_issue_importance()).fillna(0.5)
    if _use_enhanced_issue_overlay():
        return _enhanced_issue_features(adv, regions)

    adv["issue_component"] = (
        adv["emphasis_within"] * adv["salience"] * adv["mega_signed_attribution_multiplier"]
    )

    issue_advantage = (
        adv.groupby(["election_id", "slot"], as_index=False)["issue_component"]
        .sum()
        .rename(columns={"issue_component": "issue_advantage"})
    )

    sensitivity = _region_issue_sensitivity()
    if sensitivity.empty:
        sensitivity = pd.DataFrame(columns=["issue_name", "region_id", "sensitivity"])
    if "sensitivity_score" in sensitivity.columns and "sensitivity" not in sensitivity.columns:
        sensitivity = sensitivity.rename(columns={"sensitivity_score": "sensitivity"})
    required = {"issue_name", "region_id", "sensitivity"}
    if not required.issubset(sensitivity.columns):
        sensitivity = pd.DataFrame(columns=["issue_name", "region_id", "sensitivity"])
    sensitivity = sensitivity[["issue_name", "region_id", "sensitivity"]].copy()
    sensitivity["sensitivity"] = pd.to_numeric(sensitivity["sensitivity"], errors="coerce")

    region_frame = pd.DataFrame({"region_id": sorted(regions.dropna().unique())})
    regional = adv.merge(region_frame, how="cross")
    regional = regional.merge(sensitivity, on=["issue_name", "region_id"], how="left")
    regional["sensitivity"] = regional["sensitivity"].fillna(0.3)
    regional["rif_component"] = (
        regional["emphasis_within"]
        * regional["salience"]
        * regional["mega_signed_attribution_multiplier"]
        * regional["importance"]
        * regional["sensitivity"]
    )
    region_issue_fit = (
        regional.groupby(["election_id", "region_id", "slot"], as_index=False)["rif_component"]
        .sum()
        .rename(columns={"rif_component": "rif"})
    )
    return issue_advantage, region_issue_fit


def _load_stance_issue_overlay() -> pd.DataFrame:
    """Load the default PIT-safe issue-character overlay, with an off switch."""
    columns = [
        "election_id",
        "issue_name",
        "slot",
        "salience_multiplier",
        "link_multiplier",
        "available_date",
    ]
    if _rederived_float("overlay_gain") <= 0.0:
        return pd.DataFrame(columns=columns)
    configured = os.getenv("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH", "").strip()
    if configured.lower() in {"0", "off", "false", "disabled"}:
        return pd.DataFrame(columns=columns)
    raw_path = configured or ASSEMBLY_ISSUE_CHARACTER_OVERLAY
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        if not configured:
            return pd.DataFrame(columns=columns)
        raise FileNotFoundError(f"stance issue overlay not found: {path}")
    frame = pd.read_csv(path)
    required = set(columns)
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"stance issue overlay missing columns: {missing}")
    out = filter_available_by_election(
        frame[columns].copy(),
        ELECTION_DATES,
        source_name="stance_issue_overlay",
    )
    out["slot"] = out["slot"].astype(str).str.strip()
    out["salience_multiplier"] = pd.to_numeric(
        out["salience_multiplier"], errors="coerce"
    ).fillna(1.0).clip(0.88, 1.24)
    out["link_multiplier"] = pd.to_numeric(
        out["link_multiplier"], errors="coerce"
    ).fillna(1.0).clip(0.96, 1.04)
    if out.duplicated(["election_id", "issue_name", "slot"]).any():
        raise ValueError("stance issue overlay has duplicate election/issue/slot rows")
    return out


def _apply_stance_issue_overlay(adv: pd.DataFrame) -> pd.DataFrame:
    """Adjust issue variables, never create a new signed candidate direction."""
    overlay = _load_stance_issue_overlay()
    out = adv.copy()
    if overlay.empty:
        out["stance_salience_multiplier"] = 1.0
        out["stance_link_multiplier"] = 1.0
        return out
    out = out.merge(
        overlay.drop(columns="available_date"),
        on=["election_id", "issue_name", "slot"],
        how="left",
        validate="many_to_one",
    )
    out["stance_salience_multiplier"] = out["salience_multiplier"].fillna(1.0)
    out["stance_link_multiplier"] = out["link_multiplier"].fillna(1.0)
    out["salience"] = out["salience"] * out["stance_salience_multiplier"]
    out["emphasis_within"] = out["emphasis_within"] * out["stance_link_multiplier"]
    return out.drop(columns=["salience_multiplier", "link_multiplier"])


def _mega_event_shock_profiles() -> pd.DataFrame:
    """Load classified mega-event shock profiles with D-1 availability checks."""

    columns = [
        "election_id",
        "mega_event",
        "shock_type",
        "event_shock_intensity",
        "polarization",
        "target_specificity",
    ]
    frame = _read_csv_if_exists(MEGA_ISSUE_TAXONOMY)
    required = {
        "election_id",
        "mega_event",
        "shock_type",
        "severity",
        "national_scope",
        "persistence",
        "polarization",
        "target_specificity",
        "available_date",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="mega_issue_taxonomy",
    )
    if out.empty:
        return pd.DataFrame(columns=columns)
    for column in ["severity", "national_scope", "persistence", "polarization", "target_specificity"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    out["shock_type"] = out["shock_type"].astype(str).str.strip().str.lower()
    out["event_type_multiplier"] = out["shock_type"].map(MEGA_EVENT_TYPE_MULTIPLIERS).fillna(1.0)
    structural_shock = (
        0.32 * out["severity"]
        + 0.23 * out["national_scope"]
        + 0.20 * out["persistence"]
        + 0.15 * out["polarization"]
        + 0.10 * out["target_specificity"]
    )
    election_intensity = out["election_id"].astype(str).map(_mega_issue_intensity()).fillna(1.0)
    out["event_shock_intensity"] = (
        election_intensity
        * (0.60 + 0.40 * structural_shock)
        * out["event_type_multiplier"]
        * (0.75 + 0.25 * out["confidence"])
    ).clip(0.0, 2.5)
    return out[columns].drop_duplicates(["election_id", "mega_event"], keep="last")


def _mega_issue_persistence_profiles() -> pd.DataFrame:
    """Map classified mega-event persistence to its active issue axes."""

    columns = ["election_id", "issue_name", "mega_persistence"]
    axis = _read_registered_issue_seed(ENHANCED_MEGA_ISSUE_AXIS, AUTO_MEGA_ISSUE_AXIS)
    taxonomy = _read_csv_if_exists(MEGA_ISSUE_TAXONOMY)
    required_axis = {"election_id", "mega_event", "primary_issue", "secondary_issue", "available_date"}
    required_taxonomy = {"election_id", "mega_event", "persistence", "available_date"}
    if axis.empty or taxonomy.empty or not required_axis.issubset(axis.columns) or not required_taxonomy.issubset(taxonomy.columns):
        return pd.DataFrame(columns=columns)
    axis = filter_available_by_election(axis, ELECTION_DATES, source_name="mega_issue_axis")
    taxonomy = filter_available_by_election(
        taxonomy,
        ELECTION_DATES,
        source_name="mega_issue_taxonomy",
    )
    joined = axis.merge(
        taxonomy[["election_id", "mega_event", "persistence"]],
        on=["election_id", "mega_event"],
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame(columns=columns)
    joined["persistence"] = pd.to_numeric(joined["persistence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    rows: list[dict[str, object]] = []
    for row in joined.itertuples(index=False):
        for issue_name in [getattr(row, "primary_issue", ""), getattr(row, "secondary_issue", "")]:
            issue = str(issue_name).strip()
            if issue:
                rows.append(
                    {
                        "election_id": str(getattr(row, "election_id", "")),
                        "issue_name": issue,
                        "mega_persistence": float(getattr(row, "persistence", 0.0)),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).groupby(["election_id", "issue_name"], as_index=False)["mega_persistence"].max()


def _apply_issue_residual_persistence(frame: pd.DataFrame) -> pd.DataFrame:
    """Carry a decaying issue stock from the final observed mention to D-1."""

    out = frame.copy()
    profiles = _mega_issue_persistence_profiles()
    out = out.merge(profiles, on=["election_id", "issue_name"], how="left")
    out["mega_persistence"] = pd.to_numeric(out["mega_persistence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["residual_half_life_days"] = ISSUE_RESIDUAL_BASE_HALF_LIFE_DAYS * (1.0 + 1.5 * out["mega_persistence"])
    out["residual_strength"] = ISSUE_RESIDUAL_BASE_STRENGTH + (
        ISSUE_RESIDUAL_MAX_STRENGTH - ISSUE_RESIDUAL_BASE_STRENGTH
    ) * out["mega_persistence"]
    out["residual_terminal_contribution"] = 0.0

    for _, indices in out.groupby(["election_id", "issue_name"]).groups.items():
        ordered = out.loc[indices].sort_values("period")
        stock = 0.0
        previous_period: pd.Timestamp | None = None
        for row in ordered.itertuples():
            half_life = max(float(getattr(row, "residual_half_life_days")), 1.0)
            period = pd.Timestamp(getattr(row, "period"))
            if previous_period is not None:
                stock *= float(np.exp(-(period - previous_period).days / half_life))
            stock = max(stock, max(float(getattr(row, "salience_score")), 0.0))
            previous_period = period
        if previous_period is None:
            continue
        last = ordered.iloc[-1]
        cutoff = pd.Timestamp(last["_forecast_cutoff"]) if pd.notna(last["_forecast_cutoff"]) else previous_period
        gap_days = max((cutoff - previous_period).days, 0)
        half_life = max(float(last["residual_half_life_days"]), 1.0)
        decay = float(np.exp(-gap_days / half_life))
        gap_activation = min(gap_days / 14.0, 1.0)
        contribution = stock * decay * float(last["residual_strength"]) * gap_activation
        out.loc[ordered.index, "residual_terminal_contribution"] = contribution
    return out


def _apply_mega_axis_salience_boost(adv: pd.DataFrame) -> pd.DataFrame:
    """Boost salience for configured election-level mega issue axes.

    This keeps the legacy issue engine intact: slot emphasis, issue salience,
    issue importance, and regional sensitivity are still computed the old way.
    Only election-specific mega-axis issues receive a configurable salience
    multiplier before the existing issue components are aggregated.
    """

    if os.getenv("POLL_PROJECT_DISABLE_MEGA_AXIS_BOOST", "").lower() in {"1", "true", "yes", "y"}:
        return adv
    axis = _read_registered_issue_seed(ENHANCED_MEGA_ISSUE_AXIS, AUTO_MEGA_ISSUE_AXIS)
    required = {
        "election_id",
        "primary_issue",
        "secondary_issue",
        "axis_weight",
        "regime_axis_weight",
    }
    if axis.empty or not required.issubset(axis.columns):
        return adv
    axis = filter_available_by_election(
        axis,
        ELECTION_DATES,
        source_name="mega_issue_axis",
    )
    intensity = _mega_issue_intensity()
    profiles = _mega_event_shock_profiles()
    if not profiles.empty:
        axis = axis.merge(
            profiles[["election_id", "mega_event", "event_shock_intensity"]],
            on=["election_id", "mega_event"],
            how="left",
        )

    boost_rows: list[dict[str, object]] = []
    for row in axis.itertuples(index=False):
        election_id = str(getattr(row, "election_id", "")).strip()
        event_intensity = _numeric(getattr(row, "event_shock_intensity", np.nan), np.nan)
        election_intensity = event_intensity if np.isfinite(event_intensity) else intensity.get(election_id, 1.0)
        primary = str(getattr(row, "primary_issue", "")).strip()
        secondary = str(getattr(row, "secondary_issue", "")).strip()
        if election_id and primary:
            boost_rows.append(
                {
                    "election_id": election_id,
                    "issue_name": primary,
                    "mega_axis_multiplier": _dampen_mega_axis_multiplier(
                        _numeric(getattr(row, "axis_weight", 1.0), 1.0),
                        election_intensity,
                    ),
                }
            )
        if election_id and secondary:
            boost_rows.append(
                {
                    "election_id": election_id,
                    "issue_name": secondary,
                    "mega_axis_multiplier": _dampen_mega_axis_multiplier(
                        _numeric(getattr(row, "regime_axis_weight", 1.0), 1.0),
                        election_intensity,
                    ),
                }
            )
    if not boost_rows:
        return adv

    boosts = (
        pd.DataFrame(boost_rows)
        .groupby(["election_id", "issue_name"], as_index=False)["mega_axis_multiplier"]
        .max()
    )
    out = adv.merge(boosts, on=["election_id", "issue_name"], how="left")
    out["mega_axis_multiplier"] = out["mega_axis_multiplier"].fillna(1.0)
    out["salience"] = out["salience"] * out["mega_axis_multiplier"]
    return out.drop(columns=["mega_axis_multiplier"])


def _mega_signed_attribution_effects() -> pd.DataFrame:
    """Compile positive and negative mega-event attribution by candidate slot.

    The active engine accepts only ``candidate_slot`` rows because its legacy
    candidate-link input does not carry party/camp identifiers. Other target
    types remain available to the generic enhanced-issues compiler.
    """

    columns = ["election_id", "slot", "issue_name", "mega_signed_attribution_multiplier"]
    attribution = _read_registered_issue_seed(
        ENHANCED_MEGA_ISSUE_ATTRIBUTION, AUTO_MEGA_ISSUE_ATTRIBUTION
    )
    axis = _read_registered_issue_seed(ENHANCED_MEGA_ISSUE_AXIS, AUTO_MEGA_ISSUE_AXIS)
    required = {"election_id", "mega_event", "issue_name", "target_type", "target", "polarity", "weight", "available_date", "confidence"}
    axis_required = {"election_id", "mega_event", "primary_issue", "secondary_issue", "axis_weight", "regime_axis_weight", "available_date"}
    if attribution.empty or axis.empty or not required.issubset(attribution.columns) or not axis_required.issubset(axis.columns):
        return pd.DataFrame(columns=columns)
    attribution = filter_available_by_election(
        attribution,
        ELECTION_DATES,
        source_name="mega_issue_attribution",
    )
    axis = filter_available_by_election(axis, ELECTION_DATES, source_name="mega_issue_axis")
    attribution = attribution.loc[attribution["target_type"].astype(str).eq("candidate_slot")].copy()
    if attribution.empty or axis.empty:
        return pd.DataFrame(columns=columns)
    joined = attribution.merge(
        axis[["election_id", "mega_event", "primary_issue", "secondary_issue", "axis_weight", "regime_axis_weight"]],
        on=["election_id", "mega_event"],
        how="inner",
    )
    profiles = _mega_event_shock_profiles()
    if not profiles.empty:
        joined = joined.merge(profiles, on=["election_id", "mega_event"], how="left")
    matched_axis_issue = joined["issue_name"].astype(str).eq(joined["primary_issue"].astype(str)) | joined[
        "issue_name"
    ].astype(str).eq(joined["secondary_issue"].astype(str))
    joined = joined.loc[matched_axis_issue].copy()
    if joined.empty:
        return pd.DataFrame(columns=columns)
    intensity = joined["election_id"].astype(str).map(_mega_issue_intensity()).fillna(1.0)
    joined["event_shock_intensity"] = pd.to_numeric(
        joined.get("event_shock_intensity", intensity), errors="coerce"
    ).fillna(intensity)
    for column, default in [("polarization", 0.5), ("target_specificity", 0.5)]:
        if column not in joined.columns:
            joined[column] = default
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(default).clip(0.0, 1.0)
    for column in ["polarity", "weight", "confidence"]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)
    joined["signed_attribution_effect"] = (
        joined["polarity"].clip(-1.0, 1.0)
        * joined["weight"].clip(0.0, 1.0)
        * joined["confidence"].clip(0.0, 1.0)
        * joined["event_shock_intensity"]
        * (0.50 + 0.50 * joined["polarization"])
        * (0.50 + 0.50 * joined["target_specificity"])
    )
    grouped = joined.groupby(["election_id", "target", "issue_name"], as_index=False)["signed_attribution_effect"].sum()
    grouped["mega_signed_attribution_multiplier"] = (
        1.0
        + grouped["signed_attribution_effect"].clip(-1.0, 1.0)
        * MEGA_SIGNED_ATTRIBUTION_SCALE
    ).clip(1.0 - MEGA_SIGNED_ATTRIBUTION_CAP, 1.0 + MEGA_SIGNED_ATTRIBUTION_CAP)
    return grouped.rename(columns={"target": "slot"})[columns]


def _attach_mega_signed_attribution(adv: pd.DataFrame) -> pd.DataFrame:
    """Attach bounded positive/negative mega-event effects to matching slots."""

    effects = _mega_signed_attribution_effects()
    if adv.empty or effects.empty:
        out = adv.copy()
        out["mega_signed_attribution_multiplier"] = 1.0
        return out
    out = adv.merge(effects, on=["election_id", "slot", "issue_name"], how="left")
    out["mega_signed_attribution_multiplier"] = (
        pd.to_numeric(out["mega_signed_attribution_multiplier"], errors="coerce").fillna(1.0)
    )
    return out


def _apply_issue_epoch_importance(adv: pd.DataFrame) -> pd.DataFrame:
    """Apply election-specific issue importance multipliers without future leakage."""

    frame = _read_csv_if_exists(ISSUE_EPOCH_IMPORTANCE)
    required = {"election_id", "issue_name", "importance_multiplier", "available_date"}
    if adv.empty or frame.empty or not required.issubset(frame.columns):
        return adv

    overlay = frame.copy()
    overlay["importance_multiplier"] = pd.to_numeric(
        overlay["importance_multiplier"],
        errors="coerce",
    ).fillna(1.0).clip(lower=0.0)
    if "confidence" not in overlay.columns:
        overlay["confidence"] = 1.0
    overlay["confidence"] = pd.to_numeric(
        overlay["confidence"],
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)
    overlay = filter_available_by_election(
        overlay,
        ELECTION_DATES,
        source_name="issue_epoch_importance",
    )
    if overlay.empty:
        return adv

    scale = min(
        max(
            _numeric(
                os.getenv("POLL_PROJECT_ISSUE_EPOCH_IMPORTANCE_SCALE", "0.25"),
                0.25,
            ),
            0.0,
        ),
        1.0,
    )
    overlay["epoch_importance_multiplier"] = (
        1.0 + (overlay["importance_multiplier"] - 1.0) * overlay["confidence"] * scale
    ).clip(lower=0.0)
    multipliers = (
        overlay.groupby(["election_id", "issue_name"], as_index=False)["epoch_importance_multiplier"]
        .prod()
    )
    out = adv.merge(multipliers, on=["election_id", "issue_name"], how="left")
    out["epoch_importance_multiplier"] = out["epoch_importance_multiplier"].fillna(1.0)
    out["salience"] = out["salience"] * out["epoch_importance_multiplier"]
    return out.drop(columns=["epoch_importance_multiplier"])


def _apply_issue_temporal_conversion(adv: pd.DataFrame) -> pd.DataFrame:
    """Apply speaker-derived issue support-conversion multipliers.

    The input table is generated from already-extracted Assembly speech rows.
    It is intentionally weak by default because it is a conversion prior, not a
    vote label. Future-dated rows are ignored per election.
    """

    if os.getenv("POLL_PROJECT_DISABLE_ISSUE_TEMPORAL_CONVERSION", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return adv
    frame = _read_csv_if_exists(ISSUE_TEMPORAL_CONVERSION)
    required = {"election_id", "issue_name", "conversion_multiplier", "available_date"}
    if adv.empty or frame.empty or not required.issubset(frame.columns):
        return adv

    overlay = frame.copy()
    overlay["conversion_multiplier"] = pd.to_numeric(
        overlay["conversion_multiplier"],
        errors="coerce",
    ).fillna(1.0).clip(lower=0.0)
    if "confidence" not in overlay.columns:
        overlay["confidence"] = 1.0
    overlay["confidence"] = pd.to_numeric(
        overlay["confidence"],
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)
    if "temporal_sensitivity" not in overlay.columns:
        overlay["temporal_sensitivity"] = 1.0
    overlay["temporal_sensitivity"] = pd.to_numeric(
        overlay["temporal_sensitivity"],
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.5)
    overlay = filter_available_by_election(
        overlay,
        ELECTION_DATES,
        source_name="issue_temporal_conversion",
    )
    if overlay.empty:
        return adv

    scale = min(
        max(
            _numeric(
                os.getenv("POLL_PROJECT_ISSUE_TEMPORAL_CONVERSION_SCALE", "0.25"),
                0.25,
            ),
            0.0,
        ),
        1.0,
    )
    overlay["temporal_conversion_multiplier"] = (
        1.0
        + (overlay["conversion_multiplier"] - 1.0)
        * overlay["confidence"]
        * overlay["temporal_sensitivity"]
        * scale
    ).clip(lower=0.0)
    multipliers = (
        overlay.groupby(["election_id", "issue_name"], as_index=False)["temporal_conversion_multiplier"]
        .prod()
    )
    out = adv.merge(multipliers, on=["election_id", "issue_name"], how="left")
    out["temporal_conversion_multiplier"] = out["temporal_conversion_multiplier"].fillna(1.0)
    out["salience"] = out["salience"] * out["temporal_conversion_multiplier"]
    return out.drop(columns=["temporal_conversion_multiplier"])


def _mega_issue_intensity() -> dict[str, float]:
    """Load election-level mega issue intensity multipliers."""

    frame = _read_csv_if_exists(ENHANCED_MEGA_ISSUE_INTENSITY)
    if frame.empty or not {"election_id", "mega_issue_intensity", "available_date"}.issubset(frame.columns):
        return {}
    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="mega_issue_intensity",
    )[["election_id", "mega_issue_intensity"]].copy()
    out["mega_issue_intensity"] = pd.to_numeric(
        out["mega_issue_intensity"],
        errors="coerce",
    ).fillna(1.0).clip(lower=0.0)
    return dict(zip(out["election_id"].astype(str), out["mega_issue_intensity"].astype(float)))


def _dampen_mega_axis_multiplier(multiplier: float, election_intensity: float = 1.0) -> float:
    """Apply a conservative fraction of configured mega-axis salience boosts."""

    strength = _numeric(os.getenv("POLL_PROJECT_MEGA_AXIS_BOOST_STRENGTH", "0.10"), 0.10)
    strength = min(max(strength, 0.0), 1.0)
    return 1.0 + max(0.0, multiplier - 1.0) * strength * max(0.0, election_intensity)


def _use_enhanced_issue_overlay() -> bool:
    """Return whether the actual-election engine should use signed issue overlay."""

    if os.getenv("POLL_PROJECT_LEGACY_ISSUES", "").lower() in {"1", "true", "yes", "y"}:
        return False
    if os.getenv("POLL_PROJECT_ENHANCED_ISSUES", "").lower() not in {"1", "true", "yes", "y"}:
        return False
    required = [
        _registered_issue_seed_path(
            ENHANCED_CANDIDATE_ISSUE_PROFILE, AUTO_CANDIDATE_ISSUE_PROFILE
        ),
        _registered_issue_seed_path(ENHANCED_MEGA_ISSUE_AXIS, AUTO_MEGA_ISSUE_AXIS),
        _registered_issue_seed_path(
            ENHANCED_MEGA_ISSUE_ATTRIBUTION, AUTO_MEGA_ISSUE_ATTRIBUTION
        ),
        ENHANCED_ISSUE_SCOPE_WEIGHTS,
    ]
    return all(path and Path(path).exists() for path in required)


def _enhanced_issue_scale() -> float:
    """Optional global scale for actual-election enhanced issue experiments."""

    return _numeric(os.getenv("POLL_PROJECT_ENHANCED_ISSUE_SCALE", "1.0"), 1.0)


def _enhanced_issue_features(
    adv: pd.DataFrame,
    regions: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build signed national/local issue features from manual direction layers."""

    slots = adv.loc[adv["slot"].astype(str) != "alpha", ["election_id", "slot"]].drop_duplicates()
    manual = _manual_signed_issue_fit(slots)
    frame = adv.loc[adv["slot"].astype(str) != "alpha"].copy()
    frame = frame.merge(manual, on=["election_id", "slot", "issue_name"], how="left")
    frame["manual_signed_fit"] = frame["manual_signed_fit"].fillna(0.0)
    has_manual = frame.groupby(["election_id", "issue_name"])["manual_signed_fit"].transform(
        lambda values: values.abs().sum() > 0
    )
    frame["signed_fit"] = frame["emphasis_within"].where(~has_manual, frame["manual_signed_fit"])
    frame["signed_fit"] = frame["signed_fit"].fillna(0.0)
    frame["signed_fit_centered"] = frame["signed_fit"] - frame.groupby(
        ["election_id", "issue_name"]
    )["signed_fit"].transform("mean")

    scope = _issue_scope_weights()
    frame = frame.merge(scope, on="issue_name", how="left")
    frame["national_weight"] = frame["national_weight"].fillna(0.0)
    frame["local_weight"] = frame["local_weight"].fillna(1.0)
    frame["issue_component"] = (
        frame["signed_fit_centered"]
        * frame["salience"]
        * frame["importance"]
        * frame["national_weight"]
        * _enhanced_issue_scale()
    )
    issue_advantage = (
        frame.groupby(["election_id", "slot"], as_index=False)["issue_component"]
        .sum()
        .rename(columns={"issue_component": "issue_advantage"})
    )

    sensitivity = _region_issue_sensitivity()
    if sensitivity.empty:
        sensitivity = pd.DataFrame(columns=["issue_name", "region_id", "sensitivity"])
    if "sensitivity_score" in sensitivity.columns and "sensitivity" not in sensitivity.columns:
        sensitivity = sensitivity.rename(columns={"sensitivity_score": "sensitivity"})
    required = {"issue_name", "region_id", "sensitivity"}
    if not required.issubset(sensitivity.columns):
        sensitivity = pd.DataFrame(columns=["issue_name", "region_id", "sensitivity"])
    sensitivity = sensitivity[["issue_name", "region_id", "sensitivity"]].copy()
    sensitivity["sensitivity"] = pd.to_numeric(sensitivity["sensitivity"], errors="coerce")

    region_frame = pd.DataFrame({"region_id": sorted(regions.dropna().unique())})
    regional = frame.merge(region_frame, how="cross")
    regional = regional.merge(sensitivity, on=["issue_name", "region_id"], how="left")
    regional["sensitivity"] = regional["sensitivity"].fillna(0.3)
    regional["rif_component"] = (
        regional["signed_fit_centered"]
        * regional["salience"]
        * regional["importance"]
        * regional["local_weight"]
        * regional["sensitivity"]
        * _enhanced_issue_scale()
    )
    region_issue_fit = (
        regional.groupby(["election_id", "region_id", "slot"], as_index=False)["rif_component"]
        .sum()
        .rename(columns={"rif_component": "rif"})
    )
    return issue_advantage, region_issue_fit


def _issue_scope_weights() -> pd.DataFrame:
    """Load issue national/local scope weights."""

    required = {"issue_name", "national_weight", "local_weight"}
    frames: list[pd.DataFrame] = []
    for path, priority in [
        (ASSEMBLY_DERIVED_ISSUE_SCOPE_WEIGHTS, 0),
        (ENHANCED_ISSUE_SCOPE_WEIGHTS, 1),
    ]:
        frame = _read_csv_if_exists(path)
        if frame.empty or not required.issubset(frame.columns):
            continue
        out = frame[["issue_name", "national_weight", "local_weight"]].copy()
        for column in ["national_weight", "local_weight"]:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        out["_priority"] = priority
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["issue_name", "national_weight", "local_weight"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["issue_name", "_priority"]).drop_duplicates("issue_name", keep="last")
    return combined[["issue_name", "national_weight", "local_weight"]]


def _manual_signed_issue_fit(slots: pd.DataFrame) -> pd.DataFrame:
    """Compile manual profile and mega attribution rows into slot issue fit."""

    columns = ["election_id", "slot", "issue_name", "manual_signed_fit"]
    automatic_seed = _rederived_bool("automatic_issue_seed_enabled", True)
    manual_seed = _rederived_bool("manual_issue_seed_enabled", False)
    if not automatic_seed and not manual_seed:
        return pd.DataFrame(columns=columns)
    allowed_elections = set(WEIGHT_SELECTION_ELECTIONS)
    rows: list[dict[str, object]] = []
    profile = _read_registered_issue_seed(
        ENHANCED_CANDIDATE_ISSUE_PROFILE, AUTO_CANDIDATE_ISSUE_PROFILE
    )
    if not profile.empty:
        profile = profile.loc[profile["election_id"].astype(str).isin(allowed_elections)].copy()
        profile = filter_available_by_election(
            profile,
            ELECTION_DATES,
            source_name="candidate_issue_profile",
        )
        for row in profile.itertuples(index=False):
            slot = str(getattr(row, "slot", "")).strip()
            issue_name = str(getattr(row, "issue_name", "")).strip()
            election_id = str(getattr(row, "election_id", "")).strip()
            if not slot or not issue_name or not election_id:
                continue
            strength = _numeric(getattr(row, "association_strength", 0.0), 0.0)
            direction = _numeric(getattr(row, "direction", 0.0), 0.0)
            confidence = _numeric(getattr(row, "confidence", 0.0), 0.0)
            rows.append(
                {
                    "election_id": election_id,
                    "slot": slot,
                    "issue_name": issue_name,
                    "manual_signed_fit": strength * direction * confidence,
                }
            )

    attribution = _read_registered_issue_seed(
        ENHANCED_MEGA_ISSUE_ATTRIBUTION, AUTO_MEGA_ISSUE_ATTRIBUTION
    )
    axis = _read_registered_issue_seed(ENHANCED_MEGA_ISSUE_AXIS, AUTO_MEGA_ISSUE_AXIS)
    if not attribution.empty and not axis.empty:
        attribution = attribution.loc[
            attribution["election_id"].astype(str).isin(allowed_elections)
        ].copy()
        axis = axis.loc[axis["election_id"].astype(str).isin(allowed_elections)].copy()
        attribution = filter_available_by_election(
            attribution,
            ELECTION_DATES,
            source_name="mega_issue_attribution",
        )
        axis = filter_available_by_election(
            axis,
            ELECTION_DATES,
            source_name="mega_issue_axis",
        )
        joined = attribution.merge(
            axis[
                [
                    "election_id",
                    "mega_event",
                    "primary_issue",
                ]
            ],
            on=["election_id", "mega_event"],
            how="left",
        )
        profiles = _mega_event_shock_profiles()
        if not profiles.empty:
            joined = joined.merge(
                profiles[["election_id", "mega_event", "event_shock_intensity"]],
                on=["election_id", "mega_event"],
                how="left",
            )
        fallback_intensity = joined["election_id"].astype(str).map(_mega_issue_intensity()).fillna(1.0)
        joined["event_shock_intensity"] = pd.to_numeric(
            joined.get("event_shock_intensity", fallback_intensity), errors="coerce"
        ).fillna(fallback_intensity)
        joined["direction_shock_factor"] = (
            1.0 + 0.10 * (joined["event_shock_intensity"].clip(0.0, 2.0) - 1.0)
        ).clip(0.90, 1.10)
        for row in joined.itertuples(index=False):
            target_type = str(getattr(row, "target_type", "")).strip()
            if target_type != "candidate_slot":
                continue
            election_id = str(getattr(row, "election_id", "")).strip()
            slot = str(getattr(row, "target", "")).strip()
            if not _slot_exists(slots, election_id, slot):
                continue
            polarity = _numeric(getattr(row, "polarity", 0.0), 0.0)
            weight = _numeric(getattr(row, "weight", 0.0), 0.0)
            confidence = _numeric(getattr(row, "confidence", 0.0), 0.0)
            base = polarity * weight * confidence * _numeric(
                getattr(row, "direction_shock_factor", 1.0),
                1.0,
            )
            issue_name = str(getattr(row, "issue_name", "")).strip() or str(getattr(row, "primary_issue", "")).strip()
            if issue_name:
                rows.append(
                    {
                        "election_id": election_id,
                        "slot": slot,
                        "issue_name": issue_name,
                        "manual_signed_fit": base,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame.groupby(["election_id", "slot", "issue_name"], as_index=False)["manual_signed_fit"].sum()


def _slot_exists(slots: pd.DataFrame, election_id: str, slot: str) -> bool:
    if not election_id or not slot:
        return False
    mask = (slots["election_id"].astype(str) == election_id) & (slots["slot"].astype(str) == slot)
    return bool(mask.any())


def _numeric(value: object, default: float) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return float(numeric)


def _load_coalition_events() -> pd.DataFrame:
    """Load manual coalition/withdrawal transfer events for election-day analysis."""

    registry = _read_csv_if_exists(WITHDRAWAL_TRANSFER_REGISTRY)
    if not registry.empty:
        frame = registry.copy()
        if "use_in_coalition_layer" in frame.columns:
            enabled = (
                frame["use_in_coalition_layer"]
                .astype(str)
                .str.lower()
                .isin(["1", "true", "yes", "y"])
            )
            frame = frame.loc[enabled].copy()
        if "is_official_target" in frame.columns:
            official = (
                frame["is_official_target"]
                .astype(str)
                .str.lower()
                .isin(["1", "true", "yes", "y"])
            )
            frame = frame.loc[official].copy()
        if "coalition_transfer_rate" in frame.columns:
            frame["transfer_rate"] = frame["coalition_transfer_rate"]
        if "coalition_voter_compliance" in frame.columns:
            frame["voter_compliance"] = frame["coalition_voter_compliance"]
    else:
        frame = _read_csv_if_exists(COALITION_EVENTS)
    required = {
        "election_id",
        "available_date",
        "event_type",
        "source_slot",
        "target_slot",
        "transfer_rate",
        "voter_compliance",
        "source_viability_after_event",
        "exclude_source_from_evaluation",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="coalition_events",
    )
    for column in ["transfer_rate", "voter_compliance", "source_viability_after_event"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["exclude_source_from_evaluation"] = (
        out["exclude_source_from_evaluation"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
    )
    return out


def _apply_coalition_events(
    issue_advantage: pd.DataFrame,
    region_issue_fit: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transfer source-slot issue features to target slots for manual coalition events."""

    if events.empty:
        return issue_advantage, region_issue_fit

    adjusted_adv = issue_advantage.copy()
    adjusted_rif = region_issue_fit.copy()
    for event in events.itertuples(index=False):
        election_id = event.election_id
        source_slot = event.source_slot
        target_slot = event.target_slot
        transfer = float(event.transfer_rate) * float(event.voter_compliance)
        residual = float(event.source_viability_after_event)

        source_adv_mask = (adjusted_adv["election_id"] == election_id) & (adjusted_adv["slot"] == source_slot)
        target_adv_mask = (adjusted_adv["election_id"] == election_id) & (adjusted_adv["slot"] == target_slot)
        source_adv = adjusted_adv.loc[source_adv_mask, "issue_advantage"]
        if len(source_adv) and len(adjusted_adv.loc[target_adv_mask]):
            adjusted_adv.loc[target_adv_mask, "issue_advantage"] += float(source_adv.iloc[0]) * transfer
            adjusted_adv.loc[source_adv_mask, "issue_advantage"] *= residual

        source_rif = adjusted_rif.loc[
            (adjusted_rif["election_id"] == election_id) & (adjusted_rif["slot"] == source_slot),
            ["region_id", "rif"],
        ]
        if source_rif.empty:
            continue
        target_index = adjusted_rif.index[
            (adjusted_rif["election_id"] == election_id) & (adjusted_rif["slot"] == target_slot)
        ]
        source_by_region = dict(zip(source_rif["region_id"], source_rif["rif"]))
        for idx in target_index:
            region_id = adjusted_rif.at[idx, "region_id"]
            adjusted_rif.at[idx, "rif"] += float(source_by_region.get(region_id, 0.0)) * transfer
        source_index = adjusted_rif.index[
            (adjusted_rif["election_id"] == election_id) & (adjusted_rif["slot"] == source_slot)
        ]
        adjusted_rif.loc[source_index, "rif"] *= residual

    return adjusted_adv, adjusted_rif


def _excluded_event_slots(events: pd.DataFrame) -> set[tuple[str, str]]:
    """Return source slots that should not be evaluated after known withdrawals."""

    if events.empty:
        return set()
    excluded = events.loc[events["exclude_source_from_evaluation"]].copy()
    return set(zip(excluded["election_id"], excluded["source_slot"]))


def _load_scored_contest_scope_exclusions() -> set[tuple[str, str]]:
    """Return ballot candidates excluded from the modeled contest denominator."""

    frame = _read_csv_if_exists(SCORED_CONTEST_SCOPE)
    required = {
        "election_id",
        "slot",
        "include_in_scored_contest",
        "available_date",
    }
    if frame.empty:
        return set()
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"scored_contest_scope is missing columns: {missing}")
    frame = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="scored_contest_scope",
    )
    included = (
        frame["include_in_scored_contest"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )
    excluded = frame.loc[~included, ["election_id", "slot"]].copy()
    return set(zip(excluded["election_id"].astype(str), excluded["slot"].astype(str)))


def _official_issue_signal_weights(
    results: pd.DataFrame,
    excluded_slots: set[tuple[str, str]],
) -> pd.DataFrame:
    """Estimate non-uniform issue reliability from official records.

    The weight intentionally avoids using the same election's final vote share
    as an input. Major A/B slots receive a moderate issue weight, while minor
    slots are trusted according to their previous official national viability.
    Known withdrawals or inactive slots get zero issue weight.
    """

    required = {"election_id", "slot", "vote_share", "is_active_slot"}
    if results.empty or not required.issubset(results.columns):
        return pd.DataFrame(columns=["election_id", "slot", "issue_signal_weight"])

    frame = results.loc[
        results["slot"] != "alpha",
        ["election_id", "slot", "vote_share", "is_active_slot"],
    ].copy()
    frame["vote_share"] = pd.to_numeric(frame["vote_share"], errors="coerce").fillna(0.0)
    frame["is_active_slot"] = (
        frame["is_active_slot"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
    )

    activity = (
        frame.groupby(["election_id", "slot"], as_index=False)["is_active_slot"]
        .max()
        .rename(columns={"is_active_slot": "current_active"})
    )
    national_share = (
        frame.groupby(["election_id", "slot"], as_index=False)["vote_share"]
        .mean()
        .rename(columns={"vote_share": "official_slot_share"})
    )
    next_election = {election_id: ORDER[index + 1] for index, election_id in enumerate(ORDER[:-1])}
    prior_share = national_share.copy()
    prior_share["election_id"] = prior_share["election_id"].map(next_election)
    prior_share = prior_share.dropna(subset=["election_id"]).rename(
        columns={"official_slot_share": "previous_official_slot_share"}
    )

    weights = activity.merge(
        prior_share[["election_id", "slot", "previous_official_slot_share"]],
        on=["election_id", "slot"],
        how="left",
    )
    weights["previous_official_slot_share"] = weights["previous_official_slot_share"].fillna(0.0)

    def calculate(row: pd.Series) -> float:
        key = (str(row["election_id"]), str(row["slot"]))
        if key in excluded_slots or not bool(row["current_active"]):
            return 0.0
        if row["slot"] in {"A", "B"}:
            return MAJOR_SLOT_ISSUE_WEIGHT
        history_factor = min(
            float(row["previous_official_slot_share"]) / MINOR_SLOT_HISTORY_SHARE_CAP,
            1.0,
        )
        return MINOR_SLOT_BASE_ISSUE_WEIGHT + MINOR_SLOT_HISTORY_BONUS * history_factor

    weights["issue_signal_weight"] = weights.apply(calculate, axis=1)
    return weights[["election_id", "slot", "issue_signal_weight"]]


def _load_economic_indicators() -> pd.DataFrame:
    """Load official macroeconomic indicators with point-in-time dates."""

    frame = _read_csv_if_exists(ECONOMIC_INDICATORS)
    required = {"period", "indicator_name", "value", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = frame.copy()
    out["period"] = pd.to_datetime(out["period"], errors="coerce")
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["period", "available_date", "indicator_name", "value"])


def _load_interest_rate_indicators() -> pd.DataFrame:
    """Load the conservative month-end Bank of Korea policy-rate series."""

    frame = _read_csv_if_exists(INTEREST_RATE_INDICATORS)
    required = {"period", "indicator_name", "value", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = frame.copy()
    out["period"] = pd.to_datetime(out["period"], errors="coerce")
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["period", "available_date", "indicator_name", "value"])


def _load_economic_slot_alignment() -> pd.DataFrame:
    """Load manual slot-level economic responsibility settings."""

    frame = _read_csv_if_exists(ECONOMIC_SLOT_ALIGNMENT)
    required = {"election_id", "slot", "economic_responsibility_score", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = frame.copy()
    out["economic_responsibility_score"] = pd.to_numeric(
        out["economic_responsibility_score"],
        errors="coerce",
    ).fillna(0.0)
    return filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="economic_slot_alignment",
    )


def _load_housing_price_index() -> pd.DataFrame:
    """Load SIDO-level apartment transaction price index diagnostics."""

    frame = _read_csv_if_exists(HOUSING_PRICE_INDEX_SIDO)
    required = {"region_id", "period", "value", "yoy_change_pct", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = frame.copy()
    out["period"] = pd.to_datetime(out["period"], errors="coerce")
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["yoy_change_pct"] = pd.to_numeric(out["yoy_change_pct"], errors="coerce")
    return out.dropna(subset=["region_id", "period", "available_date", "value"])


def _load_housing_price_index_sgg() -> pd.DataFrame:
    """Load SGG apartment-price rows for within-province housing diagnostics."""

    frame = _read_csv_if_exists(HOUSING_PRICE_INDEX_SGG)
    required = {"region_id", "sgg_name", "period", "value", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = frame.copy()
    out["period"] = pd.to_datetime(out["period"], errors="coerce")
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["region_id", "sgg_name", "period", "available_date", "value"])


def _load_housing_slot_alignment() -> pd.DataFrame:
    """Load manual slot-level housing responsibility settings."""

    frame = _read_csv_if_exists(HOUSING_SLOT_ALIGNMENT)
    required = {"election_id", "slot", "housing_responsibility_score", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = frame.copy()
    out["housing_responsibility_score"] = pd.to_numeric(
        out["housing_responsibility_score"],
        errors="coerce",
    ).fillna(0.0)
    return filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="housing_slot_alignment",
    )


def _load_kospi_daily() -> pd.DataFrame:
    """Load daily KOSPI OHLCV rows with same-day close availability."""

    frame = _read_csv_if_exists(KOSPI_DAILY)
    required = {"date", "close", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    for column in ["close", "open", "high", "low", "volume", "daily_change_pct"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "available_date", "close"])
    if (out["available_date"] < out["date"]).any():
        raise ValueError("kospi_daily has availability before its trading date")
    return out.sort_values("date").drop_duplicates("date", keep="last")


def _economic_context_features(
    indicators: pd.DataFrame,
    alignment: pd.DataFrame,
    election_dates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a single economic context effect without using future indicators."""

    if election_dates is None:
        election_dates = ELECTION_DATES
    base_columns = [
        "election_id",
        "slot",
        "economic_context_effect",
        "economic_stress_index",
        "trade_context_effect",
        "trade_stress_index",
        "economic_responsibility_score",
        "real_gdp_growth_yoy",
        "current_account_12m_sum",
    ]
    if indicators.empty or alignment.empty:
        return pd.DataFrame(columns=base_columns)

    context_rows: list[dict[str, object]] = []
    for election_id, election_date_text in election_dates.items():
        cutoff = forecast_cutoff(election_id, election_dates)
        if cutoff is None:
            continue
        eligible = indicators.loc[indicators["available_date"] <= cutoff].copy()
        if eligible.empty:
            continue

        growth = eligible.loc[eligible["indicator_name"] == "real_gdp_growth_yoy"].sort_values("period")
        current = eligible.loc[eligible["indicator_name"] == "current_account_balance"].sort_values("period")

        latest_growth = float(growth["value"].iloc[-1]) if not growth.empty else 0.0
        growth_stress = -_historical_z(latest_growth, growth["value"].to_numpy(float))

        current_12m_sum = 0.0
        current_stress = 0.0
        if not current.empty:
            current_sums = current["value"].rolling(12, min_periods=6).sum().dropna()
            if not current_sums.empty:
                current_12m_sum = float(current_sums.iloc[-1])
                current_stress = -_historical_z(current_12m_sum, current_sums.to_numpy(float))

        context_rows.append(
            {
                "election_id": election_id,
                "economic_stress_index": float(np.clip(growth_stress, -2.0, 2.0) / 2.0),
                "trade_stress_index": float(np.clip(current_stress, -2.0, 2.0) / 2.0),
                "real_gdp_growth_yoy": latest_growth,
                "current_account_12m_sum": current_12m_sum,
            }
        )

    context = pd.DataFrame(context_rows)
    if context.empty:
        return pd.DataFrame(columns=base_columns)

    joined = alignment.merge(context, on="election_id", how="inner")
    joined["economic_context_effect"] = (
        -joined["economic_stress_index"] * joined["economic_responsibility_score"]
    )
    joined["trade_context_effect"] = (
        -joined["trade_stress_index"] * joined["economic_responsibility_score"]
    )
    return joined[base_columns]


def _historical_z(value: float, history: np.ndarray) -> float:
    """Return a z-score using only the historical values available at that point."""

    clean = history[np.isfinite(history)]
    if len(clean) < 3:
        return 0.0
    std = float(clean.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float((value - clean.mean()) / std)


def _interest_rate_context_features(
    rates: pd.DataFrame,
    alignment: pd.DataFrame,
    election_dates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build D-1 policy-rate pressure diagnostics without a direct vote effect.

    The source is intentionally monthly and uses the end of each observed month
    as its availability date. This is conservative for intra-month policy
    decisions, but prevents a later ECOS revision or publication from entering
    an earlier forecast.
    """

    if election_dates is None:
        election_dates = ELECTION_DATES
    columns = [
        "election_id",
        "slot",
        "interest_rate_context_effect",
        "interest_rate_stress_index",
        "bok_base_rate",
        "bok_base_rate_12m_change",
        "interest_rate_latest_period",
    ]
    if rates.empty or alignment.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for election_id in election_dates:
        cutoff = forecast_cutoff(election_id, election_dates)
        if cutoff is None:
            continue
        eligible = rates.loc[
            (rates["available_date"] <= cutoff)
            & rates["indicator_name"].astype(str).eq("bok_base_rate")
        ].sort_values("period")
        if len(eligible) < 3:
            continue
        latest = eligible.iloc[-1]
        latest_rate = float(latest["value"])
        baseline = eligible.loc[eligible["period"] <= cutoff - pd.Timedelta(days=365)]
        rate_change = latest_rate - float(baseline.iloc[-1]["value"]) if not baseline.empty else 0.0
        level_pressure = max(_historical_z(latest_rate, eligible["value"].to_numpy(float)), 0.0)
        change_series = eligible["value"].diff(12).dropna()
        change_pressure = max(_historical_z(rate_change, change_series.to_numpy(float)), 0.0)
        rate_stress = float(
            np.clip(0.65 * np.clip(level_pressure / 2.0, 0.0, 1.0) + 0.35 * np.clip(change_pressure / 2.0, 0.0, 1.0), 0.0, 1.0)
        )
        rows.append(
            {
                "election_id": election_id,
                "interest_rate_stress_index": rate_stress,
                "bok_base_rate": latest_rate,
                "bok_base_rate_12m_change": rate_change,
                "interest_rate_latest_period": pd.Timestamp(latest["period"]).date().isoformat(),
            }
        )

    context = pd.DataFrame(rows)
    if context.empty:
        return pd.DataFrame(columns=columns)
    joined = alignment.merge(context, on="election_id", how="inner")
    joined["interest_rate_context_effect"] = (
        -joined["interest_rate_stress_index"] * joined["economic_responsibility_score"]
    )
    return joined[columns]


def _kospi_context_features(
    kospi: pd.DataFrame,
    alignment: pd.DataFrame,
    election_dates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build pre-election market stress and responsibility interactions."""

    if election_dates is None:
        election_dates = ELECTION_DATES
    columns = [
        "election_id",
        "slot",
        "kospi_context_effect",
        "kospi_market_stress_index",
        "kospi_close",
        "kospi_return_3m",
        "kospi_return_12m",
        "kospi_drawdown_12m",
        "kospi_volatility_3m",
        "kospi_latest_date",
    ]
    if kospi.empty or alignment.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for election_id in election_dates:
        cutoff = forecast_cutoff(election_id, election_dates)
        if cutoff is None:
            continue
        eligible = kospi.loc[kospi["available_date"] <= cutoff].sort_values("date").copy()
        if len(eligible) < 20:
            continue
        latest = eligible.iloc[-1]
        latest_close = float(latest["close"])

        def return_since(days: int) -> float:
            baseline = eligible.loc[eligible["date"] <= cutoff - pd.Timedelta(days=days)]
            if baseline.empty:
                return 0.0
            baseline_close = float(baseline.iloc[-1]["close"])
            return latest_close / baseline_close - 1.0 if baseline_close > 0.0 else 0.0

        return_3m = return_since(90)
        return_12m = return_since(365)
        trailing_year = eligible.loc[eligible["date"] >= cutoff - pd.Timedelta(days=365)]
        trailing_high = float(trailing_year["close"].max()) if not trailing_year.empty else latest_close
        drawdown_12m = latest_close / trailing_high - 1.0 if trailing_high > 0.0 else 0.0

        log_returns = np.log(eligible["close"] / eligible["close"].shift(1)).replace(
            [np.inf, -np.inf], np.nan
        )
        rolling_volatility = log_returns.rolling(63, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        clean_volatility = rolling_volatility.dropna()
        volatility_3m = float(clean_volatility.iloc[-1]) if not clean_volatility.empty else 0.0
        volatility_stress = _historical_z(volatility_3m, clean_volatility.to_numpy(float))

        momentum = float(
            np.clip(
                0.65 * np.clip(return_12m / 0.30, -1.0, 1.0)
                + 0.35 * np.clip(return_3m / 0.15, -1.0, 1.0),
                -1.0,
                1.0,
            )
        )
        drawdown_stress = float(np.clip(-drawdown_12m / 0.30, 0.0, 1.0))
        market_stress = float(
            np.clip(
                -0.55 * momentum
                + 0.25 * drawdown_stress
                + 0.20 * np.clip(volatility_stress / 2.0, -1.0, 1.0),
                -1.0,
                1.0,
            )
        )
        rows.append(
            {
                "election_id": election_id,
                "kospi_market_stress_index": market_stress,
                "kospi_close": latest_close,
                "kospi_return_3m": return_3m,
                "kospi_return_12m": return_12m,
                "kospi_drawdown_12m": drawdown_12m,
                "kospi_volatility_3m": volatility_3m,
                "kospi_latest_date": pd.Timestamp(latest["date"]).date().isoformat(),
            }
        )
    context = pd.DataFrame(rows)
    if context.empty:
        return pd.DataFrame(columns=columns)
    joined = alignment.merge(context, on="election_id", how="inner")
    joined["kospi_context_effect"] = (
        -joined["kospi_market_stress_index"] * joined["economic_responsibility_score"]
    )
    return joined[columns]


def _housing_context_features(
    housing: pd.DataFrame,
    election_dates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Attach latest available housing-price index by election and region."""

    if election_dates is None:
        election_dates = ELECTION_DATES
    columns = [
        "election_id",
        "region_id",
        "housing_price_index",
        "housing_price_yoy_change_pct",
        "housing_price_period",
    ]
    if housing.empty:
        return pd.DataFrame(columns=columns)

    rows: list[pd.DataFrame] = []
    for election_id, election_date_text in election_dates.items():
        cutoff = forecast_cutoff(election_id, election_dates)
        if cutoff is None:
            continue
        eligible = housing.loc[housing["available_date"] <= cutoff].copy()
        if eligible.empty:
            continue
        latest = eligible.sort_values(["region_id", "period"]).groupby("region_id", as_index=False).tail(1)
        latest["election_id"] = election_id
        latest["housing_price_index"] = latest["value"]
        latest["housing_price_yoy_change_pct"] = latest["yoy_change_pct"]
        latest["housing_price_period"] = latest["period"].dt.date.astype(str)
        rows.append(latest[columns])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def _housing_pressure_features(
    housing: pd.DataFrame,
    alignment: pd.DataFrame,
    election_dates: dict[str, str] | None = None,
    election_order: list[str] | None = None,
    housing_sgg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build region-specific housing pressure effects for the next election.

    The pressure is the cumulative housing-price index change from the previous
    presidential election's latest available quarter to the current election's
    latest available quarter. It uses only rows available by the election date.
    """

    if election_dates is None:
        election_dates = ELECTION_DATES
    if election_order is None:
        election_order = ORDER
    columns = [
        "election_id",
        "region_id",
        "slot",
        "housing_pressure_effect",
        "housing_cumulative_change_pct",
        "housing_responsibility_score",
        "housing_baseline_period",
        "housing_current_period",
        "housing_sgg_median_change_pct",
        "housing_sgg_dispersion",
        "housing_sgg_positive_share",
        "housing_sgg_count",
        "housing_pressure_intensity",
    ]
    if housing.empty or alignment.empty:
        return pd.DataFrame(columns=columns)

    snapshots: dict[str, pd.DataFrame] = {}
    for election_id in election_order:
        election_date_text = election_dates.get(election_id)
        if election_date_text is None:
            continue
        cutoff = forecast_cutoff(election_id, election_dates)
        if cutoff is None:
            continue
        eligible = housing.loc[housing["available_date"] <= cutoff].copy()
        if eligible.empty:
            continue
        latest = eligible.sort_values(["region_id", "period"]).groupby("region_id", as_index=False).tail(1)
        latest = latest[["region_id", "period", "value"]].rename(
            columns={"period": "snapshot_period", "value": "snapshot_value"}
        )
        snapshots[election_id] = latest

    rows: list[pd.DataFrame] = []
    for index, election_id in enumerate(election_order[1:], start=1):
        previous_id = election_order[index - 1]
        if election_id not in snapshots or previous_id not in snapshots:
            continue
        current = snapshots[election_id].rename(
            columns={"snapshot_period": "housing_current_period", "snapshot_value": "housing_current_value"}
        )
        baseline = snapshots[previous_id].rename(
            columns={"snapshot_period": "housing_baseline_period", "snapshot_value": "housing_baseline_value"}
        )
        pressure = current.merge(baseline, on="region_id", how="inner")
        if pressure.empty:
            continue
        pressure["housing_cumulative_change_pct"] = (
            pressure["housing_current_value"] / pressure["housing_baseline_value"] - 1.0
        ) * 100.0
        pressure["election_id"] = election_id
        rows.append(pressure)

    if not rows:
        return pd.DataFrame(columns=columns)
    pressure_frame = pd.concat(rows, ignore_index=True)
    sgg_summary = _housing_sgg_pressure_summary(housing_sgg, election_dates, election_order)
    if not sgg_summary.empty:
        pressure_frame = pressure_frame.merge(sgg_summary, on=["election_id", "region_id"], how="left")
    for column in [
        "housing_sgg_median_change_pct",
        "housing_sgg_dispersion",
        "housing_sgg_positive_share",
        "housing_sgg_count",
    ]:
        if column not in pressure_frame.columns:
            pressure_frame[column] = 0.0
        pressure_frame[column] = pd.to_numeric(pressure_frame[column], errors="coerce").fillna(0.0)
    joined = pressure_frame.merge(alignment, on="election_id", how="inner")
    joined["housing_pressure_index"] = (joined["housing_cumulative_change_pct"] / 100.0).clip(-1.0, 1.0)
    joined["housing_pressure_intensity"] = (
        0.60 * joined["housing_cumulative_change_pct"].abs().div(100.0)
        + 0.40 * joined["housing_sgg_median_change_pct"].abs().div(100.0)
        + 0.25 * joined["housing_sgg_dispersion"]
    ).clip(0.0, 1.0)
    joined["housing_pressure_effect"] = (
        -joined["housing_pressure_index"] * joined["housing_responsibility_score"]
    )
    joined["housing_baseline_period"] = joined["housing_baseline_period"].dt.date.astype(str)
    joined["housing_current_period"] = joined["housing_current_period"].dt.date.astype(str)
    return joined[columns]


def _housing_sgg_pressure_summary(
    housing_sgg: pd.DataFrame | None,
    election_dates: dict[str, str],
    election_order: list[str],
) -> pd.DataFrame:
    """Summarize within-province price pressure using only D-1 SGG snapshots."""

    columns = [
        "election_id",
        "region_id",
        "housing_sgg_median_change_pct",
        "housing_sgg_dispersion",
        "housing_sgg_positive_share",
        "housing_sgg_count",
    ]
    if housing_sgg is None or housing_sgg.empty:
        return pd.DataFrame(columns=columns)

    snapshots: dict[str, pd.DataFrame] = {}
    for election_id in election_order:
        cutoff = forecast_cutoff(election_id, election_dates)
        if cutoff is None:
            continue
        eligible = housing_sgg.loc[housing_sgg["available_date"] <= cutoff]
        if eligible.empty:
            continue
        snapshots[election_id] = (
            eligible.sort_values(["region_id", "sgg_name", "period"])
            .groupby(["region_id", "sgg_name"], as_index=False)
            .tail(1)[["region_id", "sgg_name", "value"]]
            .rename(columns={"value": "snapshot_value"})
        )

    rows: list[pd.DataFrame] = []
    for index, election_id in enumerate(election_order[1:], start=1):
        previous_id = election_order[index - 1]
        if election_id not in snapshots or previous_id not in snapshots:
            continue
        current = snapshots[election_id].rename(columns={"snapshot_value": "current_value"})
        baseline = snapshots[previous_id].rename(columns={"snapshot_value": "baseline_value"})
        merged = current.merge(baseline, on=["region_id", "sgg_name"], how="inner")
        if merged.empty:
            continue
        merged["sgg_change_pct"] = (merged["current_value"] / merged["baseline_value"] - 1.0) * 100.0
        summary = merged.groupby("region_id", as_index=False)["sgg_change_pct"].agg(
            housing_sgg_median_change_pct="median",
            housing_sgg_dispersion="std",
            housing_sgg_count="count",
        )
        summary["housing_sgg_dispersion"] = summary["housing_sgg_dispersion"].fillna(0.0).div(100.0)
        positive = (
            merged.assign(_positive=merged["sgg_change_pct"].gt(0.0).astype(float))
            .groupby("region_id", as_index=False)["_positive"]
            .mean()
            .rename(columns={"_positive": "housing_sgg_positive_share"})
        )
        summary = summary.merge(positive, on="region_id", how="left")
        summary["election_id"] = election_id
        rows.append(summary[columns])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def _apply_epoch_macro_context_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive D-1 domain and within-economy weights for diagnostics only.

    Trade is deliberately a component of the economic domain, alongside GDP,
    KOSPI, and the policy rate. The calculated weights use no vote outcomes.
    """

    out = frame.copy()

    def epoch_abs_median(column: str, scale: float = 1.0) -> pd.Series:
        source = out[column] if column in out.columns else pd.Series(0.0, index=out.index)
        values = pd.to_numeric(source, errors="coerce").fillna(0.0)
        intensity = (values.abs() / scale).clip(0.0, 1.0)
        return intensity.groupby(out["election_id"]).transform("median")

    component_intensities = pd.DataFrame(
        {
            "growth": epoch_abs_median("economic_stress_index"),
            "trade": epoch_abs_median("trade_stress_index"),
            "kospi": epoch_abs_median("kospi_market_stress_index"),
            "interest_rate": epoch_abs_median("interest_rate_stress_index"),
        },
        index=out.index,
    )
    component_raw = 0.15 + 0.85 * component_intensities
    component_weights = component_raw.div(component_raw.sum(axis=1), axis=0).fillna(0.0)
    for component in component_weights:
        out[f"{component}_within_economy_weight"] = component_weights[component]

    out["economic_domain_effect"] = (
        out["growth_within_economy_weight"] * out["economic_context_effect"]
        + out["trade_within_economy_weight"] * out["trade_context_effect"]
        + out["kospi_within_economy_weight"] * out["kospi_context_effect"]
        + out["interest_rate_within_economy_weight"] * out["interest_rate_context_effect"]
    )
    housing_intensity = epoch_abs_median("housing_pressure_intensity")
    housing_available = out.get(
        "housing_current_period",
        pd.Series("not_available", index=out.index),
    ).astype(str).ne("not_available")
    housing_available = housing_available.groupby(out["election_id"]).transform("max").astype(float)
    economy_intensity = (component_intensities * component_weights).sum(axis=1)
    economy_raw = 0.40 + 0.60 * economy_intensity
    housing_raw = (0.30 + 0.70 * housing_intensity) * housing_available
    total = (economy_raw + housing_raw).replace(0.0, np.nan)
    out["economy_epoch_weight"] = (economy_raw / total).fillna(0.0)
    out["housing_epoch_weight"] = (housing_raw / total).fillna(0.0)
    return out


MACRO_ISSUE_DOMAIN_LOADINGS = {
    "economy_growth": (1.00, 0.00),
    "inflation_livelihood": (0.80, 0.25),
    "jobs_labor": (0.85, 0.00),
    "housing": (0.30, 1.00),
    "external_shock": (0.70, 0.00),
    "foreign_policy": (0.45, 0.00),
    "regional_dev": (0.25, 0.30),
}
MACRO_ISSUE_REINFORCEMENT_STRENGTH = 0.15
MACRO_SPEECH_STRENGTH_REINFORCEMENT = 0.08
MACRO_PHRASE_BONUS_REINFORCEMENT = 0.05


def _macro_issue_reinforcement_table(
    election_dates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return issue multipliers from D-1 macro intensity and era weights."""

    if election_dates is None:
        election_dates = ELECTION_DATES
    columns = [
        "election_id",
        "issue_name",
        "macro_issue_multiplier",
        "macro_speech_strength_multiplier",
        "macro_phrase_bonus_multiplier",
        "economy_epoch_weight",
        "housing_epoch_weight",
        "growth_within_economy_weight",
        "trade_within_economy_weight",
        "kospi_within_economy_weight",
        "interest_rate_within_economy_weight",
        "economic_issue_activation",
        "housing_issue_activation",
        "interest_rate_stress_index",
    ]
    alignment = _load_economic_slot_alignment()
    economic = _economic_context_features(
        _load_economic_indicators(),
        alignment,
        election_dates,
    )
    market = _kospi_context_features(_load_kospi_daily(), alignment, election_dates)
    rates = _interest_rate_context_features(
        _load_interest_rate_indicators(),
        alignment,
        election_dates,
    )
    housing = _housing_pressure_features(
        _load_housing_price_index(),
        _load_housing_slot_alignment(),
        election_dates,
        [election_id for election_id in ELECTION_DATES if election_id in election_dates],
        _load_housing_price_index_sgg(),
    )

    base = pd.DataFrame({"election_id": list(election_dates)})
    if not economic.empty:
        economic_summary = economic.groupby("election_id", as_index=False).first()[
            ["election_id", "economic_stress_index", "trade_stress_index"]
        ]
        base = base.merge(economic_summary, on="election_id", how="left")
    if not market.empty:
        market_summary = market.groupby("election_id", as_index=False).first()[
            ["election_id", "kospi_market_stress_index"]
        ]
        base = base.merge(market_summary, on="election_id", how="left")
    if not rates.empty:
        rate_summary = rates.groupby("election_id", as_index=False).first()[
            ["election_id", "interest_rate_stress_index"]
        ]
        base = base.merge(rate_summary, on="election_id", how="left")
    if not housing.empty:
        housing_summary = (
            housing.assign(
                housing_intensity=(
                    pd.to_numeric(housing["housing_pressure_intensity"], errors="coerce")
                    .fillna(0.0)
                    .clip(0.0, 1.0)
                )
            )
            .groupby("election_id", as_index=False)["housing_intensity"]
            .median()
        )
        base = base.merge(housing_summary, on="election_id", how="left")

    for column in [
        "economic_stress_index",
        "trade_stress_index",
        "kospi_market_stress_index",
        "interest_rate_stress_index",
        "housing_intensity",
    ]:
        if column not in base.columns:
            base[column] = 0.0
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    housing_available = base["housing_intensity"].gt(0.0).astype(float)
    growth_intensity = base["economic_stress_index"].abs()
    market_intensity = base["kospi_market_stress_index"].abs()
    trade_intensity = base["trade_stress_index"].abs()
    rate_intensity = base["interest_rate_stress_index"].abs()
    housing_intensity = base["housing_intensity"].abs()
    component_intensity = pd.DataFrame(
        {
            "growth": growth_intensity,
            "trade": trade_intensity,
            "kospi": market_intensity,
            "interest_rate": rate_intensity,
        }
    )
    component_raw = 0.15 + 0.85 * component_intensity
    component_weights = component_raw.div(component_raw.sum(axis=1), axis=0).fillna(0.0)
    for component in component_weights:
        base[f"{component}_within_economy_weight"] = component_weights[component]
    base["economic_intensity"] = (component_intensity * component_weights).sum(axis=1)

    economy_raw = 0.40 + 0.60 * base["economic_intensity"]
    housing_raw = (0.30 + 0.70 * housing_intensity) * housing_available
    total = (economy_raw + housing_raw).replace(0.0, np.nan)
    base["economy_epoch_weight"] = (economy_raw / total).fillna(0.0)
    base["housing_epoch_weight"] = (housing_raw / total).fillna(0.0)
    base["economic_issue_activation"] = base["economy_epoch_weight"] * base["economic_intensity"]
    base["housing_issue_activation"] = base["housing_epoch_weight"] * housing_intensity

    rows: list[dict[str, object]] = []
    for context in base.itertuples(index=False):
        for issue_name, loadings in MACRO_ISSUE_DOMAIN_LOADINGS.items():
            relevance = (
                loadings[0] * float(context.economic_issue_activation)
                + loadings[1] * float(context.housing_issue_activation)
            )
            rows.append(
                {
                    "election_id": context.election_id,
                    "issue_name": issue_name,
                    "macro_issue_multiplier": 1.0
                    + MACRO_ISSUE_REINFORCEMENT_STRENGTH * min(max(relevance, 0.0), 1.0),
                    "macro_speech_strength_multiplier": 1.0
                    + MACRO_SPEECH_STRENGTH_REINFORCEMENT * min(max(relevance, 0.0), 1.0),
                    "macro_phrase_bonus_multiplier": 1.0
                    + MACRO_PHRASE_BONUS_REINFORCEMENT * min(max(relevance, 0.0), 1.0),
                    "economy_epoch_weight": context.economy_epoch_weight,
                    "housing_epoch_weight": context.housing_epoch_weight,
                    "growth_within_economy_weight": context.growth_within_economy_weight,
                    "trade_within_economy_weight": context.trade_within_economy_weight,
                    "kospi_within_economy_weight": context.kospi_within_economy_weight,
                    "interest_rate_within_economy_weight": context.interest_rate_within_economy_weight,
                    "economic_issue_activation": context.economic_issue_activation,
                    "housing_issue_activation": context.housing_issue_activation,
                    "interest_rate_stress_index": context.interest_rate_stress_index,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _apply_macro_speech_strength(
    salience: pd.DataFrame,
    reinforcement: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply a bounded D-1 multiplier to existing Assembly speech strength."""

    if salience.empty or os.getenv("POLL_PROJECT_DISABLE_MACRO_ISSUE_REINFORCEMENT", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return salience
    table = _macro_issue_reinforcement_table() if reinforcement is None else reinforcement
    if table.empty or "macro_speech_strength_multiplier" not in table.columns:
        return salience
    out = salience.merge(
        table[["election_id", "issue_name", "macro_speech_strength_multiplier"]],
        on=["election_id", "issue_name"],
        how="left",
    )
    out["macro_speech_strength_multiplier"] = out["macro_speech_strength_multiplier"].fillna(1.0)
    out["salience_score"] = out["salience_score"] * out["macro_speech_strength_multiplier"]
    return out.drop(columns="macro_speech_strength_multiplier")


def _apply_macro_issue_reinforcement(
    adv: pd.DataFrame,
    reinforcement: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reinforce existing issue salience without directly shifting vote share."""

    if adv.empty or os.getenv("POLL_PROJECT_DISABLE_MACRO_ISSUE_REINFORCEMENT", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return adv
    table = _macro_issue_reinforcement_table() if reinforcement is None else reinforcement
    if table.empty:
        return adv
    out = adv.merge(
        table[["election_id", "issue_name", "macro_issue_multiplier"]],
        on=["election_id", "issue_name"],
        how="left",
    )
    out["macro_issue_multiplier"] = out["macro_issue_multiplier"].fillna(1.0)
    out["salience"] = out["salience"] * out["macro_issue_multiplier"]
    return out


def _apply_macro_phrase_bonus(
    adv: pd.DataFrame,
    reinforcement: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a small D-1 bonus to pre-existing candidate issue-phrase emphasis."""

    if adv.empty or os.getenv("POLL_PROJECT_DISABLE_MACRO_ISSUE_REINFORCEMENT", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return adv
    table = _macro_issue_reinforcement_table() if reinforcement is None else reinforcement
    if table.empty or "macro_phrase_bonus_multiplier" not in table.columns:
        return adv
    out = adv.merge(
        table[["election_id", "issue_name", "macro_phrase_bonus_multiplier"]],
        on=["election_id", "issue_name"],
        how="left",
    )
    out["macro_phrase_bonus_multiplier"] = out["macro_phrase_bonus_multiplier"].fillna(1.0)
    out["emphasis_within"] = out["emphasis_within"] * out["macro_phrase_bonus_multiplier"]
    return out


def _load_third_candidate_profile() -> pd.DataFrame:
    """Load manual third-candidate profile priors without using outcomes."""

    frame = _read_csv_if_exists(THIRD_CANDIDATE_PROFILE)
    required = {
        "election_id",
        "slot",
        "viability",
        "centrist_appeal",
        "anti_major_party_appeal",
        "regional_base_overlap",
        "available_date",
        "confidence",
    }
    columns = [
        "election_id",
        "slot",
        "third_candidate_name",
        "third_viability",
        "third_centrist_appeal",
        "third_anti_major_party_appeal",
        "third_regional_base_overlap",
        "third_profile_confidence",
    ]
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="third_candidate_profile",
    )
    for column in [
        "viability",
        "centrist_appeal",
        "anti_major_party_appeal",
        "regional_base_overlap",
        "confidence",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "candidate_name" not in out.columns:
        out["candidate_name"] = ""
    out = out.rename(
        columns={
            "candidate_name": "third_candidate_name",
            "viability": "third_viability",
            "centrist_appeal": "third_centrist_appeal",
            "anti_major_party_appeal": "third_anti_major_party_appeal",
            "regional_base_overlap": "third_regional_base_overlap",
            "confidence": "third_profile_confidence",
        }
    )
    return out[columns]


def _load_third_candidate_pressure() -> pd.DataFrame:
    """Load manual source-slot pressure from a viable third candidate."""

    frame = _read_csv_if_exists(THIRD_CANDIDATE_PRESSURE)
    required = {
        "election_id",
        "slot",
        "source_slot",
        "transfer_pressure",
        "available_date",
        "confidence",
    }
    columns = [
        "election_id",
        "slot",
        "source_slot",
        "transfer_pressure",
        "pressure_confidence",
    ]
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="third_candidate_pressure",
    )
    for column in ["transfer_pressure", "confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out = out.rename(columns={"confidence": "pressure_confidence"})
    return out[columns]


def _load_candidate_regional_base() -> pd.DataFrame:
    """Load pre-election candidate-specific regional base facts."""

    frame = _read_csv_if_exists(CANDIDATE_REGIONAL_BASE)
    required = {
        "election_id",
        "region_id",
        "slot",
        "regional_affinity",
        "organization_depth",
        "available_date",
        "confidence",
    }
    columns = [
        "election_id",
        "region_id",
        "slot",
        "candidate_regional_affinity",
        "candidate_regional_organization",
        "candidate_regional_confidence",
        "candidate_regional_base_raw",
    ]
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="candidate_regional_base",
    )
    for column in ["regional_affinity", "organization_depth", "confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["candidate_regional_base_raw"] = (
        out["regional_affinity"] * out["organization_depth"] * out["confidence"]
    )
    out = out.rename(
        columns={
            "regional_affinity": "candidate_regional_affinity",
            "organization_depth": "candidate_regional_organization",
            "confidence": "candidate_regional_confidence",
        }
    )
    return (
        out[columns]
        .groupby(["election_id", "region_id", "slot"], as_index=False)
        .mean(numeric_only=True)
    )


def _candidate_regional_base_features(base: pd.DataFrame) -> pd.DataFrame:
    """Expand sparse candidate bases to every election-region-slot row."""

    keys = ["election_id", "region_id", "slot"]
    columns = [
        *keys,
        "candidate_regional_affinity",
        "candidate_regional_organization",
        "candidate_regional_confidence",
        "candidate_regional_base_raw",
    ]
    if base.empty:
        return pd.DataFrame(columns=columns)
    out = base[keys].copy()
    regional = _load_candidate_regional_base()
    if regional.empty:
        for column in columns[len(keys) :]:
            out[column] = 0.0
        return out[columns]
    out = out.merge(regional, on=keys, how="left")
    for column in columns[len(keys) :]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out[columns]


def _third_candidate_structure_features(base: pd.DataFrame) -> pd.DataFrame:
    """Expand third-candidate priors to election-region-slot observations.

    The learned predictor keeps the expected direction in the feature itself:
    the profiled third slot receives a positive value, while major slots that
    the third candidate draws from receive negative pressure values.
    """

    columns = [
        "election_id",
        "region_id",
        "slot",
        "third_candidate_structure",
        "third_attention_score",
        "third_conversion_capacity",
        "third_attention_overhang",
        "third_viability",
        "third_centrist_appeal",
        "third_anti_major_party_appeal",
        "third_regional_base_overlap",
        "third_profile_confidence",
        "slotA_third_pressure",
        "slotB_third_pressure",
    ]
    if base.empty:
        return pd.DataFrame(columns=columns)

    out = base[["election_id", "region_id", "slot"]].copy()
    profile = _load_third_candidate_profile()
    if profile.empty:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]

    profiled = out.merge(profile, on=["election_id", "slot"], how="left")
    for column in [
        "third_viability",
        "third_centrist_appeal",
        "third_anti_major_party_appeal",
        "third_regional_base_overlap",
        "third_profile_confidence",
    ]:
        profiled[column] = pd.to_numeric(profiled[column], errors="coerce").fillna(0.0)

    profile_strength = (
        profiled["third_viability"]
        * (
            0.55
            + 0.25 * profiled["third_centrist_appeal"]
            + 0.20 * profiled["third_anti_major_party_appeal"]
        )
        * profiled["third_profile_confidence"]
    )
    conversion_capacity = (
        0.20
        + 0.35 * profiled["third_viability"]
        + 0.25 * profiled["third_regional_base_overlap"]
        + 0.20 * profiled["third_centrist_appeal"]
    ).clip(0.0, 1.0)
    profiled["third_attention_score"] = profile_strength
    profiled["third_conversion_capacity"] = np.where(
        profiled["slot"].astype(str).eq("C"),
        conversion_capacity,
        0.0,
    )
    profiled["third_attention_overhang"] = np.where(
        profiled["slot"].astype(str).eq("C"),
        profiled["third_attention_score"] * (1.0 - profiled["third_conversion_capacity"]),
        0.0,
    )
    profiled["third_candidate_structure"] = profile_strength * np.where(
        profiled["slot"].astype(str).eq("C"),
        profiled["third_conversion_capacity"],
        1.0,
    )
    profiled["slotA_third_pressure"] = 0.0
    profiled["slotB_third_pressure"] = 0.0

    pressure = _load_third_candidate_pressure()
    if not pressure.empty:
        excluded_slots = _excluded_event_slots(_load_coalition_events())
        if excluded_slots:
            keep = [
                (str(row.election_id), str(row.slot)) not in excluded_slots
                for row in pressure.itertuples(index=False)
            ]
            pressure = pressure.loc[keep].copy()
        pressure = pressure.merge(
            profile[["election_id", "slot", "third_viability"]],
            on=["election_id", "slot"],
            how="left",
        )
        pressure["third_viability"] = pressure["third_viability"].fillna(0.0)
        pressure["pressure_effect"] = (
            pressure["transfer_pressure"]
            * pressure["pressure_confidence"]
            * pressure["third_viability"]
        )
        pressure_wide = (
            pressure.pivot_table(
                index=["election_id", "source_slot"],
                columns="slot",
                values="pressure_effect",
                aggfunc="sum",
                fill_value=0.0,
            )
            .sum(axis=1)
            .rename("third_pressure")
            .reset_index()
            .rename(columns={"source_slot": "slot"})
        )
        profiled = profiled.merge(pressure_wide, on=["election_id", "slot"], how="left")
        profiled["third_pressure"] = profiled["third_pressure"].fillna(0.0)
        profiled["third_candidate_structure"] -= profiled["third_pressure"]
        profiled["slotA_third_pressure"] = np.where(
            profiled["slot"] == "A",
            profiled["third_pressure"],
            0.0,
        )
        profiled["slotB_third_pressure"] = np.where(
            profiled["slot"] == "B",
            profiled["third_pressure"],
            0.0,
        )

    return profiled[columns]


def _load_withdrawn_candidate_transfers() -> pd.DataFrame:
    """Load viable withdrawn-candidate transfers that are not final ballot slots."""

    registry = _read_csv_if_exists(WITHDRAWAL_TRANSFER_REGISTRY)
    if not registry.empty:
        frame = registry.copy()
        if "use_in_withdrawn_feature_layer" in frame.columns:
            enabled = (
                frame["use_in_withdrawn_feature_layer"]
                .astype(str)
                .str.lower()
                .isin(["1", "true", "yes", "y"])
            )
            frame = frame.loc[enabled].copy()
        if "withdrawn_transfer_rate" in frame.columns:
            frame["transfer_rate"] = frame["withdrawn_transfer_rate"]
        if "withdrawn_voter_compliance" in frame.columns:
            frame["voter_compliance"] = frame["withdrawn_voter_compliance"]
        if "withdrawn_confidence" in frame.columns:
            frame["confidence"] = frame["withdrawn_confidence"]
    else:
        frame = _read_csv_if_exists(WITHDRAWN_CANDIDATE_TRANSFERS)
    required = {
        "election_id",
        "candidate_name",
        "target_slot",
        "viability",
        "transfer_rate",
        "voter_compliance",
        "available_date",
        "confidence",
    }
    columns = [
        "election_id",
        "candidate_name",
        "slot",
        "withdrawn_candidate_transfer",
        "withdrawn_candidate_viability",
        "withdrawn_transfer_rate",
        "withdrawn_voter_compliance",
        "withdrawn_transfer_confidence",
        "withdrawn_landscape_affinity",
    ]
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="withdrawn_candidate_transfers",
    )
    for column in ["viability", "transfer_rate", "voter_compliance", "confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out = _apply_withdrawn_landscape_affinity(out)
    out["withdrawn_candidate_transfer"] = (
        out["viability"]
        * out["transfer_rate"]
        * out["voter_compliance"]
        * out["confidence"]
        * out["withdrawn_landscape_affinity"]
    )
    out = out.rename(
        columns={
            "target_slot": "slot",
            "viability": "withdrawn_candidate_viability",
            "transfer_rate": "withdrawn_transfer_rate",
            "voter_compliance": "withdrawn_voter_compliance",
            "confidence": "withdrawn_transfer_confidence",
        }
    )
    grouped = (
        out.groupby(["election_id", "slot"], as_index=False)
        .agg(
            candidate_name=("candidate_name", lambda values: "; ".join(sorted(set(map(str, values))))),
            withdrawn_candidate_transfer=("withdrawn_candidate_transfer", "sum"),
            withdrawn_candidate_viability=("withdrawn_candidate_viability", "max"),
            withdrawn_transfer_rate=("withdrawn_transfer_rate", "max"),
            withdrawn_voter_compliance=("withdrawn_voter_compliance", "max"),
            withdrawn_transfer_confidence=("withdrawn_transfer_confidence", "max"),
            withdrawn_landscape_affinity=("withdrawn_landscape_affinity", "max"),
        )
    )
    return grouped[columns]


def _load_candidate_political_landscape() -> pd.DataFrame:
    """Load manual candidate political landscape vectors."""

    required = {
        "election_id",
        "slot",
        "candidate_name",
        "candidate_role",
        "available_date",
        "confidence",
        *LANDSCAPE_VECTOR_COLUMNS,
    }
    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "candidate_role",
        *LANDSCAPE_VECTOR_COLUMNS,
        "confidence",
    ]
    frame = _read_csv_if_exists(CANDIDATE_POLITICAL_LANDSCAPE)
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="candidate_political_landscape",
    )
    for column in [*LANDSCAPE_VECTOR_COLUMNS, "confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out[columns]


def _load_assembly15_candidate_legacy_landscape() -> pd.DataFrame:
    """Load optional 15th-Assembly-derived candidate legacy landscape vectors."""

    required = {
        "election_id",
        "slot",
        "candidate_name",
        "available_date",
        "confidence",
        *LANDSCAPE_VECTOR_COLUMNS,
    }
    columns = [
        "election_id",
        "slot",
        "candidate_name",
        *LANDSCAPE_VECTOR_COLUMNS,
        "confidence",
    ]
    frame = _read_csv_if_exists(ASSEMBLY15_CANDIDATE_LEGACY_LANDSCAPE)
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    out = filter_available_by_election(
        frame,
        ELECTION_DATES,
        source_name="assembly15_candidate_legacy_landscape",
    )
    for column in [*LANDSCAPE_VECTOR_COLUMNS, "confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out[columns]


def _candidate_political_landscape_features(base: pd.DataFrame) -> pd.DataFrame:
    """Attach low-dimensional political landscape features to every final candidate row."""

    base_landscape_columns = [
        column for column in LANDSCAPE_FEATURE_COLUMNS if column != "landscape_inferred_prior"
    ]
    columns = ["election_id", "region_id", "slot", *base_landscape_columns]
    out = base[["election_id", "region_id", "slot", "bloc"]].copy()
    if "candidate_name" in base.columns:
        out["candidate_name"] = base["candidate_name"].astype(str).to_numpy()
    else:
        out["candidate_name"] = ""
    landscape = _load_candidate_political_landscape()
    if landscape.empty:
        for column in base_landscape_columns:
            out[column] = 0.0
        return out[columns]

    final = (
        landscape.loc[landscape["candidate_role"].astype(str).str.lower() == "final"]
        .sort_values("confidence", ascending=False)
        .drop_duplicates(["election_id", "slot"])
    )
    if final.empty:
        for column in base_landscape_columns:
            out[column] = 0.0
        return out[columns]

    final = final[["election_id", "slot", *LANDSCAPE_VECTOR_COLUMNS, "confidence"]].rename(
        columns={"confidence": "landscape_confidence"}
    )
    out = out.merge(final, on=["election_id", "slot"], how="left")
    for column in LANDSCAPE_VECTOR_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["landscape_confidence"] = (
        pd.to_numeric(out["landscape_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )

    legacy = _load_assembly15_candidate_legacy_landscape()
    if not legacy.empty:
        legacy = legacy.rename(
            columns={
                "confidence": "legacy_confidence",
                **{column: f"legacy_{column}" for column in LANDSCAPE_VECTOR_COLUMNS},
            }
        )
        out = out.merge(
            legacy,
            on=["election_id", "slot", "candidate_name"],
            how="left",
        )
        scale = _numeric(os.getenv("POLL_PROJECT_ASSEMBLY15_LEGACY_BLEND_SCALE", "0.15"), 0.15)
        scale = min(max(scale, 0.0), 1.0)
        out["landscape_legacy_confidence"] = (
            pd.to_numeric(out["legacy_confidence"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        out["landscape_legacy_blend"] = (out["landscape_legacy_confidence"] * scale).clip(0.0, 0.65)
        for column in LANDSCAPE_VECTOR_COLUMNS:
            legacy_column = f"legacy_{column}"
            out[legacy_column] = pd.to_numeric(out[legacy_column], errors="coerce").fillna(out[column])
            out[column] = (
                out[column] * (1.0 - out["landscape_legacy_blend"])
                + out[legacy_column] * out["landscape_legacy_blend"]
            ).clip(0.0, 1.0)
    else:
        out["landscape_legacy_confidence"] = 0.0
        out["landscape_legacy_blend"] = 0.0

    for column in LANDSCAPE_VECTOR_COLUMNS:
        out[f"landscape_axis_{column}"] = out[column]

    vector_matrix = out[LANDSCAPE_VECTOR_COLUMNS].to_numpy(float)
    weights = np.array(
        [
            [
                LANDSCAPE_BLOC_AXIS_WEIGHTS.get(str(bloc), {}).get(column, 0.0)
                for column in LANDSCAPE_VECTOR_COLUMNS
            ]
            for bloc in out["bloc"]
        ],
        dtype=float,
    )
    out["_bloc_alignment_raw"] = np.sum(vector_matrix * weights, axis=1) / np.maximum(
        weights.sum(axis=1),
        1.0,
    )
    out["_left_right_raw"] = out["conservative"] - (
        out["liberal"] + out["progressive"]
    ) / 2.0
    out["_centrist_raw"] = out["centrist"]
    out["_reform_anti_establishment_raw"] = (
        out["reform"] + out["anti_establishment"]
    ) / 2.0
    out["_regionalist_raw"] = out["regionalist"]

    for source, target in [
        ("_bloc_alignment_raw", "landscape_bloc_alignment"),
        ("_left_right_raw", "landscape_left_right"),
        ("_centrist_raw", "landscape_centrist"),
        ("_reform_anti_establishment_raw", "landscape_reform_anti_establishment"),
        ("_regionalist_raw", "landscape_regionalist"),
    ]:
        election_mean = out.groupby("election_id")[source].transform("mean")
        out[target] = (out[source] - election_mean) * out["landscape_confidence"]

    return out[columns]


def _landscape_inferred_prior_for_weights(
    base: pd.DataFrame,
    history: pd.DataFrame,
    *,
    value_column: str,
    evidence_column: str,
    election_type_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Project regional bloc history onto each candidate's political landscape."""

    columns = ["election_id", "region_id", "slot", value_column, evidence_column]
    out = base[["election_id", "region_id", "slot"]].copy()
    required_axis_columns = {f"landscape_axis_{column}" for column in LANDSCAPE_VECTOR_COLUMNS}
    if history.empty or not required_axis_columns.issubset(base.columns):
        out[value_column] = 0.0
        out[evidence_column] = 0.0
        return out[columns]

    pieces: list[pd.DataFrame] = []
    axis_columns = [f"landscape_axis_{column}" for column in LANDSCAPE_VECTOR_COLUMNS]
    for election_id in base["election_id"].drop_duplicates():
        prior = compute_bloc_prior(
            history,
            str(election_id),
            ORDER,
            election_type_weights=election_type_weights,
        )
        if prior.empty:
            continue
        prior = _split_partisan_prior_layers(prior)
        prior = prior[
            ["region_id", "bloc", "partisan_prior", "effective_election_count"]
        ].copy()
        candidates = (
            base.loc[base["election_id"] == election_id, ["election_id", "slot", *axis_columns]]
            .drop_duplicates()
            .copy()
        )
        for candidate in candidates.itertuples(index=False):
            candidate_vector = np.array(
                [float(getattr(candidate, column)) for column in axis_columns],
                dtype=float,
            )
            affinities: list[dict[str, object]] = []
            for bloc in prior["bloc"].drop_duplicates():
                bloc_vector = np.array(
                    [
                        LANDSCAPE_BLOC_AXIS_WEIGHTS.get(str(bloc), {}).get(column, 0.0)
                        for column in LANDSCAPE_VECTOR_COLUMNS
                    ],
                    dtype=float,
                )
                affinities.append(
                    {
                        "bloc": bloc,
                        "landscape_bloc_affinity": _cosine_similarity(
                            candidate_vector,
                            bloc_vector,
                        ),
                    }
                )
            affinity = pd.DataFrame(affinities)
            joined = prior.merge(affinity, on="bloc", how="left")
            joined["landscape_bloc_affinity"] = joined["landscape_bloc_affinity"].fillna(0.0)
            affinity_sum = joined.groupby("region_id")["landscape_bloc_affinity"].transform("sum")
            safe_affinity_sum = affinity_sum.where(affinity_sum > 0, 1.0)
            joined["weighted_prior"] = np.where(
                affinity_sum.to_numpy(float) > 0,
                joined["partisan_prior"].to_numpy(float)
                * joined["landscape_bloc_affinity"].to_numpy(float)
                / safe_affinity_sum.to_numpy(float),
                0.0,
            )
            joined["weighted_evidence"] = np.where(
                affinity_sum.to_numpy(float) > 0,
                joined["effective_election_count"].to_numpy(float)
                * joined["landscape_bloc_affinity"].to_numpy(float)
                / safe_affinity_sum.to_numpy(float),
                0.0,
            )
            piece = (
                joined.groupby("region_id", as_index=False)[
                    ["weighted_prior", "weighted_evidence"]
                ]
                .sum()
                .rename(
                    columns={
                        "weighted_prior": value_column,
                        "weighted_evidence": evidence_column,
                    }
                )
            )
            piece["election_id"] = candidate.election_id
            piece["slot"] = candidate.slot
            pieces.append(piece)

    if not pieces:
        out[value_column] = 0.0
        out[evidence_column] = 0.0
        return out[columns]
    inferred = pd.concat(pieces, ignore_index=True)
    out = out.merge(inferred, on=["election_id", "region_id", "slot"], how="left")
    out[value_column] = out[value_column].fillna(0.0)
    out[evidence_column] = out[evidence_column].fillna(0.0)
    return out[columns]


def _candidate_landscape_inferred_prior_features(
    base: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Infer broad and constituency-only regional priors for each candidate."""

    keys = ["election_id", "region_id", "slot"]
    broad = _landscape_inferred_prior_for_weights(
        base,
        history,
        value_column="landscape_inferred_prior",
        evidence_column="landscape_inferred_prior_evidence",
    )
    district = _landscape_inferred_prior_for_weights(
        base,
        history,
        value_column="landscape_inferred_district_prior",
        evidence_column="landscape_inferred_district_evidence",
        election_type_weights=DISTRICT_TERRAIN_TYPE_WEIGHTS,
    )
    return broad.merge(district, on=keys, how="outer")


def _split_partisan_prior_layers(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate durable regional core support from movable general support."""

    out = frame.copy()
    raw = pd.to_numeric(out.get("partisan_prior", 0.0), errors="coerce").fillna(0.0)
    evidence = pd.to_numeric(out.get("effective_election_count", 0.0), errors="coerce").fillna(0.0)
    evidence_weight = (evidence / 3.0).clip(0.0, 1.0)
    sign = np.sign(raw.to_numpy(float))
    raw_abs = raw.abs().to_numpy(float)
    concrete_cap = CONCRETE_PRIOR_CAP * (0.75 + 0.25 * evidence_weight.to_numpy(float))
    concrete_abs = np.minimum(raw_abs, concrete_cap)
    general_abs = np.maximum(raw_abs - concrete_abs, 0.0)
    concrete = sign * concrete_abs
    general = sign * general_abs * GENERAL_PRIOR_SHRINK
    out["partisan_prior_raw"] = raw
    out["concrete_partisan_prior"] = concrete
    out["general_partisan_prior"] = general
    out["partisan_prior"] = concrete + general
    return out


def _attach_electorate_layer_features(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    salience: pd.DataFrame,
    link: pd.DataFrame,
) -> pd.DataFrame:
    """Attach PIT-safe ecological layer masses and issue-class signals."""

    keys = ["election_id", "region_id", "slot"]
    candidate_columns = [*keys, "bloc"]
    candidate_columns.extend(
        column
        for column in [
            "landscape_axis_conservative",
            "landscape_axis_liberal",
            "landscape_axis_progressive",
            "landscape_axis_centrist",
            "landscape_confidence",
        ]
        if column in frame.columns
    )
    candidates = frame[candidate_columns].copy()
    masses = estimate_electorate_layers(
        candidates,
        history,
        mass_profile=(
            ELECTORATE_LAYER_CONFIG.mass_profile
            if ELECTORATE_LAYER_ENABLED
            else "legacy"
        ),
    )
    signals = compile_issue_class_signals(
        candidates,
        salience,
        link,
        _read_csv_if_exists(ASSEMBLY_ISSUE_CHARACTER_OVERLAY),
        regional_sensitivity=_region_issue_sensitivity(),
        candidate_stance=_read_csv_if_exists(CANDIDATE_PARTY_TONE_GAP),
        election_dates=ELECTION_DATES,
    )
    mass_columns = [
        *keys,
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
        "core_voting_mass",
        "critical_voting_mass",
        "swing_voting_mass",
        "candidate_camp",
        "candidate_source_camp",
        "candidate_camp_origin_weight",
        "candidate_camp_claim",
        "candidate_claim_camp_conservative",
        "candidate_claim_camp_liberal",
        "candidate_claim_camp_progressive",
        "candidate_claim_camp_centrist",
        "camp_core_total",
        "camp_critical_total",
        "camp_core_voting_mass",
        "camp_critical_voting_mass",
        "camp_swing_voting_mass",
        "camp_core_regional_mean",
        "camp_core_regional_lean",
    ]
    out = frame.merge(masses[mass_columns], on=keys, how="left").merge(
        signals, on=keys, how="left"
    )
    numeric = [
        *mass_columns[3:],
        *[
            column
            for column in signals.columns
            if column.startswith("issue_pref_") or column.startswith("issue_attention_")
        ],
        "issue_preference_strength",
        "issue_attention_strength",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    # Official prior regional turnout history is not yet populated. Keeping the
    # reservoir explicit makes the disabled state auditable and future input
    # backward compatible.
    out["nonvoter_reservoir"] = 0.0
    return out


def _candidate_attention_support_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Separate issue attention from estimated support-conversion capacity."""

    out = frame.copy()
    attention = pd.to_numeric(out.get("issue_advantage", 0.0), errors="coerce").fillna(0.0).abs()
    local_attention = pd.to_numeric(out.get("rif", 0.0), errors="coerce").fillna(0.0).abs()
    party_anchor = (
        pd.to_numeric(out.get("partisan_prior", 0.0), errors="coerce").fillna(0.0).abs()
        / 0.25
    ).clip(0.0, 1.0)
    landscape_anchor = (
        pd.to_numeric(out.get("landscape_bloc_alignment", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        / 0.05
    ).clip(0.0, 1.0)
    major_slot_anchor = out.get("slot", "").astype(str).isin(["A", "B"]).astype(float)
    third_anchor = (
        pd.to_numeric(out.get("third_viability", 0.0), errors="coerce").fillna(0.0)
        * (
            0.50
            + 0.30
            * pd.to_numeric(out.get("third_centrist_appeal", 0.0), errors="coerce").fillna(0.0)
            + 0.20
            * pd.to_numeric(out.get("third_regional_base_overlap", 0.0), errors="coerce").fillna(0.0)
        )
        * pd.to_numeric(out.get("third_profile_confidence", 0.0), errors="coerce").fillna(0.0)
    ).clip(0.0, 1.0)
    support_conversion = (
        0.35 * major_slot_anchor
        + 0.30 * party_anchor
        + 0.20 * landscape_anchor
        + 0.15 * third_anchor
    ).clip(0.0, 1.0)
    out["issue_attention_score"] = attention
    out["issue_local_attention_score"] = local_attention
    out["support_conversion_score"] = support_conversion
    out["issue_support_signal"] = attention * support_conversion
    out["issue_attention_overhang"] = attention * (1.0 - support_conversion)
    return out


def _load_candidate_party_speech_context() -> pd.DataFrame:
    """Load optional party elite speech-context features."""

    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "same_bloc_issue_alignment",
        "same_bloc_frame_convergence",
        "cross_bloc_attack_pressure",
        "intra_bloc_conflict_score",
        "party_elite_support_score",
        "party_elite_fragmentation_score",
        "party_context_support",
        "organization_strength",
        "outsider_status",
        "available_date",
        "confidence",
    ]
    frame = _read_csv_if_exists(CANDIDATE_PARTY_SPEECH_CONTEXT)
    required = {"election_id", "slot", "candidate_name", "party_context_support", "available_date", "confidence"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0 if column not in {"election_id", "slot", "candidate_name", "available_date"} else ""
    out = filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="candidate_party_speech_context",
    )
    for column in columns[3:12] + ["confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    bounded = [
        "same_bloc_issue_alignment",
        "same_bloc_frame_convergence",
        "cross_bloc_attack_pressure",
        "intra_bloc_conflict_score",
        "party_elite_support_score",
        "party_elite_fragmentation_score",
        "organization_strength",
        "outsider_status",
        "confidence",
    ]
    for column in bounded:
        out[column] = out[column].clip(0.0, 1.0)
    out["party_context_support"] = out["party_context_support"].clip(-1.0, 1.0)
    return out[columns]


def _candidate_party_speech_context_features(base: pd.DataFrame) -> pd.DataFrame:
    """Attach party elite support and fragmentation proxies to candidate rows."""

    columns = [
        "election_id",
        "region_id",
        "slot",
        "same_bloc_issue_alignment",
        "same_bloc_frame_convergence",
        "cross_bloc_attack_pressure",
        "intra_bloc_conflict_score",
        "party_elite_support_score",
        "party_elite_fragmentation_score",
        "party_context_support",
        "party_context_confidence",
        "party_context_support_weighted",
        "party_context_support_weighted_centered",
        "organization_strength",
        "outsider_status",
    ]
    out = base[["election_id", "region_id", "slot"]].copy()
    out["candidate_name"] = base["candidate_name"].astype(str).to_numpy() if "candidate_name" in base.columns else ""
    context = _load_candidate_party_speech_context()
    if context.empty:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]
    context = context.rename(columns={"confidence": "party_context_confidence"})
    out = out.merge(
        context.drop(columns=["available_date"]),
        on=["election_id", "slot", "candidate_name"],
        how="left",
    )
    for column in columns[3:11] + ["organization_strength", "outsider_status"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["party_context_support_weighted"] = (
        out["party_context_support"] * out["party_context_confidence"]
    )
    out["party_context_support_weighted_centered"] = (
        out["party_context_support_weighted"]
        - out.groupby("election_id")["party_context_support_weighted"].transform("mean")
    )
    return out[columns]


def _load_candidate_party_tone_gap() -> pd.DataFrame:
    """Load optional same-party versus cross-party tone-gap features."""

    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "same_party_positive_tone",
        "same_party_negative_tone",
        "same_party_net_tone",
        "same_party_supportive_tone",
        "cross_party_positive_tone",
        "cross_party_negative_tone",
        "cross_party_net_tone",
        "cross_party_adverse_tone",
        "party_tone_contrast",
        "same_party_endorsement_proxy",
        "same_party_defense_proxy",
        "cross_party_attack_proxy",
        "cross_party_rebuttal_proxy",
        "party_stance_signal",
        "party_stance_signal_centered",
        "party_stance_proxy_available",
        "same_party_supportive_tone_centered",
        "cross_party_positive_tone_centered",
        "cross_party_adverse_tone_centered",
        "party_tone_contrast_centered",
        "manual_valence_coverage",
        "available_date",
        "confidence",
    ]
    frame = _read_csv_if_exists(CANDIDATE_PARTY_TONE_GAP)
    required = {"election_id", "slot", "candidate_name", "same_party_supportive_tone", "available_date", "confidence"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    out["party_stance_proxy_available"] = float("party_stance_signal_centered" in out.columns)
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0 if column not in {"election_id", "slot", "candidate_name", "available_date"} else ""
    out = filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="candidate_party_tone_gap",
    )
    numeric_columns = [column for column in columns[3:] if column != "available_date"]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    bounded = [
        "same_party_positive_tone",
        "same_party_negative_tone",
        "same_party_supportive_tone",
        "cross_party_positive_tone",
        "cross_party_negative_tone",
        "cross_party_adverse_tone",
        "same_party_endorsement_proxy",
        "same_party_defense_proxy",
        "cross_party_attack_proxy",
        "cross_party_rebuttal_proxy",
        "manual_valence_coverage",
        "confidence",
    ]
    for column in bounded:
        out[column] = out[column].clip(0.0, 1.0)
    for column in [
        "same_party_net_tone",
        "cross_party_net_tone",
        "party_tone_contrast",
        "same_party_endorsement_proxy",
        "same_party_defense_proxy",
        "cross_party_attack_proxy",
        "cross_party_rebuttal_proxy",
        "party_stance_signal",
        "party_stance_signal_centered",
        "party_stance_proxy_available",
        "same_party_supportive_tone_centered",
        "cross_party_positive_tone_centered",
        "cross_party_adverse_tone_centered",
        "party_tone_contrast_centered",
    ]:
        out[column] = out[column].clip(-1.0, 1.0)
    return out[columns]


def _candidate_party_tone_gap_features(base: pd.DataFrame) -> pd.DataFrame:
    """Attach same-party and cross-party treatment tone features."""

    columns = [
        "election_id",
        "region_id",
        "slot",
        "same_party_positive_tone",
        "same_party_negative_tone",
        "same_party_net_tone",
        "same_party_supportive_tone",
        "cross_party_positive_tone",
        "cross_party_negative_tone",
        "cross_party_net_tone",
        "cross_party_adverse_tone",
        "party_tone_contrast",
        "party_stance_signal",
        "party_stance_signal_centered",
        "same_party_supportive_tone_centered",
        "cross_party_positive_tone_centered",
        "cross_party_adverse_tone_centered",
        "party_tone_contrast_centered",
        "manual_valence_coverage",
        "party_tone_confidence",
    ]
    out = base[["election_id", "region_id", "slot"]].copy()
    out["candidate_name"] = base["candidate_name"].astype(str).to_numpy() if "candidate_name" in base.columns else ""
    tone = _load_candidate_party_tone_gap()
    if tone.empty:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]
    tone = tone.rename(columns={"confidence": "party_tone_confidence"})
    out = out.merge(
        tone.drop(columns=["available_date"]),
        on=["election_id", "slot", "candidate_name"],
        how="left",
    )
    for column in columns[3:]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out[columns]


def _orientation_label(row: pd.Series) -> str:
    """Classify a candidate into a broad ideological lane for split-risk checks."""

    bloc = str(row.get("bloc", ""))
    conservative = float(row.get("landscape_axis_conservative", 0.0))
    liberal = float(row.get("landscape_axis_liberal", 0.0))
    progressive = float(row.get("landscape_axis_progressive", 0.0))
    reform = float(row.get("landscape_axis_reform", 0.0))
    if bloc == "국민의힘" or conservative >= 0.42:
        return "conservative"
    if bloc in {"더불어민주당", "진보정당계"}:
        return "liberal"
    if liberal + progressive + reform >= conservative + 0.20:
        return "liberal_centrist"
    if conservative >= liberal + progressive * 0.50:
        return "conservative_centrist"
    return "centrist"


def _orientation_affinity(left: pd.Series, right: pd.Series) -> float:
    """Estimate whether two active candidates occupy substitutable voter lanes."""

    left_label = _orientation_label(left)
    right_label = _orientation_label(right)
    if left_label == right_label:
        return 1.0
    pair = {left_label, right_label}
    if pair == {"conservative", "conservative_centrist"}:
        return 0.85
    if pair == {"liberal_centrist", "centrist"}:
        return 0.65
    if pair == {"liberal", "liberal_centrist"}:
        return 0.70
    if pair == {"conservative", "centrist"}:
        return 0.35
    if pair == {"liberal", "centrist"}:
        return 0.45
    if pair == {"conservative_centrist", "centrist"}:
        return 0.50
    return 0.0


def _same_orientation_external_features(base: pd.DataFrame) -> pd.DataFrame:
    """Model vote dispersion only when a similar active external candidate exists."""

    columns = [
        "election_id",
        "region_id",
        "slot",
        "same_orientation_external_pressure",
        "same_orientation_party_weakness",
        "same_orientation_dispersion_risk",
        "same_orientation_anchor_pressure",
    ]
    out = base[["election_id", "region_id", "slot"]].copy()
    required = {
        "candidate_name",
        "bloc",
        "organization_strength",
        "party_context_support",
        "party_context_confidence",
        "outsider_status",
        "third_viability",
        *{f"landscape_axis_{column}" for column in LANDSCAPE_VECTOR_COLUMNS},
    }
    if not required.issubset(base.columns):
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]

    candidates = (
        base[
            [
                "election_id",
                "slot",
                "candidate_name",
                "bloc",
                "organization_strength",
                "party_context_support",
                "party_context_confidence",
                "outsider_status",
                "third_viability",
                *[f"landscape_axis_{column}" for column in LANDSCAPE_VECTOR_COLUMNS],
            ]
        ]
        .drop_duplicates(["election_id", "slot"])
        .copy()
    )
    for column in [
        "organization_strength",
        "party_context_support",
        "party_context_confidence",
        "outsider_status",
        "third_viability",
        *[f"landscape_axis_{axis}" for axis in LANDSCAPE_VECTOR_COLUMNS],
    ]:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0.0)
    candidates["party_cohesion"] = (
        0.50 * candidates["organization_strength"].clip(0.0, 1.0)
        + 0.35 * candidates["party_context_support"].clip(0.0, 1.0)
        + 0.15 * candidates["party_context_confidence"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    candidates["candidate_viability"] = np.where(
        candidates["slot"].astype(str).isin({"A", "B"}),
        1.0,
        candidates["third_viability"].clip(0.0, 1.0),
    )

    rows: list[dict[str, object]] = []
    for election_id, group in candidates.groupby("election_id"):
        group = group.reset_index(drop=True)
        for _, candidate in group.iterrows():
            external_pressure = 0.0
            anchor_pressure = 0.0
            for _, other in group.iterrows():
                if str(candidate["slot"]) == str(other["slot"]):
                    continue
                affinity = _orientation_affinity(candidate, other)
                if affinity <= 0.0:
                    continue
                external_pressure = max(
                    external_pressure,
                    affinity * float(other["candidate_viability"]),
                )
                anchor_pressure = max(
                    anchor_pressure,
                    affinity
                    * float(other["party_cohesion"])
                    * float(other["candidate_viability"]),
                )
            party_weakness = 1.0 - float(candidate["party_cohesion"])
            outsider_status = float(candidate["outsider_status"])
            rows.append(
                {
                    "election_id": election_id,
                    "slot": candidate["slot"],
                    "same_orientation_external_pressure": external_pressure,
                    "same_orientation_party_weakness": party_weakness,
                    "same_orientation_dispersion_risk": external_pressure * party_weakness,
                    "same_orientation_anchor_pressure": anchor_pressure * outsider_status,
                }
            )
    if not rows:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]
    features = pd.DataFrame(rows)
    out = out.merge(features, on=["election_id", "slot"], how="left")
    for column in columns[3:]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out[columns]


def _load_candidate_public_treatment() -> pd.DataFrame:
    """Load optional assembly-derived public treatment proxy features."""

    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "serious_contender_score",
        "legitimacy_score",
        "negative_treatment_score",
        "scandal_salience_score",
        "fatigue_score",
        "alternative_score",
        "protest_vote_score",
        "ridicule_or_gaffe_score",
        "public_treatment_support",
        "public_treatment_support_centered",
        "available_date",
        "confidence",
    ]
    frame = _read_csv_if_exists(CANDIDATE_PUBLIC_TREATMENT)
    required = {"election_id", "slot", "candidate_name", "public_treatment_support_centered", "available_date", "confidence"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0 if column not in {"election_id", "slot", "candidate_name", "available_date"} else ""
    out = filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="candidate_public_treatment",
    )
    for column in columns[3:13] + ["confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    bounded = [
        "serious_contender_score",
        "legitimacy_score",
        "negative_treatment_score",
        "scandal_salience_score",
        "fatigue_score",
        "alternative_score",
        "protest_vote_score",
        "ridicule_or_gaffe_score",
        "confidence",
    ]
    for column in bounded:
        out[column] = out[column].clip(0.0, 1.0)
    for column in ["public_treatment_support", "public_treatment_support_centered"]:
        out[column] = out[column].clip(-1.0, 1.0)
    return out[columns]


def _candidate_public_treatment_features(base: pd.DataFrame) -> pd.DataFrame:
    """Attach assembly-derived public treatment proxy features."""

    columns = [
        "election_id",
        "region_id",
        "slot",
        "serious_contender_score",
        "legitimacy_score",
        "negative_treatment_score",
        "scandal_salience_score",
        "fatigue_score",
        "alternative_score",
        "protest_vote_score",
        "ridicule_or_gaffe_score",
        "public_treatment_support",
        "public_treatment_support_centered",
        "public_treatment_confidence",
    ]
    out = base[["election_id", "region_id", "slot"]].copy()
    out["candidate_name"] = base["candidate_name"].astype(str).to_numpy() if "candidate_name" in base.columns else ""
    treatment = _load_candidate_public_treatment()
    if treatment.empty:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]
    treatment = treatment.rename(columns={"confidence": "public_treatment_confidence"})
    out = out.merge(
        treatment.drop(columns=["available_date"]),
        on=["election_id", "slot", "candidate_name"],
        how="left",
    )
    for column in columns[3:]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out[columns]


def _load_candidate_vote_conversion_context() -> pd.DataFrame:
    """Load optional candidate cohesion and vote-conversion context."""

    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "candidate_weight",
        "coalition_cohesion",
        "coalition_mobilization_score",
        "wasted_vote_resistance",
        "major_party_gravity",
        "third_character_constraint",
        "third_candidate_overexposure_risk",
        "attention_to_support_gap",
        "conversion_capacity",
        "available_date",
        "confidence",
    ]
    frame = _read_csv_if_exists(CANDIDATE_VOTE_CONVERSION_CONTEXT)
    required = {
        "election_id",
        "slot",
        "candidate_name",
        "candidate_weight",
        "wasted_vote_resistance",
        "conversion_capacity",
        "available_date",
        "confidence",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0 if column not in {"election_id", "slot", "candidate_name", "available_date"} else ""
    out = filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="candidate_vote_conversion_context",
    )
    for column in columns[3:10] + ["confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out[columns]


def _candidate_vote_conversion_context_features(base: pd.DataFrame) -> pd.DataFrame:
    """Attach candidate cohesion and wasted-vote resistance features."""

    columns = [
        "election_id",
        "region_id",
        "slot",
        "candidate_weight",
        "coalition_cohesion",
        "coalition_mobilization_score",
        "wasted_vote_resistance",
        "major_party_gravity",
        "third_character_constraint",
        "third_candidate_overexposure_risk",
        "attention_to_support_gap",
        "conversion_capacity",
        "conversion_capacity_centered",
        "coalition_mobilization_centered",
        "candidate_conversion_confidence",
    ]
    out = base[["election_id", "region_id", "slot"]].copy()
    out["candidate_name"] = base["candidate_name"].astype(str).to_numpy() if "candidate_name" in base.columns else ""
    context = _load_candidate_vote_conversion_context()
    if context.empty:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]
    context = context.rename(columns={"confidence": "candidate_conversion_confidence"})
    out = out.merge(
        context.drop(columns=["available_date"]),
        on=["election_id", "slot", "candidate_name"],
        how="left",
    )
    for column in [
        "candidate_weight",
        "coalition_cohesion",
        "coalition_mobilization_score",
        "wasted_vote_resistance",
        "major_party_gravity",
        "third_character_constraint",
        "third_candidate_overexposure_risk",
        "attention_to_support_gap",
        "conversion_capacity",
        "candidate_conversion_confidence",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["conversion_capacity_centered"] = (
        out["conversion_capacity"] - out.groupby("election_id")["conversion_capacity"].transform("mean")
    )
    out["coalition_mobilization_centered"] = (
        out["coalition_mobilization_score"]
        - out.groupby("election_id")["coalition_mobilization_score"].transform("mean")
    )
    return out[columns]


def _third_candidate_competitiveness_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Gate third-candidate attention and characterize how it competes."""

    keys = ["election_id", "region_id", "slot"]
    out = frame[keys].copy()
    required = {
        "third_viability",
        "third_profile_confidence",
        "candidate_weight",
        "wasted_vote_resistance",
        "coalition_mobilization_score",
        "conversion_capacity",
    }
    if not required.issubset(frame.columns):
        out["third_competitiveness_gate"] = 0.0
        out["third_competitiveness_multiplier"] = 1.0
        out["third_regime_competitiveness"] = 0.0
        out["third_regime_two_way_score"] = 1.0
        out["third_regime_niche_minor_score"] = 0.0
        out["third_regime_reform_minor_score"] = 0.0
        out["third_regime_bloc_split_score"] = 0.0
        out["third_regime_independent_pole_score"] = 0.0
        out["third_regime_character_multiplier"] = 1.0
        out["third_regime_character"] = "two_way_withdrawn_or_absent"
        return out

    conversion_base = (
        0.35 * pd.to_numeric(frame["candidate_weight"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(frame["wasted_vote_resistance"], errors="coerce").fillna(0.0)
        + 0.20
        * pd.to_numeric(frame["coalition_mobilization_score"], errors="coerce").fillna(0.0)
        + 0.15 * pd.to_numeric(frame["conversion_capacity"], errors="coerce").fillna(0.0)
    ).clip(0.0, 1.0)
    viability = pd.to_numeric(frame["third_viability"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    confidence = (
        pd.to_numeric(frame["third_profile_confidence"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    is_third = frame["slot"].astype(str).eq("C")
    out["third_competitiveness_gate"] = np.where(
        is_third,
        np.sqrt((viability * confidence * conversion_base).clip(0.0, 1.0)),
        0.0,
    )
    election_gate = out.groupby("election_id")["third_competitiveness_gate"].transform("max")
    multiplier = (election_gate / THIRD_COMPETITIVENESS_REFERENCE).clip(
        THIRD_COMPETITIVENESS_MULTIPLIER_FLOOR,
        THIRD_COMPETITIVENESS_MULTIPLIER_CAP,
    )
    out["third_competitiveness_multiplier"] = np.where(election_gate.gt(0.0), multiplier, 1.0)

    has_third = frame["slot"].astype(str).eq("C").groupby(frame["election_id"]).transform("max")
    centrist_source = (
        frame["third_centrist_appeal"]
        if "third_centrist_appeal" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    centrist = (
        pd.to_numeric(centrist_source, errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .groupby(frame["election_id"])
        .transform("max")
    )
    anti_major_source = (
        frame["third_anti_major_party_appeal"]
        if "third_anti_major_party_appeal" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    anti_major = (
        pd.to_numeric(anti_major_source, errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .groupby(frame["election_id"])
        .transform("max")
    )
    competitiveness = 1.0 / (1.0 + np.exp(-(election_gate - 0.25) / 0.06))
    minor_mass = 1.0 - competitiveness
    raw = pd.DataFrame(
        {
            "niche_minor": minor_mass * (1.0 - anti_major),
            "reform_minor": minor_mass * anti_major,
            "bloc_split": competitiveness
            * (1.0 - centrist)
            * (0.50 + 0.50 * anti_major),
            "independent_pole": competitiveness
            * centrist
            * (0.50 + 0.50 * anti_major),
        },
        index=frame.index,
    )
    total = raw.sum(axis=1).replace(0.0, 1.0)
    probabilities = raw.div(total, axis=0)
    probabilities.loc[~has_third, :] = 0.0

    out["third_regime_competitiveness"] = np.where(has_third, competitiveness, 0.0)
    out["third_regime_two_way_score"] = np.where(
        has_third,
        1.0 - competitiveness,
        1.0,
    )
    out["third_regime_niche_minor_score"] = probabilities["niche_minor"].to_numpy(float)
    out["third_regime_reform_minor_score"] = probabilities["reform_minor"].to_numpy(float)
    out["third_regime_bloc_split_score"] = probabilities["bloc_split"].to_numpy(float)
    out["third_regime_independent_pole_score"] = probabilities["independent_pole"].to_numpy(float)
    character = probabilities.idxmax(axis=1).astype(str)
    character = character.where(has_third, "two_way_withdrawn_or_absent")
    out["third_regime_character"] = character.to_numpy()
    out["third_regime_character_multiplier"] = np.where(
        has_third,
        0.90 * probabilities["niche_minor"]
        + 0.96 * probabilities["reform_minor"]
        + 1.04 * probabilities["bloc_split"]
        + 1.08 * probabilities["independent_pole"],
        1.0,
    )
    return out


def _finalize_candidate_regionalism_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a concentration-only regional signal without a national bonus."""

    out = frame.copy()
    raw_source = (
        out["candidate_regional_base_raw"]
        if "candidate_regional_base_raw" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    raw = pd.to_numeric(raw_source, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    character_factor = pd.Series(1.0, index=out.index)
    if "third_competitiveness_gate" in out.columns:
        third_gate = pd.to_numeric(
            out["third_competitiveness_gate"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)
        bloc_split_floor = pd.Series(0.0, index=out.index)
        if "third_regime_bloc_split_score" in out.columns:
            bloc_split_floor = np.sqrt(
                pd.to_numeric(
                    out["third_regime_bloc_split_score"], errors="coerce"
                ).fillna(0.0).clip(0.0, 1.0)
            )
        third_factor = np.maximum(third_gate, bloc_split_floor)
        character_factor = character_factor.where(
            ~out["slot"].astype(str).eq("C"), third_factor
        )
    out["candidate_regional_character_factor"] = character_factor
    raw = raw * character_factor
    # A bloc-split candidate's documented local organization must not be erased
    # by a weaker national-viability gate. The signal remains concentration-only
    # because the zero-sum centering below removes every national candidate bonus.
    out["candidate_regional_base_gated"] = raw
    candidate_mean = out.groupby(["election_id", "slot"])[
        "candidate_regional_base_gated"
    ].transform("mean")
    candidate_centered = out["candidate_regional_base_gated"] - candidate_mean
    region_mean = candidate_centered.groupby([out["election_id"], out["region_id"]]).transform("mean")
    regional_signal = (candidate_centered - region_mean).clip(-1.0, 1.0)
    if {"outsider_status", "organization_strength"}.issubset(out.columns):
        outsider = pd.to_numeric(out["outsider_status"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        organization = (
            pd.to_numeric(out["organization_strength"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        anchor_strength = _rederived_float(
            "regional_anchor_strength",
            CANDIDATE_REGIONAL_ANCHOR_STRENGTH,
        )
        anchor_strength = min(max(anchor_strength, 0.0), 4.0)
        anchor = (1.0 + anchor_strength * outsider * (1.0 - organization)).clip(
            1.0,
            CANDIDATE_REGIONAL_ANCHOR_CAP,
        )
    else:
        anchor = pd.Series(1.0, index=out.index)
    out["candidate_regional_anchor_multiplier"] = anchor
    anchored_signal = regional_signal * anchor
    anchored_region_mean = anchored_signal.groupby(
        [out["election_id"], out["region_id"]]
    ).transform("mean")
    out["candidate_regionalism_signal"] = (anchored_signal - anchored_region_mean).clip(-1.0, 1.0)
    return out


def _finalize_district_terrain_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a constituency-history signal with no national candidate bonus."""

    out = frame.copy()
    raw = pd.to_numeric(
        out.get(
            "landscape_inferred_district_prior", pd.Series(0.0, index=out.index)
        ),
        errors="coerce",
    ).fillna(0.0)
    evidence = pd.to_numeric(
        out.get(
            "landscape_inferred_district_evidence", pd.Series(0.0, index=out.index)
        ),
        errors="coerce",
    ).fillna(0.0)
    reliability = (evidence / 3.0).clip(0.0, 1.0)
    out["district_terrain_raw"] = raw
    out["district_terrain_reliability"] = reliability
    reliable_raw = raw * (0.50 + 0.50 * reliability)
    candidate_mean = reliable_raw.groupby([out["election_id"], out["slot"]]).transform(
        "mean"
    )
    candidate_centered = reliable_raw - candidate_mean
    region_mean = candidate_centered.groupby(
        [out["election_id"], out["region_id"]]
    ).transform("mean")
    out["district_terrain_signal"] = (candidate_centered - region_mean).clip(-1.0, 1.0)
    return out


def _finalize_within_bloc_regional_transfer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a zero-sum regional transfer for an active bloc-split candidate."""

    out = frame.copy()
    out["within_bloc_transfer_activation"] = 0.0
    out["within_bloc_transfer_base_profile"] = 0.0
    out["within_bloc_transfer_profile"] = 0.0
    out["within_bloc_same_lane_reservoir"] = 0.0
    out["within_bloc_reservoir_confirmation"] = 0.0
    out["within_bloc_personal_stronghold"] = 0.0
    out["within_bloc_stronghold_reinforcement"] = 0.0
    out["within_bloc_base_transfer_signal"] = 0.0
    out["within_bloc_reservoir_transfer_signal"] = 0.0
    out["within_bloc_stronghold_transfer_signal"] = 0.0
    out["within_bloc_regional_transfer_signal"] = 0.0
    required = {
        "slot",
        "candidate_name",
        "bloc",
        "third_regime_competitiveness",
        "third_regime_bloc_split_score",
        "third_regime_independent_pole_score",
        "candidate_regionalism_signal",
        "candidate_regional_base_gated",
        "district_terrain_signal",
        "district_terrain_reliability",
        "partisan_prior",
        "effective_election_count",
        *{f"landscape_axis_{column}" for column in LANDSCAPE_VECTOR_COLUMNS},
    }
    if out.empty or not required.issubset(out.columns):
        return out

    for election_id, election in out.groupby("election_id", sort=False):
        candidate_rows = election.drop_duplicates("slot").set_index("slot")
        if "C" not in candidate_rows.index:
            continue
        third = candidate_rows.loc["C"]
        competitiveness = float(
            pd.to_numeric(third["third_regime_competitiveness"], errors="coerce")
        )
        bloc_split = float(
            pd.to_numeric(third["third_regime_bloc_split_score"], errors="coerce")
        )
        independent_pole = float(
            pd.to_numeric(third["third_regime_independent_pole_score"], errors="coerce")
        )
        activation = float(
            np.clip(competitiveness * max(bloc_split - independent_pole, 0.0), 0.0, 1.0)
        )
        if activation <= 0.0:
            continue

        donor_affinity: dict[str, float] = {}
        for slot, candidate in candidate_rows.iterrows():
            if str(slot) == "C":
                continue
            affinity = _orientation_affinity(third, candidate)
            if affinity > 0.0:
                donor_affinity[str(slot)] = affinity
        affinity_total = sum(donor_affinity.values())
        if affinity_total <= 0.0:
            continue
        donor_weights = {
            slot: affinity / affinity_total for slot, affinity in donor_affinity.items()
        }

        third_mask = out["election_id"].eq(election_id) & out["slot"].astype(str).eq("C")
        third_rows = out.loc[third_mask]
        personal = pd.to_numeric(
            third_rows["candidate_regionalism_signal"], errors="coerce"
        ).fillna(0.0)
        district = pd.to_numeric(
            third_rows["district_terrain_signal"], errors="coerce"
        ).fillna(0.0)
        reliability = pd.to_numeric(
            third_rows["district_terrain_reliability"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0)

        def normalized_profile(values: pd.Series) -> pd.Series:
            centered = values - values.mean()
            peak = float(centered.abs().max())
            return centered / peak if peak > 0.0 else centered * 0.0

        personal_profile = normalized_profile(personal)
        profile = 0.50 * personal_profile
        profile += 0.50 * normalized_profile(district) * reliability
        profile = normalized_profile(profile)
        base_profile = profile.copy()

        reservoir = pd.Series(0.0, index=third_rows.index, dtype=float)
        for donor_slot, donor_weight in donor_weights.items():
            donor_rows = out.loc[
                out["election_id"].eq(election_id)
                & out["slot"].astype(str).eq(donor_slot)
            ].set_index("region_id")
            donor_prior = pd.to_numeric(
                donor_rows["partisan_prior"], errors="coerce"
            ).fillna(0.0)
            donor_evidence = pd.to_numeric(
                donor_rows["effective_election_count"], errors="coerce"
            ).fillna(0.0)
            donor_reliability = np.sqrt((donor_evidence / 5.0).clip(0.0, 1.0))
            mapped_prior = third_rows["region_id"].map(donor_prior).fillna(0.0)
            mapped_reliability = (
                third_rows["region_id"].map(donor_reliability).fillna(0.0)
            )
            reservoir += (
                donor_weight
                * mapped_prior.to_numpy(float)
                * mapped_reliability.to_numpy(float)
            )
        reservoir = normalized_profile(reservoir)
        confirmation = reservoir.where(profile * reservoir > 0.0, 0.0)
        confirmation *= 1.0 - profile.abs()
        balance_capacity = (1.0 - personal_profile.abs()).clip(0.0, 1.0)
        capacity_total = float(balance_capacity.sum())
        if capacity_total > 0.0:
            confirmation -= (
                float(confirmation.sum()) * balance_capacity / capacity_total
            )
        else:
            confirmation -= confirmation.mean()
        personal_base = pd.to_numeric(
            third_rows["candidate_regional_base_gated"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        personal_peak = float(personal_base.max())
        stronghold = (
            personal_base / personal_peak if personal_peak > 0.0 else personal_base * 0.0
        )
        reinforcement = stronghold.pow(2.0)
        stronghold_balance_capacity = (1.0 - stronghold).clip(0.0, 1.0)
        stronghold_capacity_total = float(stronghold_balance_capacity.sum())
        if stronghold_capacity_total > 0.0:
            reinforcement -= (
                float(reinforcement.sum())
                * stronghold_balance_capacity
                / stronghold_capacity_total
            )
        else:
            reinforcement -= reinforcement.mean()
        stronghold_gain = min(
            max(_rederived_float("within_bloc_stronghold_gain", 0.0), 0.0),
            0.5,
        )
        reservoir_gain = min(
            max(_rederived_float("within_bloc_reservoir_gain", 1.0), 0.0),
            1.0,
        )
        profile = profile + reservoir_gain * confirmation
        profile += stronghold_gain * reinforcement
        profile_by_region = dict(zip(third_rows["region_id"].astype(str), profile))
        base_profile_by_region = dict(
            zip(third_rows["region_id"].astype(str), base_profile)
        )
        reservoir_by_region = dict(zip(third_rows["region_id"].astype(str), reservoir))
        confirmation_by_region = dict(zip(third_rows["region_id"].astype(str), confirmation))
        stronghold_by_region = dict(zip(third_rows["region_id"].astype(str), stronghold))
        reinforcement_by_region = dict(
            zip(third_rows["region_id"].astype(str), reinforcement)
        )

        election_mask = out["election_id"].eq(election_id)
        region_profile = out.loc[election_mask, "region_id"].astype(str).map(profile_by_region).fillna(0.0)
        out.loc[election_mask, "within_bloc_transfer_activation"] = activation
        out.loc[election_mask, "within_bloc_transfer_base_profile"] = (
            out.loc[election_mask, "region_id"]
            .astype(str)
            .map(base_profile_by_region)
            .fillna(0.0)
            .to_numpy(float)
        )
        out.loc[election_mask, "within_bloc_transfer_profile"] = region_profile.to_numpy(float)
        out.loc[election_mask, "within_bloc_same_lane_reservoir"] = (
            out.loc[election_mask, "region_id"]
            .astype(str)
            .map(reservoir_by_region)
            .fillna(0.0)
            .to_numpy(float)
        )
        out.loc[election_mask, "within_bloc_reservoir_confirmation"] = (
            out.loc[election_mask, "region_id"]
            .astype(str)
            .map(confirmation_by_region)
            .fillna(0.0)
            .to_numpy(float)
        )
        out.loc[election_mask, "within_bloc_personal_stronghold"] = (
            out.loc[election_mask, "region_id"]
            .astype(str)
            .map(stronghold_by_region)
            .fillna(0.0)
            .to_numpy(float)
        )
        out.loc[election_mask, "within_bloc_stronghold_reinforcement"] = (
            out.loc[election_mask, "region_id"]
            .astype(str)
            .map(reinforcement_by_region)
            .fillna(0.0)
            .to_numpy(float)
        )
        component_profiles = {
            "within_bloc_base_transfer_signal": base_profile,
            "within_bloc_reservoir_transfer_signal": confirmation,
            "within_bloc_stronghold_transfer_signal": reinforcement,
        }
        component_profiles_by_region = {
            column: dict(zip(third_rows["region_id"].astype(str), component))
            for column, component in component_profiles.items()
        }
        for column, component in component_profiles.items():
            out.loc[third_mask, column] = activation * component.to_numpy(float)
        out.loc[third_mask, "within_bloc_regional_transfer_signal"] = (
            activation * profile.to_numpy(float)
        )
        for donor_slot, donor_weight in donor_weights.items():
            donor_mask = election_mask & out["slot"].astype(str).eq(donor_slot)
            donor_profile = (
                out.loc[donor_mask, "region_id"]
                .astype(str)
                .map(profile_by_region)
                .fillna(0.0)
            )
            for column, component_by_region in component_profiles_by_region.items():
                donor_component = (
                    out.loc[donor_mask, "region_id"]
                    .astype(str)
                    .map(component_by_region)
                    .fillna(0.0)
                )
                out.loc[donor_mask, column] = (
                    -activation * donor_weight * donor_component.to_numpy(float)
                )
            out.loc[donor_mask, "within_bloc_regional_transfer_signal"] = (
                -activation * donor_weight * donor_profile.to_numpy(float)
            )

    return out


def _load_candidate_neutral_issue_context() -> pd.DataFrame:
    """Load the optional pre-election Assembly neutral-issue signal."""

    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "assembly_neutral_issue_signal",
        "evidence_count",
        "context_neutral_count",
        "context_issue_overlap_count",
        "global_context_neutral_count",
        "global_context_issue_overlap_count",
        "global_context_structure_strength",
        "global_context_content_strength",
        "global_context_strength",
        "global_context_relative_strength",
        "coverage_gate_passed",
        "available_date",
        "confidence",
    ]
    frame = _read_csv_if_exists(CANDIDATE_NEUTRAL_ISSUE_CONTEXT)
    required = {
        "election_id",
        "slot",
        "candidate_name",
        "assembly_neutral_issue_signal",
        "available_date",
        "confidence",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0 if column not in {"election_id", "slot", "candidate_name", "available_date"} else ""
    out = filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="candidate_neutral_issue_context",
    )
    for column in columns[3:14] + ["confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["confidence"] = out["confidence"].clip(0.0, 1.0)
    return out[columns]


def _candidate_neutral_issue_context_features(base: pd.DataFrame) -> pd.DataFrame:
    """Attach candidate-level neutral issue signals without direct vote adjustment."""

    columns = [
        "election_id",
        "region_id",
        "slot",
        "assembly_neutral_issue_signal",
        "assembly_neutral_issue_confidence",
        "assembly_neutral_directional_evidence_count",
        "assembly_neutral_target_context_count",
        "assembly_neutral_global_context_count",
        "assembly_neutral_global_context_strength",
    ]
    out = base[["election_id", "region_id", "slot"]].copy()
    out["candidate_name"] = base["candidate_name"].astype(str).to_numpy() if "candidate_name" in base.columns else ""
    context = _load_candidate_neutral_issue_context()
    if context.empty:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]
    context = context.rename(
        columns={
            "confidence": "assembly_neutral_issue_confidence",
            "evidence_count": "assembly_neutral_directional_evidence_count",
            "context_neutral_count": "assembly_neutral_target_context_count",
            "global_context_neutral_count": "assembly_neutral_global_context_count",
            "global_context_strength": "assembly_neutral_global_context_strength",
        }
    )
    keep = [
        "election_id",
        "slot",
        "candidate_name",
        "assembly_neutral_issue_signal",
        "assembly_neutral_issue_confidence",
        "assembly_neutral_directional_evidence_count",
        "assembly_neutral_target_context_count",
        "assembly_neutral_global_context_count",
        "assembly_neutral_global_context_strength",
    ]
    out = out.merge(context[keep], on=["election_id", "slot", "candidate_name"], how="left")
    for column in columns[3:]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out[columns]


def _load_election_generation_weights() -> pd.DataFrame:
    """Load election-level generational electorate weights."""

    columns = ["election_id", "young_weight", "middle_weight", "senior_weight", "available_date", "notes"]
    frame = _read_csv_if_exists(ELECTION_GENERATION_WEIGHTS)
    required = {"election_id", "young_weight", "middle_weight", "senior_weight", "available_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = "" if column in {"election_id", "available_date", "notes"} else 0.0
    out = filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="election_generation_weights",
    )
    for column in ["young_weight", "middle_weight", "senior_weight"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    total = out[["young_weight", "middle_weight", "senior_weight"]].sum(axis=1)
    valid = total > 0
    for column in ["young_weight", "middle_weight", "senior_weight"]:
        out.loc[valid, column] = out.loc[valid, column] / total.loc[valid]
    return out[columns]


def _load_candidate_generation_profile() -> pd.DataFrame:
    """Load manual candidate-level generational appeal profiles."""

    columns = [
        "election_id",
        "slot",
        "candidate_name",
        "young_affinity",
        "middle_affinity",
        "senior_affinity",
        "available_date",
        "confidence",
        "notes",
    ]
    frame = _read_csv_if_exists(CANDIDATE_GENERATION_PROFILE)
    required = {"election_id", "slot", "young_affinity", "middle_affinity", "senior_affinity", "available_date", "confidence"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = "" if column in {"election_id", "slot", "candidate_name", "available_date", "notes"} else 0.0
    out = filter_available_by_election(
        out,
        ELECTION_DATES,
        source_name="candidate_generation_profile",
    )
    for column in ["young_affinity", "middle_affinity", "senior_affinity", "confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out[columns]


def _candidate_generation_features(base: pd.DataFrame) -> pd.DataFrame:
    """Attach centered generation-alignment features for each candidate row."""

    columns = [
        "election_id",
        "region_id",
        "slot",
        "young_affinity",
        "middle_affinity",
        "senior_affinity",
        "generation_support_score",
        "generation_alignment",
        "generation_youth_niche",
        "generation_confidence",
    ]
    out = base[["election_id", "region_id", "slot"]].copy()
    profile = _load_candidate_generation_profile()
    if profile.empty:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]

    weights = _load_election_generation_weights()
    out = out.merge(
        profile.drop(columns=["available_date", "notes"]),
        on=["election_id", "slot"],
        how="left",
    )
    if weights.empty:
        out["young_weight"] = 0.25
        out["middle_weight"] = 0.45
        out["senior_weight"] = 0.30
    else:
        out = out.merge(
            weights.drop(columns=["available_date", "notes"]),
            on="election_id",
            how="left",
        )
        out["young_weight"] = pd.to_numeric(out["young_weight"], errors="coerce").fillna(0.25)
        out["middle_weight"] = pd.to_numeric(out["middle_weight"], errors="coerce").fillna(0.45)
        out["senior_weight"] = pd.to_numeric(out["senior_weight"], errors="coerce").fillna(0.30)

    for column in ["young_affinity", "middle_affinity", "senior_affinity", "confidence"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["generation_support_score"] = (
        out["young_affinity"] * out["young_weight"]
        + out["middle_affinity"] * out["middle_weight"]
        + out["senior_affinity"] * out["senior_weight"]
    )
    out["_broad_base"] = out[["middle_affinity", "senior_affinity"]].max(axis=1)
    out["generation_youth_niche"] = (
        out["young_affinity"] - out["_broad_base"]
    ).clip(lower=0.0) * out["young_weight"]
    out["generation_confidence"] = out["confidence"]
    out["generation_alignment"] = (
        out["generation_support_score"]
        - out.groupby("election_id")["generation_support_score"].transform("mean")
    ) * out["generation_confidence"]
    out["generation_youth_niche"] = (
        out["generation_youth_niche"]
        - out.groupby("election_id")["generation_youth_niche"].transform("mean")
    ) * out["generation_confidence"]
    return out[columns]


def _apply_withdrawn_landscape_affinity(transfers: pd.DataFrame) -> pd.DataFrame:
    """Reweight withdrawn-candidate transfers by source/target political affinity."""

    out = transfers.copy()
    out["withdrawn_landscape_affinity"] = 1.0
    landscape = _load_candidate_political_landscape()
    if landscape.empty:
        return out

    vector_columns = [
        "conservative",
        "liberal",
        "progressive",
        "centrist",
        "anti_establishment",
        "reform",
        "regionalist",
    ]
    candidate_keys = ["election_id", "candidate_name"]
    source = (
        landscape.loc[landscape["candidate_role"].astype(str).str.lower() == "withdrawn"]
        .sort_values("confidence", ascending=False)
        .drop_duplicates(candidate_keys)
    )
    if source.empty:
        return out
    source = source[candidate_keys + vector_columns].rename(
        columns={column: f"source_{column}" for column in vector_columns}
    )

    target = (
        landscape.loc[landscape["candidate_role"].astype(str).str.lower() == "final"]
        .sort_values("confidence", ascending=False)
        .drop_duplicates(["election_id", "slot"])
    )
    if target.empty:
        return out
    target = target[["election_id", "slot", *vector_columns]].rename(
        columns={"slot": "target_slot", **{column: f"target_{column}" for column in vector_columns}}
    )

    joined = out.merge(source, on=candidate_keys, how="left").merge(
        target,
        on=["election_id", "target_slot"],
        how="left",
    )
    source_matrix = joined[[f"source_{column}" for column in vector_columns]].to_numpy(float)
    target_matrix = joined[[f"target_{column}" for column in vector_columns]].to_numpy(float)
    affinity = np.array(
        [_cosine_similarity(source_row, target_row) for source_row, target_row in zip(source_matrix, target_matrix)]
    )
    joined["_base_transfer_mass"] = (
        joined["transfer_rate"] * joined["voter_compliance"] * joined["confidence"]
    )
    joined["_affinity_mass"] = joined["_base_transfer_mass"] * (0.5 + 0.5 * affinity)
    group_keys = ["election_id", "candidate_name"]
    base_sum = joined.groupby(group_keys)["_base_transfer_mass"].transform("sum")
    affinity_sum = joined.groupby(group_keys)["_affinity_mass"].transform("sum")
    valid = (base_sum > 0) & (affinity_sum > 0) & np.isfinite(affinity)
    joined["withdrawn_landscape_affinity"] = 1.0
    joined.loc[valid, "withdrawn_landscape_affinity"] = (
        joined.loc[valid, "_affinity_mass"]
        / joined.loc[valid, "_base_transfer_mass"].replace(0.0, np.nan)
        * (base_sum[valid] / affinity_sum[valid])
    )
    out["withdrawn_landscape_affinity"] = joined["withdrawn_landscape_affinity"].fillna(1.0).to_numpy(float)
    return out


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return a bounded cosine similarity for non-negative political vectors."""

    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return 0.0
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    if norm == 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / norm, 0.0, 1.0))


def _withdrawn_candidate_transfer_features(base: pd.DataFrame) -> pd.DataFrame:
    """Expand latent withdrawn-candidate transfer events to active target slots."""

    columns = [
        "election_id",
        "region_id",
        "slot",
        "withdrawn_candidate_transfer",
        "withdrawn_candidate_viability",
        "withdrawn_transfer_rate",
        "withdrawn_voter_compliance",
        "withdrawn_transfer_confidence",
        "withdrawn_landscape_affinity",
    ]
    out = base[["election_id", "region_id", "slot"]].copy()
    transfers = _load_withdrawn_candidate_transfers()
    if transfers.empty:
        for column in columns[3:]:
            out[column] = 0.0
        return out[columns]

    out = out.merge(
        transfers.drop(columns=["candidate_name"]),
        on=["election_id", "slot"],
        how="left",
    )
    for column in columns[3:]:
        out[column] = out[column].fillna(0.0)
    return out[columns]


def assemble() -> pd.DataFrame:
    """Assemble election-region-slot observations for the OLS engine."""

    results = pd.read_csv(RESULTS)
    salience = pd.read_csv(SALIENCE)
    link = pd.read_csv(LINK)

    results["bloc"] = results["party_name"].map(party_bloc).map(normalize_bloc)
    results["eidx"] = results["election_id"].map({election_id: i for i, election_id in enumerate(ORDER)})
    issue_advantage, region_issue_fit = _issue_features(salience, link, results["region_id"])
    coalition_events = _load_coalition_events()
    issue_advantage, region_issue_fit = _apply_coalition_events(
        issue_advantage,
        region_issue_fit,
        coalition_events,
    )
    excluded_slots = _excluded_event_slots(coalition_events)
    scored_scope_exclusions = _load_scored_contest_scope_exclusions()
    issue_signal_weights = _official_issue_signal_weights(results, excluded_slots)
    history = load_bloc_history(BLOC_HISTORY, presidential_results=results)

    result_bloc_share = (
        results.groupby(["election_id", "region_id", "bloc"], as_index=False)["vote_share"]
        .sum()
    )
    history_presidential = history.loc[
        history["election_type"] == "presidential",
        ["election_id", "region_id", "bloc", "vote_share"],
    ].copy()
    result_elections = set(result_bloc_share["election_id"].astype(str))
    history_presidential = history_presidential.loc[
        ~history_presidential["election_id"].astype(str).isin(result_elections)
    ]
    bloc_share = pd.concat([result_bloc_share, history_presidential], ignore_index=True)
    national_share = (
        results.groupby(["election_id", "bloc"], as_index=False)["vote_share"]
        .mean()
        .rename(columns={"vote_share": "national_share"})
    )
    if not history_presidential.empty:
        history_national_share = (
            history_presidential.groupby(["election_id", "bloc"], as_index=False)["vote_share"]
            .mean()
            .rename(columns={"vote_share": "national_share"})
        )
        national_share = pd.concat([national_share, history_national_share], ignore_index=True)
    bloc_share = bloc_share.merge(national_share, on=["election_id", "bloc"])
    bloc_share["lean"] = bloc_share["vote_share"] - bloc_share["national_share"]

    rows: list[dict[str, object]] = []
    for row in results.itertuples():
        active_slot = str(row.is_active_slot).lower() in {"1", "true", "yes", "y"}
        if row.slot == "alpha" or pd.isna(row.eidx) or not active_slot:
            continue
        if (row.election_id, row.slot) in excluded_slots:
            continue
        regional_base = 0.0
        if row.election_id in REGIONAL_BASE_ORDER:
            base_index = REGIONAL_BASE_ORDER.index(row.election_id)
        else:
            base_index = -1
        if base_index > 0:
            previous_election = REGIONAL_BASE_ORDER[base_index - 1]
            lean = bloc_share.loc[
                (bloc_share["election_id"] == previous_election)
                & (bloc_share["region_id"] == row.region_id)
                & (bloc_share["bloc"] == row.bloc),
                "lean",
            ]
            regional_base = float(lean.iloc[0]) if len(lean) else 0.0
        issue = issue_advantage.loc[
            (issue_advantage["election_id"] == row.election_id)
            & (issue_advantage["slot"] == row.slot),
            "issue_advantage",
        ]
        rows.append(
            {
                "election_id": row.election_id,
                "region_id": row.region_id,
                "slot": row.slot,
                "candidate_name": row.candidate_name,
                "bloc": row.bloc,
                "is_scored_contest_row": (row.election_id, row.slot)
                not in scored_scope_exclusions,
                "votes": float(row.votes) if not pd.isna(row.votes) else 0.0,
                "vote_share": float(row.vote_share),
                "slot_A": 1.0 if row.slot == "A" else 0.0,
                "slot_B": 1.0 if row.slot == "B" else 0.0,
                "regional_base": regional_base,
                "issue_advantage": float(issue.iloc[0]) if len(issue) else 0.0,
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.merge(region_issue_fit, on=["election_id", "region_id", "slot"], how="left")
    frame["rif"] = frame["rif"].fillna(0.0)
    frame = frame.merge(issue_signal_weights, on=["election_id", "slot"], how="left")
    frame["issue_signal_weight"] = frame["issue_signal_weight"].fillna(MINOR_SLOT_BASE_ISSUE_WEIGHT)
    frame["issue_advantage_raw"] = frame["issue_advantage"]
    frame["rif_raw"] = frame["rif"]
    frame["issue_advantage"] = frame["issue_advantage"] * frame["issue_signal_weight"]
    frame["rif"] = frame["rif"] * frame["issue_signal_weight"]
    economic_features = _economic_context_features(
        _load_economic_indicators(),
        _load_economic_slot_alignment(),
    )
    frame = frame.merge(economic_features, on=["election_id", "slot"], how="left")
    for column in [
        "economic_context_effect",
        "economic_stress_index",
        "trade_context_effect",
        "trade_stress_index",
        "economic_responsibility_score",
        "real_gdp_growth_yoy",
        "current_account_12m_sum",
    ]:
        frame[column] = frame[column].fillna(0.0)
    interest_rate_features = _interest_rate_context_features(
        _load_interest_rate_indicators(),
        _load_economic_slot_alignment(),
    )
    frame = frame.merge(interest_rate_features, on=["election_id", "slot"], how="left")
    for column in [
        "interest_rate_context_effect",
        "interest_rate_stress_index",
        "bok_base_rate",
        "bok_base_rate_12m_change",
    ]:
        frame[column] = frame[column].fillna(0.0)
    frame["interest_rate_latest_period"] = frame["interest_rate_latest_period"].fillna("not_available")
    housing_features = _housing_context_features(_load_housing_price_index())
    frame = frame.merge(housing_features, on=["election_id", "region_id"], how="left")
    frame["housing_price_period"] = frame["housing_price_period"].fillna("not_available")
    for column in ["housing_price_index", "housing_price_yoy_change_pct"]:
        frame[column] = frame[column].fillna(0.0)
    housing_pressure = _housing_pressure_features(
        _load_housing_price_index(),
        _load_housing_slot_alignment(),
        housing_sgg=_load_housing_price_index_sgg(),
    )
    frame = frame.merge(housing_pressure, on=["election_id", "region_id", "slot"], how="left")
    for column in [
        "housing_pressure_effect",
        "housing_cumulative_change_pct",
        "housing_responsibility_score",
        "housing_sgg_median_change_pct",
        "housing_sgg_dispersion",
        "housing_sgg_positive_share",
        "housing_sgg_count",
        "housing_pressure_intensity",
    ]:
        frame[column] = frame[column].fillna(0.0)
    for column in ["housing_baseline_period", "housing_current_period"]:
        frame[column] = frame[column].fillna("not_available")
    kospi_features = _kospi_context_features(
        _load_kospi_daily(),
        _load_economic_slot_alignment(),
    )
    frame = frame.merge(kospi_features, on=["election_id", "slot"], how="left")
    for column in [
        "kospi_context_effect",
        "kospi_market_stress_index",
        "kospi_close",
        "kospi_return_3m",
        "kospi_return_12m",
        "kospi_drawdown_12m",
        "kospi_volatility_3m",
    ]:
        frame[column] = frame[column].fillna(0.0)
    frame["kospi_latest_date"] = frame["kospi_latest_date"].fillna("not_available")
    frame = _apply_epoch_macro_context_weights(frame)

    candidate_regional_features = _candidate_regional_base_features(frame)
    frame = frame.merge(
        candidate_regional_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "candidate_regional_affinity",
        "candidate_regional_organization",
        "candidate_regional_confidence",
        "candidate_regional_base_raw",
    ]:
        frame[column] = frame[column].fillna(0.0)

    third_candidate_features = _third_candidate_structure_features(frame)
    frame = frame.merge(
        third_candidate_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "third_candidate_structure",
        "third_attention_score",
        "third_conversion_capacity",
        "third_attention_overhang",
        "third_viability",
        "third_centrist_appeal",
        "third_anti_major_party_appeal",
        "third_regional_base_overlap",
        "third_profile_confidence",
        "slotA_third_pressure",
        "slotB_third_pressure",
    ]:
        frame[column] = frame[column].fillna(0.0)

    withdrawn_transfer_features = _withdrawn_candidate_transfer_features(frame)
    frame = frame.merge(
        withdrawn_transfer_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "withdrawn_candidate_transfer",
        "withdrawn_candidate_viability",
        "withdrawn_transfer_rate",
        "withdrawn_voter_compliance",
        "withdrawn_transfer_confidence",
        "withdrawn_landscape_affinity",
    ]:
        frame[column] = frame[column].fillna(0.0)

    landscape_features = _candidate_political_landscape_features(frame)
    frame = frame.merge(
        landscape_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in LANDSCAPE_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0)

    frame = attach_bloc_prior(frame, history, ORDER)
    frame = _split_partisan_prior_layers(frame)
    inferred_prior_features = _candidate_landscape_inferred_prior_features(frame, history)
    frame = frame.drop(
        columns=[
            "landscape_inferred_prior",
            "landscape_inferred_prior_evidence",
            "landscape_inferred_district_prior",
            "landscape_inferred_district_evidence",
        ],
        errors="ignore",
    ).merge(
        inferred_prior_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "landscape_inferred_prior",
        "landscape_inferred_prior_evidence",
        "landscape_inferred_district_prior",
        "landscape_inferred_district_evidence",
    ]:
        frame[column] = frame[column].fillna(0.0)
    frame["slotA_prior"] = frame["slot_A"] * frame["partisan_prior"]
    frame["slotB_prior"] = frame["slot_B"] * frame["partisan_prior"]
    party_context_features = _candidate_party_speech_context_features(frame)
    frame = frame.merge(
        party_context_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "same_bloc_issue_alignment",
        "same_bloc_frame_convergence",
        "cross_bloc_attack_pressure",
        "intra_bloc_conflict_score",
        "party_elite_support_score",
        "party_elite_fragmentation_score",
        "party_context_support",
        "party_context_confidence",
        "party_context_support_weighted",
        "party_context_support_weighted_centered",
        "organization_strength",
        "outsider_status",
    ]:
        frame[column] = frame[column].fillna(0.0)
    party_tone_features = _candidate_party_tone_gap_features(frame)
    frame = frame.merge(
        party_tone_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "same_party_positive_tone",
        "same_party_negative_tone",
        "same_party_net_tone",
        "same_party_supportive_tone",
        "cross_party_positive_tone",
        "cross_party_negative_tone",
        "cross_party_net_tone",
        "cross_party_adverse_tone",
        "party_tone_contrast",
        "same_party_supportive_tone_centered",
        "cross_party_adverse_tone_centered",
        "party_tone_contrast_centered",
        "manual_valence_coverage",
        "party_tone_confidence",
    ]:
        frame[column] = frame[column].fillna(0.0)
    same_orientation_features = _same_orientation_external_features(frame)
    frame = frame.merge(
        same_orientation_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "same_orientation_external_pressure",
        "same_orientation_party_weakness",
        "same_orientation_dispersion_risk",
        "same_orientation_anchor_pressure",
    ]:
        frame[column] = frame[column].fillna(0.0)
    public_treatment_features = _candidate_public_treatment_features(frame)
    frame = frame.merge(
        public_treatment_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "serious_contender_score",
        "legitimacy_score",
        "negative_treatment_score",
        "scandal_salience_score",
        "fatigue_score",
        "alternative_score",
        "protest_vote_score",
        "ridicule_or_gaffe_score",
        "public_treatment_support",
        "public_treatment_support_centered",
        "public_treatment_confidence",
    ]:
        frame[column] = frame[column].fillna(0.0)
    conversion_context_features = _candidate_vote_conversion_context_features(frame)
    frame = frame.merge(
        conversion_context_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "candidate_weight",
        "coalition_cohesion",
        "coalition_mobilization_score",
        "wasted_vote_resistance",
        "major_party_gravity",
        "third_character_constraint",
        "third_candidate_overexposure_risk",
        "attention_to_support_gap",
        "conversion_capacity",
        "conversion_capacity_centered",
        "coalition_mobilization_centered",
        "candidate_conversion_confidence",
    ]:
        frame[column] = frame[column].fillna(0.0)
    third_competitiveness_features = _third_candidate_competitiveness_features(frame)
    frame = frame.merge(
        third_competitiveness_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    frame["third_competitiveness_gate"] = frame["third_competitiveness_gate"].fillna(0.0)
    frame["third_competitiveness_multiplier"] = frame[
        "third_competitiveness_multiplier"
    ].fillna(1.0)
    for column in [
        "third_regime_competitiveness",
        "third_regime_two_way_score",
        "third_regime_niche_minor_score",
        "third_regime_reform_minor_score",
        "third_regime_bloc_split_score",
        "third_regime_independent_pole_score",
    ]:
        frame[column] = frame[column].fillna(0.0)
    frame["third_regime_character_multiplier"] = frame[
        "third_regime_character_multiplier"
    ].fillna(1.0)
    frame["third_regime_character"] = frame["third_regime_character"].fillna(
        "two_way_withdrawn_or_absent"
    )
    neutral_issue_features = _candidate_neutral_issue_context_features(frame)
    frame = frame.merge(
        neutral_issue_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "assembly_neutral_issue_signal",
        "assembly_neutral_issue_confidence",
        "assembly_neutral_directional_evidence_count",
        "assembly_neutral_target_context_count",
        "assembly_neutral_global_context_count",
        "assembly_neutral_global_context_strength",
    ]:
        frame[column] = frame[column].fillna(0.0)
    generation_features = _candidate_generation_features(frame)
    frame = frame.merge(
        generation_features,
        on=["election_id", "region_id", "slot"],
        how="left",
    )
    for column in [
        "young_affinity",
        "middle_affinity",
        "senior_affinity",
        "generation_support_score",
        "generation_alignment",
        "generation_youth_niche",
        "generation_confidence",
    ]:
        frame[column] = frame[column].fillna(0.0)
    frame = _candidate_attention_support_features(frame)
    frame = _finalize_district_terrain_features(frame)
    frame = _finalize_candidate_regionalism_features(frame)
    frame = _finalize_within_bloc_regional_transfer_features(frame)
    frame = _attach_electorate_layer_features(frame, history, salience, link)
    return frame


def historical_presidential_warmup_frame(history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build pre-2002 presidential rows for rolling-origin warmup training only."""

    history = load_bloc_history(BLOC_HISTORY) if history is None else history.copy()
    if history.empty:
        return pd.DataFrame(columns=["election_id", "region_id", "slot", *PREDICTORS])

    presidential = history.loc[
        (history["election_type"] == "presidential")
        & (history["election_id"].isin(WARMUP_ORDER)),
        ["election_id", "region_id", "bloc", "vote_share"],
    ].copy()
    if presidential.empty:
        return pd.DataFrame(columns=["election_id", "region_id", "slot", *PREDICTORS])

    national = (
        presidential.groupby(["election_id", "bloc"], as_index=False)["vote_share"]
        .mean()
        .rename(columns={"vote_share": "national_share"})
    )
    selected = []
    for election_id, group in national.groupby("election_id"):
        mapped = group.loc[
            group["bloc"].map(lambda bloc: (str(election_id), str(bloc)) in WARMUP_PRESIDENTIAL_SLOTS)
        ].copy()
        if mapped.empty:
            mapped = group.nlargest(3, "national_share").copy()
            mapped["slot"] = ["A", "B", "C"][: len(mapped)]
            mapped["candidate_name"] = mapped["bloc"]
        else:
            mapped["slot"] = mapped["bloc"].map(
                lambda bloc: WARMUP_PRESIDENTIAL_SLOTS[(str(election_id), str(bloc))][0]
            )
            mapped["candidate_name"] = mapped["bloc"].map(
                lambda bloc: WARMUP_PRESIDENTIAL_SLOTS[(str(election_id), str(bloc))][1]
            )
        selected.append(mapped[["election_id", "bloc", "slot", "candidate_name"]])
    if not selected:
        return pd.DataFrame(columns=["election_id", "region_id", "slot", *PREDICTORS])

    selected_frame = pd.concat(selected, ignore_index=True)
    frame = presidential.merge(selected_frame, on=["election_id", "bloc"], how="inner")
    frame["votes"] = frame["vote_share"]
    frame["slot_A"] = (frame["slot"] == "A").astype(float)
    frame["slot_B"] = (frame["slot"] == "B").astype(float)
    for column in [
        "issue_advantage",
        "rif",
        "landscape_bloc_alignment",
        "landscape_centrist",
        "landscape_inferred_prior",
    ]:
        frame[column] = 0.0

    bloc_share = presidential.copy()
    national_share = (
        bloc_share.groupby(["election_id", "bloc"], as_index=False)["vote_share"]
        .mean()
        .rename(columns={"vote_share": "national_share"})
    )
    bloc_share = bloc_share.merge(national_share, on=["election_id", "bloc"], how="left")
    bloc_share["lean"] = bloc_share["vote_share"] - bloc_share["national_share"]
    frame["regional_base"] = 0.0
    for index, row in frame.iterrows():
        base_index = REGIONAL_BASE_ORDER.index(row["election_id"])
        if base_index <= 0:
            continue
        previous_election = REGIONAL_BASE_ORDER[base_index - 1]
        lean = bloc_share.loc[
            (bloc_share["election_id"] == previous_election)
            & (bloc_share["region_id"] == row["region_id"])
            & (bloc_share["bloc"] == row["bloc"]),
            "lean",
        ]
        if len(lean):
            frame.at[index, "regional_base"] = float(lean.iloc[0])

    frame = attach_bloc_prior(frame, history, REGIONAL_BASE_ORDER)
    frame = _split_partisan_prior_layers(frame)
    frame["slotA_prior"] = frame["slot_A"] * frame["partisan_prior"]
    frame["slotB_prior"] = frame["slot_B"] * frame["partisan_prior"]
    frame["votes"] = frame["vote_share"]
    for predictor in PREDICTORS:
        if predictor not in frame.columns:
            frame[predictor] = 0.0
        frame[predictor] = pd.to_numeric(frame[predictor], errors="coerce").fillna(0.0)
    return frame[
        [
            "election_id",
            "region_id",
            "slot",
            "candidate_name",
            "bloc",
            "votes",
            "vote_share",
            *PREDICTORS,
        ]
    ].copy()


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit OLS with an intercept and return beta, R-squared, covariance, residuals."""

    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ beta
    resid = y - yhat
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n_obs, n_cols = design.shape
    sigma2 = ss_res / max(n_obs - n_cols, 1)
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    return beta, r2, cov, resid


def ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = RIDGE_ALPHA,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardized ridge regression with an unpenalized intercept."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if sample_weight is None:
        weights = np.ones(len(y), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        weights = np.where(np.isfinite(weights), weights, 1.0)
        weights = np.clip(weights, 0.05, None)
    if len(weights) != len(y):
        raise ValueError("sample_weight length must match y length")

    weights = weights / max(float(weights.mean()), 1e-12)
    weight_sum = max(float(weights.sum()), 1e-12)
    means = (X * weights[:, None]).sum(axis=0) / weight_sum
    centered = X - means
    scales = np.sqrt((centered**2 * weights[:, None]).sum(axis=0) / weight_sum)
    scales[scales == 0] = 1.0
    x_scaled = centered / scales
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_y = y * np.sqrt(weights)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    penalized = weighted_design.T @ weighted_design + alpha * penalty
    beta = np.linalg.solve(penalized, weighted_design.T @ weighted_y)
    yhat = design @ beta
    resid = y - yhat
    y_mean = float((y * weights).sum() / weight_sum)
    ss_res = float((weights * resid**2).sum())
    ss_tot = float((weights * (y - y_mean) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n_obs, n_cols = design.shape
    effective_n = (weights.sum() ** 2) / max(float((weights**2).sum()), 1e-12)
    sigma2 = ss_res / max(effective_n - n_cols, 1)
    bread = np.linalg.pinv(penalized)
    cov = sigma2 * bread @ (weighted_design.T @ weighted_design) @ bread.T
    return beta, r2, cov, resid, means, scales


def election_epoch_sample_weight(frame: pd.DataFrame) -> np.ndarray:
    """Return historical sample weights so older political regimes train less strongly."""

    if frame.empty or "election_id" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    return (
        frame["election_id"]
        .astype(str)
        .map(ELECTION_EPOCH_SAMPLE_WEIGHTS)
        .fillna(1.0)
        .astype(float)
        .to_numpy()
    )


def ridge_predict(beta: np.ndarray, X: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Predict from a standardized ridge fit."""

    x_scaled = (X - means) / scales
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    return design @ beta


def normalized_vote_share_target(frame: pd.DataFrame) -> np.ndarray:
    """Return actual vote share normalized within evaluated election-region slots."""

    target_sum = frame.groupby(["election_id", "region_id"])["vote_share"].transform("sum")
    target = np.where(
        target_sum.to_numpy(float) > 0,
        frame["vote_share"].to_numpy(float) / target_sum.to_numpy(float),
        0.0,
    )
    return target


def normalize_vote_share_predictions(frame: pd.DataFrame, pred: np.ndarray | pd.Series) -> np.ndarray:
    """Normalize row predictions to sum to 1 within each election-region.

    The engine evaluates active non-alpha slots only. Predictions are therefore
    interpreted as shares within the modeled candidate set.
    """

    out = frame[["election_id", "region_id"]].copy()
    out["_pred_raw"] = np.asarray(pred, dtype=float)
    out["_pred_nonnegative"] = out["_pred_raw"].clip(lower=0.0)
    pred_sum = out.groupby(["election_id", "region_id"])["_pred_nonnegative"].transform("sum")
    group_size = out.groupby(["election_id", "region_id"])["_pred_nonnegative"].transform("size")
    fallback = 1.0 / group_size.replace(0, np.nan)
    normalized = fallback.fillna(0.0).to_numpy(float).copy()
    positive = pred_sum.to_numpy(float) > 0
    normalized[positive] = (
        out["_pred_nonnegative"].to_numpy(float)[positive]
        / pred_sum.to_numpy(float)[positive]
    )
    return normalized


def apply_partisan_layer_prediction_moderation(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Limit regional prediction deviations by durable and movable support layers."""

    required = {"concrete_partisan_prior", "general_partisan_prior"}
    if frame.empty or not required.issubset(frame.columns):
        return np.asarray(pred, dtype=float)

    out = frame[["election_id", "region_id", "slot"]].copy()
    out["_pred"] = np.asarray(pred, dtype=float)
    national_slot_mean = out.groupby(["election_id", "slot"])["_pred"].transform("mean")
    concrete = pd.to_numeric(frame["concrete_partisan_prior"], errors="coerce").fillna(0.0).abs()
    general = pd.to_numeric(frame["general_partisan_prior"], errors="coerce").fillna(0.0).abs()
    cap = (
        PRIOR_MODERATION_MIN_DEVIATION
        + PRIOR_MODERATION_CORE_MULTIPLIER * concrete.to_numpy(float)
        + PRIOR_MODERATION_GENERAL_MULTIPLIER * general.to_numpy(float)
    )
    deviation = out["_pred"].to_numpy(float) - national_slot_mean.to_numpy(float)
    out["_moderated"] = national_slot_mean.to_numpy(float) + np.clip(deviation, -cap, cap)
    return normalize_vote_share_predictions(frame, out["_moderated"].to_numpy(float))


def apply_party_context_prediction_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Convert weak party cohesion into bounded supporter defection.

    Party context is deliberately not added to a candidate's total support.
    It only releases part of the candidate-aligned core/critical mass into the
    region's contestable pool, which is then allocated by the pre-adjustment
    prediction. Missing or zero-confidence context is an exact identity.
    """

    required = {
        "election_id",
        "region_id",
        "party_context_support",
        "party_context_confidence",
        "party_elite_fragmentation_score",
        "core_voting_mass",
        "critical_voting_mass",
    }
    if frame.empty or not required.issubset(frame.columns):
        return np.asarray(pred, dtype=float)
    baseline = normalize_vote_share_predictions(frame, np.asarray(pred, dtype=float))
    support = (
        pd.to_numeric(frame["party_context_support"], errors="coerce")
        .fillna(0.0)
        .clip(-1.0, 1.0)
        .to_numpy(float)
    )
    confidence = (
        pd.to_numeric(frame["party_context_confidence"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .to_numpy(float)
    )
    fragmentation = (
        pd.to_numeric(frame["party_elite_fragmentation_score"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .to_numpy(float)
    )
    core = (
        pd.to_numeric(frame["core_voting_mass"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(float)
    )
    critical = (
        pd.to_numeric(frame["critical_voting_mass"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(float)
    )

    support_strength = 0.5 * (support + 1.0)
    defection_risk = confidence * np.clip(
        0.65 * (1.0 - support_strength) + 0.35 * fragmentation,
        0.0,
        1.0,
    )
    released = defection_risk * (
        PARTY_CONTEXT_CORE_DEFECTION_CAP * core
        + PARTY_CONTEXT_CRITICAL_DEFECTION_CAP * critical
    )
    released = np.minimum(released, 0.95 * baseline)

    adjusted = baseline - released
    grouping = frame[["election_id", "region_id"]].copy()
    grouping["_baseline"] = baseline
    grouping["_released"] = released
    released_total = grouping.groupby(
        ["election_id", "region_id"]
    )["_released"].transform("sum").to_numpy(float)
    baseline_total = grouping.groupby(
        ["election_id", "region_id"]
    )["_baseline"].transform("sum").to_numpy(float)
    contestable_profile = np.divide(
        baseline,
        baseline_total,
        out=np.zeros_like(baseline),
        where=baseline_total > 1e-12,
    )
    adjusted += released_total * contestable_profile
    return normalize_vote_share_predictions(frame, adjusted)


def apply_party_tone_gap_prediction_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Apply an explicitly labeled party-stance proxy when available.

    The proxy distinguishes own-bloc endorsement/defense from other-bloc attack
    using candidate valence and bloc issue emphasis. It is not presented as
    sentence-level sentiment. Older generic tone columns remain only as a
    backward-compatible fallback for historical CSVs.
    """

    stance_required = {
        "party_stance_signal_centered",
        "party_tone_confidence",
        "manual_valence_coverage",
    }
    stance_disabled = os.getenv("POLL_PROJECT_DISABLE_PARTY_STANCE_PROXY", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    stance_enabled = os.getenv("POLL_PROJECT_ENABLE_PARTY_STANCE_PROXY", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if stance_enabled and not stance_disabled and not frame.empty and stance_required.issubset(frame.columns):
        adjusted = np.asarray(pred, dtype=float).copy()
        signal = pd.to_numeric(frame["party_stance_signal_centered"], errors="coerce").fillna(0.0)
        confidence = pd.to_numeric(frame["party_tone_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        coverage = pd.to_numeric(frame["manual_valence_coverage"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        available_source = (
            frame["party_stance_proxy_available"]
            if "party_stance_proxy_available" in frame.columns
            else pd.Series(0.0, index=frame.index)
        )
        available = pd.to_numeric(available_source, errors="coerce").fillna(0.0).clip(0.0, 1.0)
        if not available.gt(0.0).any():
            return np.asarray(pred, dtype=float)
        adjusted += (
            PARTY_STANCE_PROXY_ADJUSTMENT_SCALE
            * signal.to_numpy(float)
            * confidence.to_numpy(float)
            * coverage.to_numpy(float)
            * available.to_numpy(float)
        )
        return normalize_vote_share_predictions(frame, adjusted)

    legacy_enabled = os.getenv("POLL_PROJECT_USE_LEGACY_PARTY_TONE", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if not legacy_enabled:
        return np.asarray(pred, dtype=float)

    required = {
        "same_party_supportive_tone_centered",
        "cross_party_positive_tone_centered",
        "cross_party_adverse_tone_centered",
        "party_tone_confidence",
    }
    if frame.empty or not required.issubset(frame.columns):
        return np.asarray(pred, dtype=float)
    adjusted = np.asarray(pred, dtype=float).copy()
    same_support = pd.to_numeric(
        frame["same_party_supportive_tone_centered"],
        errors="coerce",
    ).fillna(0.0)
    cross_adverse = pd.to_numeric(
        frame["cross_party_adverse_tone_centered"],
        errors="coerce",
    ).fillna(0.0)
    cross_positive = pd.to_numeric(
        frame["cross_party_positive_tone_centered"],
        errors="coerce",
    ).fillna(0.0)
    confidence = pd.to_numeric(frame["party_tone_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    adjusted = (
        adjusted
        + SAME_PARTY_TONE_ADJUSTMENT_SCALE * same_support.to_numpy(float) * confidence.to_numpy(float)
        + CROSS_PARTY_POSITIVE_ADJUSTMENT_SCALE * cross_positive.to_numpy(float) * confidence.to_numpy(float)
        - CROSS_PARTY_ADVERSE_ADJUSTMENT_SCALE * cross_adverse.to_numpy(float) * confidence.to_numpy(float)
    )
    return normalize_vote_share_predictions(frame, adjusted)


def apply_same_orientation_external_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Apply split-risk only when a similar active external candidate exists."""

    required = {"same_orientation_dispersion_risk", "same_orientation_anchor_pressure"}
    if frame.empty or not required.issubset(frame.columns):
        return np.asarray(pred, dtype=float)
    adjusted = np.asarray(pred, dtype=float).copy()
    dispersion = pd.to_numeric(
        frame["same_orientation_dispersion_risk"],
        errors="coerce",
    ).fillna(0.0)
    anchor = pd.to_numeric(
        frame["same_orientation_anchor_pressure"],
        errors="coerce",
    ).fillna(0.0)
    dispersion_centered = dispersion - dispersion.groupby(frame["election_id"]).transform("mean")
    anchor_centered = anchor - anchor.groupby(frame["election_id"]).transform("mean")
    adjusted = (
        adjusted
        - SAME_ORIENTATION_DISPERSION_ADJUSTMENT_SCALE * dispersion_centered.to_numpy(float)
        - SAME_ORIENTATION_ANCHOR_ADJUSTMENT_SCALE * anchor_centered.to_numpy(float)
    )
    return normalize_vote_share_predictions(frame, adjusted)


def apply_public_treatment_prediction_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Apply a small assembly-discourse treatment adjustment."""

    required = {"public_treatment_support_centered", "serious_contender_score"}
    if frame.empty or not required.issubset(frame.columns):
        return np.asarray(pred, dtype=float)
    adjusted = np.asarray(pred, dtype=float).copy()
    support = pd.to_numeric(
        frame["public_treatment_support_centered"],
        errors="coerce",
    ).fillna(0.0)
    serious = pd.to_numeric(frame["serious_contender_score"], errors="coerce").fillna(0.0)
    serious_centered = serious - serious.groupby(frame["election_id"]).transform("mean")
    adjusted = (
        adjusted
        + PUBLIC_TREATMENT_SUPPORT_ADJUSTMENT_SCALE * support.to_numpy(float)
        + PUBLIC_TREATMENT_SERIOUS_ADJUSTMENT_SCALE * serious_centered.to_numpy(float)
    )
    return normalize_vote_share_predictions(frame, adjusted)


def apply_generation_prediction_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Apply a small bounded adjustment for candidate-generation fit."""

    required = {"generation_alignment", "generation_youth_niche", "generation_confidence"}
    if frame.empty or not required.issubset(frame.columns):
        return np.asarray(pred, dtype=float)
    adjusted = np.asarray(pred, dtype=float).copy()
    alignment = pd.to_numeric(frame["generation_alignment"], errors="coerce").fillna(0.0)
    youth_niche = pd.to_numeric(frame["generation_youth_niche"], errors="coerce").fillna(0.0)
    confidence = pd.to_numeric(frame["generation_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    adjusted = (
        adjusted
        + GENERATION_ALIGNMENT_ADJUSTMENT_SCALE * alignment.to_numpy(float)
        - GENERATION_YOUTH_NICHE_PENALTY_SCALE * youth_niche.to_numpy(float) * confidence.to_numpy(float)
    )
    return normalize_vote_share_predictions(frame, adjusted)


def apply_candidate_conversion_context_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Apply non-party candidate stature and third-candidate vote conversion.

    Party cohesion is excluded here because it is handled by supporter
    retention. This direct vote-share path uses only candidate-facing stature;
    coalition context remains relevant solely to the third-candidate
    wasted-vote penalty.
    """

    selected_scale = _rederived_float("conversion_scale")
    if selected_scale <= 0.0:
        return np.asarray(pred, dtype=float)
    required = {
        "wasted_vote_resistance",
        "major_party_gravity",
        "third_candidate_overexposure_risk",
        "candidate_conversion_confidence",
    }
    if frame.empty or not required.issubset(frame.columns):
        return np.asarray(pred, dtype=float)
    if os.getenv("POLL_PROJECT_DISABLE_CANDIDATE_CONVERSION_CONTEXT", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return np.asarray(pred, dtype=float)
    scale = selected_scale
    scale = min(max(scale, 0.0), 0.20)
    adjusted = np.asarray(pred, dtype=float).copy()
    def stature_component(column: str) -> pd.Series:
        source = frame.get(column, pd.Series(0.0, index=frame.index))
        return pd.to_numeric(source, errors="coerce").fillna(0.0).clip(0.0, 1.0)

    serious = stature_component("serious_contender_score")
    legitimacy = stature_component("legitimacy_score")
    organization = stature_component("organization_strength")
    alternative = stature_component("alternative_score")
    candidate_stature = (
        0.35 * serious
        + 0.30 * legitimacy
        + 0.25 * organization
        + 0.10 * alternative
    )
    stature_centered = candidate_stature - candidate_stature.groupby(
        frame["election_id"]
    ).transform("mean")
    resistance = pd.to_numeric(frame["wasted_vote_resistance"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    gravity = pd.to_numeric(frame["major_party_gravity"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    character_source = (
        frame["third_character_constraint"]
        if "third_character_constraint" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    character = pd.to_numeric(character_source, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    overexposure = pd.to_numeric(
        frame["third_candidate_overexposure_risk"],
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    confidence = pd.to_numeric(
        frame["candidate_conversion_confidence"],
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    third_slot = frame["slot"].astype(str).eq("C").astype(float)
    wasted_vote_penalty = third_slot * (
        (1.0 - resistance.to_numpy(float)) * gravity.to_numpy(float)
        + 0.65 * overexposure.to_numpy(float)
        + 0.35 * character.to_numpy(float)
    )
    effect = (
        0.55 * stature_centered.to_numpy(float)
        - 0.95 * wasted_vote_penalty
    ) * confidence.to_numpy(float)
    adjusted = adjusted + scale * effect
    return normalize_vote_share_predictions(frame, adjusted)


def apply_candidate_regionalism_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Redistribute candidate support toward documented personal bases."""

    values = np.asarray(pred, dtype=float)
    selected_scale = _rederived_float("regionalism_scale")
    if selected_scale <= 0.0:
        return values
    if frame.empty or "candidate_regionalism_signal" not in frame.columns:
        return values
    if os.getenv("POLL_PROJECT_DISABLE_CANDIDATE_REGIONALISM", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return values
    scale = selected_scale
    scale = min(max(scale, 0.0), 0.25)
    signal = pd.to_numeric(
        frame["candidate_regionalism_signal"], errors="coerce"
    ).fillna(0.0)
    return normalize_vote_share_predictions(frame, values + scale * signal.to_numpy(float))


def apply_within_bloc_regional_transfer_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Transfer regional vote share only between same-lane candidates."""

    selected_scale = _rederived_float("within_bloc_transfer_scale")
    if (
        selected_scale <= 0.0
        or frame.empty
        or "within_bloc_regional_transfer_signal" not in frame.columns
    ):
        return np.asarray(pred, dtype=float)
    if os.getenv("POLL_PROJECT_DISABLE_WITHIN_BLOC_TRANSFER", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return np.asarray(pred, dtype=float)
    scale = min(max(selected_scale, 0.0), 1.0)
    component_columns = {
        "within_bloc_base_transfer_signal": 1.0,
        "within_bloc_reservoir_transfer_signal": min(
            max(_rederived_float("within_bloc_reservoir_gain", 1.0), 0.0), 1.0
        ),
        "within_bloc_stronghold_transfer_signal": min(
            max(_rederived_float("within_bloc_stronghold_gain", 0.0), 0.0), 0.5
        ),
    }
    if set(component_columns).issubset(frame.columns):
        signal = pd.Series(0.0, index=frame.index, dtype=float)
        for column, gain in component_columns.items():
            signal += gain * pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    else:
        signal = pd.to_numeric(
            frame["within_bloc_regional_transfer_signal"], errors="coerce"
        ).fillna(0.0)
    return normalize_vote_share_predictions(
        frame,
        np.asarray(pred, dtype=float) + scale * signal.to_numpy(float),
    )


def apply_district_terrain_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Redistribute support using constituency-only regional history."""

    values = np.asarray(pred, dtype=float)
    selected_scale = _rederived_float("district_terrain_scale")
    if selected_scale <= 0.0 or frame.empty or "district_terrain_signal" not in frame.columns:
        return values
    if os.getenv("POLL_PROJECT_DISABLE_DISTRICT_TERRAIN", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return values
    scale = min(max(selected_scale, 0.0), 0.25)
    signal = pd.to_numeric(frame["district_terrain_signal"], errors="coerce").fillna(0.0)
    return normalize_vote_share_predictions(frame, values + scale * signal.to_numpy(float))


def apply_third_candidate_prediction_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Apply a bounded prior adjustment for viable third-candidate structure."""

    adjusted = np.asarray(pred, dtype=float).copy()
    if os.getenv("POLL_PROJECT_DISABLE_THIRD_CANDIDATE_ADJUSTMENT", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return adjusted
    if "third_candidate_structure" not in frame.columns:
        return adjusted
    scale = _numeric(os.getenv("POLL_PROJECT_THIRD_CANDIDATE_ADJUSTMENT_SCALE", "0.15"), 0.15)
    scale = min(max(scale, 0.0), 1.0)
    multiplier = np.ones(len(frame), dtype=float)
    gate_disabled = not _rederived_bool(
        "third_competitiveness_gate_enabled", False
    ) or os.getenv(
        "POLL_PROJECT_DISABLE_THIRD_COMPETITIVENESS_GATE", ""
    ).lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if not gate_disabled and "third_competitiveness_multiplier" in frame.columns:
        multiplier = (
            pd.to_numeric(frame["third_competitiveness_multiplier"], errors="coerce")
            .fillna(1.0)
            .clip(
                THIRD_COMPETITIVENESS_MULTIPLIER_FLOOR,
                THIRD_COMPETITIVENESS_MULTIPLIER_CAP,
            )
            .to_numpy(float)
        )
    character_disabled = not _rederived_bool(
        "third_character_multiplier_enabled", False
    ) or os.getenv(
        "POLL_PROJECT_DISABLE_THIRD_REGIME_CHARACTER", ""
    ).lower() in {"1", "true", "yes", "y"}
    if not character_disabled and "third_regime_character_multiplier" in frame.columns:
        multiplier = multiplier * (
            pd.to_numeric(frame["third_regime_character_multiplier"], errors="coerce")
            .fillna(1.0)
            .clip(0.85, 1.10)
            .to_numpy(float)
        )
    return adjusted + scale * frame["third_candidate_structure"].to_numpy(float) * multiplier


def apply_withdrawn_candidate_prediction_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Apply optional prediction adjustment for withdrawn-candidate transfers."""

    adjusted = np.asarray(pred, dtype=float).copy()
    if os.getenv("POLL_PROJECT_DISABLE_WITHDRAWN_CANDIDATE_ADJUSTMENT", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return adjusted
    if "withdrawn_candidate_transfer" not in frame.columns:
        return adjusted
    scale = _numeric(os.getenv("POLL_PROJECT_WITHDRAWN_CANDIDATE_ADJUSTMENT_SCALE", "0.15"), 0.15)
    scale = min(max(scale, 0.0), 1.0)
    return adjusted + scale * frame["withdrawn_candidate_transfer"].to_numpy(float)


def apply_region_residual_calibration(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_pred: np.ndarray | pd.Series,
    test_pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Apply shrunken residuals only when the prior-election history is sparse."""

    adjusted = np.asarray(test_pred, dtype=float).copy()
    if not _rederived_bool("residual_enabled", False):
        return adjusted
    required = {"election_id", "region_id", "slot", "vote_share"}
    if train.empty or test.empty or not required.issubset(train.columns) or not required.issubset(test.columns):
        return adjusted

    max_prior_elections = REGION_RESIDUAL_MAX_PRIOR_ELECTIONS
    prior_elections = train["election_id"].dropna().astype(str).nunique()
    if prior_elections == 0 or prior_elections > max_prior_elections:
        return adjusted

    scale = _rederived_float("residual_scale", 0.0)
    shrinkage = _rederived_float("residual_shrinkage", 8.0)
    scale = min(max(scale, 0.0), 1.0)
    shrinkage = max(shrinkage, 0.0)

    train_frame = train[["region_id", "slot"]].copy()
    train_frame["residual"] = normalized_vote_share_target(train) - normalize_vote_share_predictions(
        train,
        train_pred,
    )
    calibration = (
        train_frame.groupby(["region_id", "slot"], as_index=False)["residual"]
        .agg(["mean", "count"])
        .reset_index()
    )
    calibration["region_residual_adjustment"] = (
        calibration["mean"]
        * (calibration["count"] / (calibration["count"] + shrinkage))
        * scale
    )
    joined = test[["region_id", "slot"]].merge(
        calibration[["region_id", "slot", "region_residual_adjustment"]],
        on=["region_id", "slot"],
        how="left",
    )
    return adjusted + joined["region_residual_adjustment"].fillna(0.0).to_numpy(float)


def standardized_betas(df: pd.DataFrame, predictors: list[str], y_col: str) -> dict[str, float]:
    """Estimate standardized beta coefficients for quick variable comparison."""

    x_std = df[predictors].std(ddof=0).replace(0, np.nan)
    y_std = df[y_col].std(ddof=0)
    z = ((df[predictors] - df[predictors].mean()) / x_std).fillna(0.0)
    yz = (df[y_col] - df[y_col].mean()) / y_std if y_std else df[y_col] * 0.0
    beta, *_ = ols(z.to_numpy(float), yz.to_numpy(float))
    return {p: round(float(b), 3) for p, b in zip(predictors, beta[1:])}


def vif(df: pd.DataFrame, predictors: list[str]) -> dict[str, float]:
    """Compute simple variance inflation factors."""

    out: dict[str, float] = {}
    for predictor in predictors:
        others = [candidate for candidate in predictors if candidate != predictor]
        if not others:
            out[predictor] = 1.0
            continue
        _, r2, _, _ = ols(df[others].to_numpy(float), df[predictor].to_numpy(float))
        out[predictor] = round(1 / (1 - r2), 2) if r2 < 1 else float("inf")
    return out


def neutral_issue_context_scale() -> float:
    """Return the configured fixed neutral-context scale."""

    return max(_rederived_float("neutral_context_scale", 0.0), 0.0)


def apply_neutral_issue_context_adjustment(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
    scale: float | None = None,
) -> np.ndarray:
    """Apply the fixed neutral-context shadow scale used by the experiment."""

    values = np.asarray(pred, dtype=float)
    scale = neutral_issue_context_scale() if scale is None else max(float(scale), 0.0)
    if scale <= 0.0 or "assembly_neutral_issue_signal" not in frame.columns:
        return values
    signal = pd.to_numeric(
        frame["assembly_neutral_issue_signal"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    return normalize_vote_share_predictions(frame, values + scale * signal)


def apply_prediction_postprocess(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
    *,
    partisan_layer: bool = True,
    party_tone: bool = True,
    electorate_layer: bool = True,
) -> np.ndarray:
    """Apply the shared deterministic post-model sequence."""

    adjusted = np.asarray(pred, dtype=float)
    if partisan_layer:
        adjusted = apply_partisan_layer_prediction_moderation(frame, adjusted)
    adjusted = apply_party_context_prediction_adjustment(frame, adjusted)
    use_electorate_layer = electorate_layer and ELECTORATE_LAYER_ENABLED
    if party_tone and not use_electorate_layer:
        adjusted = apply_party_tone_gap_prediction_adjustment(frame, adjusted)
    adjusted = apply_same_orientation_external_adjustment(frame, adjusted)
    adjusted = apply_public_treatment_prediction_adjustment(frame, adjusted)
    adjusted = apply_generation_prediction_adjustment(frame, adjusted)
    adjusted = apply_candidate_conversion_context_adjustment(frame, adjusted)
    adjusted = apply_neutral_issue_context_adjustment(frame, adjusted)
    adjusted = apply_district_terrain_adjustment(frame, adjusted)
    adjusted = apply_candidate_regionalism_adjustment(frame, adjusted)
    adjusted = apply_within_bloc_regional_transfer_adjustment(frame, adjusted)
    if use_electorate_layer:
        adjusted, _ = apply_electorate_layer_response(
            frame,
            adjusted,
            ELECTORATE_LAYER_CONFIG,
        )
    return adjusted


def scored_contest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows included in the declared scored contest denominator."""

    if "is_scored_contest_row" not in frame.columns:
        return frame.copy()
    included = (
        frame["is_scored_contest_row"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )
    return frame.loc[included].copy()


def loeo_cv(
    df: pd.DataFrame,
    predictors: list[str],
    alpha: float = RIDGE_ALPHA,
    third_candidate_adjustment: bool = True,
) -> float:
    """Leave-one-election-out MAE in percentage points.

    This is useful as a stability diagnostic, but it is not a historical
    forecast backtest because later elections can train earlier held-out years.
    Use rolling_origin_cv for leakage-safe forecast evaluation.
    """

    errs: list[float] = []
    for election_id in df["election_id"].unique():
        train = df[df.election_id != election_id]
        test = scored_contest_rows(df[df.election_id == election_id])
        if len(train) < len(predictors) + 2 or test.empty:
            continue
        X_train = train[predictors].to_numpy(float)
        y_train = normalized_vote_share_target(train)
        X_test = test[predictors].to_numpy(float)
        if alpha > 0:
            beta, _, _, _, means, scales = ridge_fit(
                X_train,
                y_train,
                alpha=alpha,
                sample_weight=election_epoch_sample_weight(train),
            )
            train_pred = ridge_predict(beta, X_train, means, scales)
            pred = ridge_predict(beta, X_test, means, scales)
        else:
            beta, *_ = ols(X_train, y_train)
            train_design = np.column_stack([np.ones(len(train)), X_train])
            test_design = np.column_stack([np.ones(len(test)), X_test])
            train_pred = train_design @ beta
            pred = test_design @ beta
        if third_candidate_adjustment:
            train_pred = apply_third_candidate_prediction_adjustment(train, train_pred)
            train_pred = apply_withdrawn_candidate_prediction_adjustment(train, train_pred)
            pred = apply_third_candidate_prediction_adjustment(test, pred)
            pred = apply_withdrawn_candidate_prediction_adjustment(test, pred)
            pred = apply_region_residual_calibration(train, test, train_pred, pred)
        pred = normalize_vote_share_predictions(test, pred)
        if third_candidate_adjustment:
            pred = apply_prediction_postprocess(test, pred)
        errs.extend(np.abs(pred - normalized_vote_share_target(test)) * 100)
    return float(np.mean(errs)) if errs else float("nan")


def rolling_training_with_slot_backfill(
    train: pd.DataFrame,
    test: pd.DataFrame,
    warmup_ids: set[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Prefer scored history and backfill only target slots absent from it.

    The returned boolean mask identifies rows eligible for residual
    calibration. Warmup slot backfills help estimate an otherwise unseen slot
    but do not increase the apparent count of prior scored elections.
    """

    if train.empty or not warmup_ids:
        return train.copy(), np.ones(len(train), dtype=bool)
    is_warmup = train["election_id"].astype(str).isin(warmup_ids)
    real_train = train.loc[~is_warmup].copy()
    target_slots = set(test["slot"].astype(str))
    if real_train.empty:
        selected = train.loc[is_warmup & train["slot"].astype(str).isin(target_slots)].copy()
        if "_rolling_target" in selected.columns:
            selected["_rolling_target"] = normalized_vote_share_target(selected)
        return selected, np.ones(len(selected), dtype=bool)

    missing_slots = target_slots - set(real_train["slot"].astype(str))
    backfill = train.loc[
        is_warmup & train["slot"].astype(str).isin(missing_slots)
    ].copy()
    selected = pd.concat([real_train, backfill], ignore_index=True, sort=False)
    residual_mask = ~selected["election_id"].astype(str).isin(warmup_ids).to_numpy()
    return selected, residual_mask


def rolling_origin_cv(
    df: pd.DataFrame,
    predictors: list[str],
    alpha: float = RIDGE_ALPHA,
    election_order: list[str] = ORDER,
    third_candidate_adjustment: bool = True,
    warmup: pd.DataFrame | None = None,
    warmup_order: list[str] = ROLLING_WARMUP_ORDER,
) -> tuple[float, dict[str, float]]:
    """Chronological backtest using only prior presidential elections."""

    errs: list[float] = []
    by_election: dict[str, float] = {}
    full_order = [*warmup_order, *election_order] if warmup is not None and not warmup.empty else election_order
    order_lookup = {election_id: index for index, election_id in enumerate(full_order)}
    frame = df.copy()
    warmup_ids = set()
    if warmup is not None and not warmup.empty:
        warmup_ids = set(warmup["election_id"].astype(str))
        frame = pd.concat([warmup.copy(), frame], ignore_index=True, sort=False)
        for predictor in predictors:
            frame[predictor] = pd.to_numeric(frame[predictor], errors="coerce").fillna(0.0)
    frame = frame.copy()
    frame["_order"] = frame["election_id"].map(order_lookup)
    for election_id in election_order:
        target_order = order_lookup[election_id]
        test = scored_contest_rows(frame[frame["election_id"] == election_id])
        train = frame[frame["_order"] < target_order].copy()
        train["_rolling_target"] = normalized_vote_share_target(train)
        train, residual_mask = rolling_training_with_slot_backfill(
            train,
            test,
            warmup_ids,
        )
        if test.empty or train.empty:
            continue
        X_train = train[predictors].to_numpy(float)
        y_train = train["_rolling_target"].to_numpy(float)
        X_test = test[predictors].to_numpy(float)
        if alpha > 0:
            beta, _, _, _, means, scales = ridge_fit(
                X_train,
                y_train,
                alpha=alpha,
                sample_weight=election_epoch_sample_weight(train),
            )
            train_pred = ridge_predict(beta, X_train, means, scales)
            pred = ridge_predict(beta, X_test, means, scales)
        else:
            beta, *_ = ols(X_train, y_train)
            train_pred = np.column_stack([np.ones(len(train)), X_train]) @ beta
            pred = np.column_stack([np.ones(len(test)), X_test]) @ beta
        if third_candidate_adjustment:
            train_pred = apply_third_candidate_prediction_adjustment(train, train_pred)
            train_pred = apply_withdrawn_candidate_prediction_adjustment(train, train_pred)
            pred = apply_third_candidate_prediction_adjustment(test, pred)
            pred = apply_withdrawn_candidate_prediction_adjustment(test, pred)
            residual_train = train.loc[residual_mask].copy()
            residual_train_pred = train_pred[residual_mask]
            pred = apply_region_residual_calibration(
                residual_train,
                test,
                residual_train_pred,
                pred,
            )
        pred = normalize_vote_share_predictions(test, pred)
        if third_candidate_adjustment:
            pred = apply_prediction_postprocess(test, pred)
        election_errs = np.abs(pred - normalized_vote_share_target(test)) * 100
        by_election[election_id] = float(np.mean(election_errs))
        errs.extend(election_errs)
    frame.drop(columns=["_order"], inplace=True)
    return (float(np.mean(errs)) if errs else float("nan")), by_election


def _center_draws_by_group(
    draws: np.ndarray,
    frame: pd.DataFrame,
    group_columns: list[str],
) -> np.ndarray:
    """Center simulation noise within compositional vote-share groups."""

    centered = np.asarray(draws, dtype=float).copy()
    groups = frame.groupby(group_columns, sort=False).indices
    for indices in groups.values():
        index_array = np.fromiter(indices, dtype=int)
        if len(index_array) <= 1:
            continue
        centered[:, index_array] -= centered[:, index_array].mean(axis=1, keepdims=True)
    return centered


def _apply_monte_carlo_postprocess(
    frame: pd.DataFrame,
    pred: np.ndarray | pd.Series,
    *,
    electorate_layer: bool = True,
) -> np.ndarray:
    """Apply deterministic post-model adjustments used by Monte Carlo draws."""

    adjusted = apply_withdrawn_candidate_prediction_adjustment(
        frame,
        apply_third_candidate_prediction_adjustment(frame, pred),
    )
    # apply_prediction_postprocess starts with the partisan layer, so pass the
    # pre-partisan normalized values to keep every deterministic path identical.
    return apply_prediction_postprocess(
        frame,
        normalize_vote_share_predictions(frame, adjusted),
        electorate_layer=electorate_layer,
    )


def _weighted_group_average(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    """Return a robust weighted average for residual summaries."""

    value_array = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(float)
    weight_array = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
    total = float(weight_array.sum())
    if total <= 0.0:
        return float(value_array.mean()) if len(value_array) else 0.0
    return float(np.average(value_array, weights=weight_array))


def _draw_decomposed_residual_noise(
    rng: np.random.Generator,
    frame: pd.DataFrame,
    residuals: np.ndarray,
    n_sim: int,
    empirical: bool = True,
    include_local: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Draw residual noise as election-slot common shock plus local row noise."""

    residual_frame = frame[["election_id", "region_id", "slot"]].copy()
    residual_frame["_residual"] = np.asarray(residuals, dtype=float)
    if "votes" in frame.columns:
        residual_frame["_weight"] = (
            frame.groupby(["election_id", "region_id"])["votes"].transform("sum").to_numpy(float)
        )
    else:
        residual_frame["_weight"] = 1.0

    common = (
        residual_frame.groupby(["election_id", "slot"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "_common_residual": _weighted_group_average(
                        group["_residual"],
                        group["_weight"],
                    )
                }
            ),
            include_groups=False,
        )
    )
    residual_frame = residual_frame.merge(common, on=["election_id", "slot"], how="left")
    residual_frame["_common_residual"] = residual_frame["_common_residual"].fillna(0.0)
    residual_frame["_local_residual"] = residual_frame["_residual"] - residual_frame["_common_residual"]

    common_values = common["_common_residual"].to_numpy(float)
    local_values = residual_frame["_local_residual"].to_numpy(float)
    common_sigma = float(np.sqrt(np.mean(common_values**2))) if len(common_values) else 0.0
    local_sigma = float(np.sqrt(np.mean(local_values**2))) if len(local_values) else 0.0

    noise = np.zeros((n_sim, len(frame)), dtype=float)
    common_keys = common[["election_id", "slot"]].copy()
    common_keys["_draw_index"] = np.arange(len(common_keys))
    if empirical and len(common_values):
        centered_common_values = common_values - float(np.mean(common_values))
        common_draws = rng.choice(
            centered_common_values,
            size=(n_sim, len(common_keys)),
            replace=True,
        )
    else:
        common_draws = (
            rng.normal(0.0, common_sigma, size=(n_sim, len(common_keys)))
            if common_sigma > 0
            else np.zeros((n_sim, len(common_keys)), dtype=float)
        )
    for _, key_indices in common_keys.groupby("election_id").indices.items():
        index_array = np.fromiter(key_indices, dtype=int)
        if len(index_array) > 1:
            common_draws[:, index_array] -= common_draws[:, index_array].mean(axis=1, keepdims=True)
    row_draw_index = (
        frame[["election_id", "slot"]]
        .merge(common_keys, on=["election_id", "slot"], how="left")["_draw_index"]
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )
    valid = row_draw_index >= 0
    noise[:, valid] += common_draws[:, row_draw_index[valid]]

    if not include_local:
        return noise, common_sigma, local_sigma
    if empirical and len(local_values):
        centered_local_values = local_values - float(np.mean(local_values))
        local_noise = rng.choice(
            centered_local_values,
            size=noise.shape,
            replace=True,
        )
        local_noise = _center_draws_by_group(local_noise, frame, ["election_id", "region_id"])
        noise += local_noise
    elif local_sigma > 0:
        local_noise = rng.normal(0.0, local_sigma, size=noise.shape)
        local_noise = _center_draws_by_group(local_noise, frame, ["election_id", "region_id"])
        noise += local_noise
    return noise, common_sigma, local_sigma


def monte_carlo(
    df: pd.DataFrame,
    predictors: list[str],
    n_sim: int = 2000,
    seed: int = 0,
    alpha: float = RIDGE_ALPHA,
    include_residual_uncertainty: bool = True,
    residual_structure: str = "common_shock",
) -> pd.DataFrame:
    """Draw coefficient samples and produce calibrated prediction intervals."""

    rng = np.random.default_rng(seed)
    X = df[predictors].to_numpy(float)
    y = normalized_vote_share_target(df)
    sample_weight = election_epoch_sample_weight(df)
    beta, _, cov, resid, means, scales = ridge_fit(
        X,
        y,
        alpha=alpha,
        sample_weight=sample_weight,
    )
    sigma2 = float((sample_weight * resid**2).sum() / max(len(y) - (len(predictors) + 1), 1))
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    X_scaled = (X - means) / scales
    design = np.column_stack([np.ones(len(df)), X_scaled])
    sims = rng.multivariate_normal(beta, cov, size=n_sim)
    preds = sims @ design.T
    adjusted_preds = np.vstack(
        [
            apply_withdrawn_candidate_prediction_adjustment(
                df,
                apply_third_candidate_prediction_adjustment(df, pred),
            )
            for pred in preds
        ]
    )
    point_pred = _apply_monte_carlo_postprocess(df, design @ beta)
    neutral_context_scale = neutral_issue_context_scale()
    point_pred = apply_neutral_issue_context_adjustment(df, point_pred, neutral_context_scale)
    pre_layer_preds = np.vstack(
        [
            _apply_monte_carlo_postprocess(df, pred, electorate_layer=False)
            for pred in preds
        ]
    )
    if ELECTORATE_LAYER_ENABLED:
        pre_layer_preds = apply_electorate_layer_response_draws(
            df,
            pre_layer_preds,
            ELECTORATE_LAYER_CONFIG,
        )
    normalized_preds = np.vstack(
        [
            apply_neutral_issue_context_adjustment(df, pred, neutral_context_scale)
            for pred in pre_layer_preds
        ]
    )
    final_residual = y - point_pred
    interval_preds = normalized_preds
    common_sigma = 0.0
    local_sigma = 0.0
    if include_residual_uncertainty and sigma > 0.0:
        if residual_structure == "row_independent":
            residual_noise = rng.normal(0.0, sigma, size=normalized_preds.shape)
            residual_noise = _center_draws_by_group(
                residual_noise,
                df,
                ["election_id", "region_id"],
            )
            local_sigma = sigma
        elif residual_structure in {"common_shock", "decomposed", "normal_decomposed", "empirical_decomposed"}:
            residual_noise, common_sigma, local_sigma = _draw_decomposed_residual_noise(
                rng,
                df,
                final_residual,
                n_sim,
                empirical=residual_structure in {"common_shock", "empirical_decomposed"},
                include_local=residual_structure != "common_shock",
            )
        else:
            raise ValueError(
                "residual_structure must be 'common_shock', 'empirical_decomposed', 'normal_decomposed', 'decomposed', or 'row_independent'"
            )
        interval_preds = np.vstack(
            [
                normalize_vote_share_predictions(df, pred + noise)
                for pred, noise in zip(normalized_preds, residual_noise)
            ]
        )

    output_columns = ["election_id", "region_id", "slot", "bloc", "votes", "vote_share"]
    if "candidate_name" in df.columns:
        output_columns.insert(3, "candidate_name")
    out = df[output_columns].copy()
    active_slot_count = df.groupby(["election_id", "region_id"])["slot"].transform("nunique")
    out["contest_type"] = np.where(active_slot_count.to_numpy(int) == 2, "two_way", "three_way")
    out["contest_active_slots"] = active_slot_count.to_numpy(int)
    out["contest_votes"] = df.groupby(["election_id", "region_id"])["votes"].transform("sum").to_numpy(float)
    out["contest_vote_share"] = normalized_vote_share_target(df)
    out["vote_share_normalized"] = out["contest_vote_share"]
    out["pred_raw"] = adjusted_preds.mean(axis=0)
    out["pred"] = normalized_preds.mean(axis=0)
    out["prediction_residual_sigma"] = sigma
    out["prediction_residual_variance"] = sigma2
    out["prediction_residual_common_sigma"] = common_sigma
    out["prediction_residual_local_sigma"] = local_sigma
    out["prediction_residual_structure"] = residual_structure
    out["neutral_issue_context_scale"] = neutral_context_scale
    out["interval_includes_local_residual_uncertainty"] = bool(
        include_residual_uncertainty and residual_structure != "common_shock" and local_sigma > 0.0
    )
    out["interval_includes_residual_uncertainty"] = bool(include_residual_uncertainty and sigma > 0.0)
    out["mean_interval_includes_residual_uncertainty"] = False
    point_values = out["pred"].to_numpy(float)
    for level, lower_percentile, upper_percentile in (
        (90, 5.0, 95.0),
        (95, 2.5, 97.5),
        (99, 0.5, 99.5),
    ):
        mean_lower = np.percentile(normalized_preds, lower_percentile, axis=0)
        mean_upper = np.percentile(normalized_preds, upper_percentile, axis=0)
        out[f"mean_lo{level}"] = np.minimum(mean_lower, point_values)
        out[f"mean_hi{level}"] = np.maximum(mean_upper, point_values)
        lower = np.percentile(interval_preds, lower_percentile, axis=0)
        upper = np.percentile(interval_preds, upper_percentile, axis=0)
        out[f"lo{level}"] = np.minimum(lower, point_values)
        out[f"hi{level}"] = np.maximum(upper, point_values)
    for predictor in predictors:
        out[predictor] = df[predictor].to_numpy(float)
    diagnostic_columns: dict[str, np.ndarray] = {}
    for diagnostic in [
        "issue_signal_weight",
        "issue_advantage_raw",
        "issue_attention_score",
        "issue_local_attention_score",
        "support_conversion_score",
        "issue_support_signal",
        "issue_attention_overhang",
        "rif_raw",
        "same_bloc_issue_alignment",
        "same_bloc_frame_convergence",
        "cross_bloc_attack_pressure",
        "intra_bloc_conflict_score",
        "party_elite_support_score",
        "party_elite_fragmentation_score",
        "party_context_support",
        "party_context_confidence",
        "party_context_support_weighted",
        "party_context_support_weighted_centered",
        "organization_strength",
        "outsider_status",
        "same_party_positive_tone",
        "same_party_negative_tone",
        "same_party_net_tone",
        "same_party_supportive_tone",
        "cross_party_positive_tone",
        "cross_party_negative_tone",
        "cross_party_net_tone",
        "cross_party_adverse_tone",
        "party_tone_contrast",
        "same_party_supportive_tone_centered",
        "cross_party_adverse_tone_centered",
        "party_tone_contrast_centered",
        "manual_valence_coverage",
        "party_tone_confidence",
        "same_orientation_external_pressure",
        "same_orientation_party_weakness",
        "same_orientation_dispersion_risk",
        "same_orientation_anchor_pressure",
        "serious_contender_score",
        "legitimacy_score",
        "negative_treatment_score",
        "scandal_salience_score",
        "fatigue_score",
        "alternative_score",
        "protest_vote_score",
        "ridicule_or_gaffe_score",
        "public_treatment_support",
        "public_treatment_support_centered",
        "public_treatment_confidence",
        "candidate_weight",
        "coalition_cohesion",
        "coalition_mobilization_score",
        "wasted_vote_resistance",
        "major_party_gravity",
        "third_character_constraint",
        "third_candidate_overexposure_risk",
        "attention_to_support_gap",
        "conversion_capacity",
        "conversion_capacity_centered",
        "coalition_mobilization_centered",
        "candidate_conversion_confidence",
        "young_affinity",
        "middle_affinity",
        "senior_affinity",
        "generation_support_score",
        "generation_alignment",
        "generation_youth_niche",
        "generation_confidence",
        "assembly_neutral_issue_signal",
        "assembly_neutral_issue_confidence",
        "assembly_neutral_directional_evidence_count",
        "assembly_neutral_target_context_count",
        "assembly_neutral_global_context_count",
        "assembly_neutral_global_context_strength",
        "economic_context_effect",
        "economic_stress_index",
        "trade_context_effect",
        "trade_stress_index",
        "economic_responsibility_score",
        "real_gdp_growth_yoy",
        "current_account_12m_sum",
        "housing_price_index",
        "housing_price_yoy_change_pct",
        "housing_pressure_effect",
        "housing_cumulative_change_pct",
        "housing_responsibility_score",
        "housing_sgg_median_change_pct",
        "housing_sgg_dispersion",
        "housing_sgg_positive_share",
        "housing_sgg_count",
        "housing_pressure_intensity",
        "kospi_context_effect",
        "kospi_market_stress_index",
        "kospi_close",
        "kospi_return_3m",
        "kospi_return_12m",
        "kospi_drawdown_12m",
        "kospi_volatility_3m",
        "interest_rate_context_effect",
        "interest_rate_stress_index",
        "bok_base_rate",
        "bok_base_rate_12m_change",
        "growth_within_economy_weight",
        "trade_within_economy_weight",
        "kospi_within_economy_weight",
        "interest_rate_within_economy_weight",
        "economic_domain_effect",
        "economy_epoch_weight",
        "housing_epoch_weight",
        "third_candidate_structure",
        "third_attention_score",
        "third_conversion_capacity",
        "third_attention_overhang",
        "third_viability",
        "third_centrist_appeal",
        "third_anti_major_party_appeal",
        "third_regional_base_overlap",
        "third_profile_confidence",
        "third_competitiveness_gate",
        "third_competitiveness_multiplier",
        "third_regime_competitiveness",
        "third_regime_two_way_score",
        "third_regime_niche_minor_score",
        "third_regime_reform_minor_score",
        "third_regime_bloc_split_score",
        "third_regime_independent_pole_score",
        "third_regime_character_multiplier",
        "slotA_third_pressure",
        "slotB_third_pressure",
        "candidate_regional_affinity",
        "candidate_regional_organization",
        "candidate_regional_confidence",
        "candidate_regional_base_raw",
        "candidate_regional_base_gated",
        "candidate_regional_anchor_multiplier",
        "candidate_regional_character_factor",
        "candidate_regionalism_signal",
        "landscape_inferred_prior_evidence",
        "landscape_inferred_district_prior",
        "landscape_inferred_district_evidence",
        "district_terrain_raw",
        "district_terrain_reliability",
        "district_terrain_signal",
        "within_bloc_transfer_activation",
        "within_bloc_transfer_base_profile",
        "within_bloc_transfer_profile",
        "within_bloc_same_lane_reservoir",
        "within_bloc_reservoir_confirmation",
        "within_bloc_personal_stronghold",
        "within_bloc_stronghold_reinforcement",
        "within_bloc_base_transfer_signal",
        "within_bloc_reservoir_transfer_signal",
        "within_bloc_stronghold_transfer_signal",
        "within_bloc_regional_transfer_signal",
        "withdrawn_candidate_transfer",
        "withdrawn_candidate_viability",
        "withdrawn_transfer_rate",
        "withdrawn_voter_compliance",
        "withdrawn_transfer_confidence",
        "withdrawn_landscape_affinity",
        "partisan_prior_raw",
        "concrete_partisan_prior",
        "general_partisan_prior",
        "durable_core_raw",
        "recent_bloc_base",
        "critical_support_raw",
        "bloc_vote_volatility",
        "layer_effective_elections",
        "core_voting_mass",
        "critical_voting_mass",
        "swing_voting_mass",
        "nonvoter_reservoir",
        "bloc_loyalty",
        "bloc_strength",
        "effective_election_count",
        *LANDSCAPE_FEATURE_COLUMNS,
    ]:
        if diagnostic in df.columns and diagnostic not in out.columns:
            diagnostic_columns[diagnostic] = df[diagnostic].to_numpy(float)
    for diagnostic in df.columns:
        if (
            diagnostic.startswith("issue_pref_")
            or diagnostic.startswith("issue_attention_")
        ) and diagnostic not in out.columns:
            diagnostic_columns[diagnostic] = pd.to_numeric(
                df[diagnostic], errors="coerce"
            ).fillna(0.0).to_numpy(float)
    if "housing_price_period" in df.columns:
        diagnostic_columns["housing_price_period"] = df["housing_price_period"].astype(str).to_numpy()
    for diagnostic in ["housing_baseline_period", "housing_current_period"]:
        if diagnostic in df.columns:
            diagnostic_columns[diagnostic] = df[diagnostic].astype(str).to_numpy()
    if "kospi_latest_date" in df.columns:
        diagnostic_columns["kospi_latest_date"] = df["kospi_latest_date"].astype(str).to_numpy()
    if "third_regime_character" in df.columns:
        diagnostic_columns["third_regime_character"] = (
            df["third_regime_character"].astype(str).to_numpy()
        )
    if diagnostic_columns:
        out = pd.concat([out, pd.DataFrame(diagnostic_columns, index=out.index)], axis=1)
    return out


def main() -> None:
    """Run the engine and write the prediction interval table."""

    context_df = assemble()
    df = scored_contest_rows(context_df)
    print(
        f"[assemble] training-context rows={len(context_df)} | scored rows={len(df)} | "
        f"elections: {sorted(df.election_id.unique())}"
    )
    print(f"predictors: {PREDICTORS}\n")
    warmup = historical_presidential_warmup_frame()
    rolling_warmup = warmup.loc[warmup["election_id"].isin(ROLLING_WARMUP_ORDER)].copy()
    if not rolling_warmup.empty:
        print(
            f"[rolling warmup] {len(rolling_warmup)} rows | "
            f"elections: {sorted(rolling_warmup.election_id.unique())}\n"
        )

    y_model = normalized_vote_share_target(df)
    _, r2_base, _, _ = ols(df[["regional_base"]].to_numpy(float), y_model)
    beta_full, r2_full, _, _, means_full, scales_full = ridge_fit(
        df[PREDICTORS].to_numpy(float),
        y_model,
        alpha=RIDGE_ALPHA,
        sample_weight=election_epoch_sample_weight(df),
    )
    pred_eval = ridge_predict(beta_full, df[PREDICTORS].to_numpy(float), means_full, scales_full)
    pred_eval = apply_third_candidate_prediction_adjustment(df, pred_eval)
    pred_eval = apply_withdrawn_candidate_prediction_adjustment(df, pred_eval)
    pred_eval = normalize_vote_share_predictions(df, pred_eval)
    pred_eval = apply_prediction_postprocess(df, pred_eval)
    ss_res_eval = float(((y_model - pred_eval) ** 2).sum())
    ss_tot_eval = float(((y_model - y_model.mean()) ** 2).sum())
    r2_adjusted = 1 - ss_res_eval / ss_tot_eval if ss_tot_eval > 0 else float("nan")
    print("Ridge fit")
    print(f"  alpha                  = {RIDGE_ALPHA:.3f}")
    print(f"  M0 regional_base only R2 = {r2_base:.3f}")
    print(f"  M1 latent predictors R2 = {r2_full:.3f}")
    print(f"  M1 adjusted final R2    = {r2_adjusted:.3f}")
    print(f"  delta final R2          = {r2_adjusted - r2_base:+.3f}")
    print(f"  neutral context scale  = {neutral_issue_context_scale():.2f}")
    print("  standardized coefficients:")
    print(f"    const = {beta_full[0]:.6f}")
    for predictor, coefficient in zip(PREDICTORS, beta_full[1:]):
        print(f"    {predictor} = {coefficient:.6f}")
    print()

    print("standardized betas")
    beta_std_frame = df.copy()
    beta_std_frame["vote_share_normalized"] = y_model
    print(f"  {standardized_betas(beta_std_frame, PREDICTORS, 'vote_share_normalized')}\n")

    print("VIF")
    print(f"  {vif(df, PREDICTORS)}\n")

    print("leave-one-election-out CV")
    print(
        "  M0 %p MAE = "
        f"{loeo_cv(context_df, ['regional_base'], alpha=0.0, third_candidate_adjustment=False):.2f}"
    )
    print(
        "  M1 adjusted + regional calibration %p MAE = "
        f"{loeo_cv(context_df, PREDICTORS):.2f}\n"
    )

    rolling_mae, rolling_by_election = rolling_origin_cv(
        context_df,
        PREDICTORS,
        warmup=rolling_warmup,
    )
    print("rolling-origin CV")
    print(f"  M1 adjusted + regional calibration %p MAE = {rolling_mae:.2f}")
    for election_id, mae in rolling_by_election.items():
        print(f"    {election_id}: {mae:.2f}")
    print(
        "  region residual calibration "
        f"enabled={_rederived_bool('residual_enabled', False)} "
        f"scale={_rederived_float('residual_scale', 0.0):.2f} "
        f"shrinkage={_rederived_float('residual_shrinkage', 8.0):.2f} "
        f"max_prior_elections={REGION_RESIDUAL_MAX_PRIOR_ELECTIONS}"
    )
    print()

    mc = monte_carlo(df, PREDICTORS)
    print("Monte Carlo intervals")
    for level in (90, 95, 99):
        mean_width = (mc[f"mean_hi{level}"] - mc[f"mean_lo{level}"]).mean()
        predictive_width = (mc[f"hi{level}"] - mc[f"lo{level}"]).mean()
        print(
            f"  {level}% expected-share mean interval width = {mean_width * 100:.2f}%p"
        )
        print(
            f"  {level}% residual-inclusive predictive interval width = "
            f"{predictive_width * 100:.2f}%p"
        )

    out = "presidential_issue_engine/report/tables/issue_vote_engine.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    mc.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
