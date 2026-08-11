"""Re-derive formerly outcome-aware layer strengths using elections through 2022.

The feature formulas remain fixed. Layer activation and top-level strengths are
selected with nested rolling-origin evaluation. No later election row or metric
is loaded by this script.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "presidential_issue_engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import issue_vote_engine as engine  # noqa: E402
import robustness_check as robustness  # noqa: E402


ALLOWED_ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
WARMUP_ELECTIONS = ("pres_1997",)
RIDGE_ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20)
RESIDUAL_OPTIONS = (
    (False, 0.0, 8.0),
    (True, 0.5, 8.0),
    (True, 1.0, 4.0),
    (True, 1.0, 8.0),
    (True, 1.0, 16.0),
)
NEUTRAL_CONTEXT_SCALES = (0.0, 0.20, 0.40, 0.60, 0.80)
OVERLAY_GAINS = (0.0, 0.08, 0.12, 0.16, 0.24, 0.32)
CONVERSION_SCALES = (0.0, 0.01, 0.02, 0.035, 0.05)
DISTRICT_TERRAIN_SCALES = (0.0, 0.05, 0.10)
WITHIN_BLOC_TRANSFER_SCALES = (0.0, 0.20, 0.35, 0.50)
WITHIN_BLOC_STRONGHOLD_GAINS = (0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
WITHIN_BLOC_MIN_EFFECTIVE_ACTIVATION = 0.001
WITHIN_BLOC_MIN_EFFECTIVE_ELECTIONS = 2
REGIONAL_OPTIONS = (
    (0.0, 1.0),
    (0.05, 1.0),
    (0.05, 2.0),
    (0.05, 3.5),
    (0.10, 1.0),
    (0.10, 2.0),
    (0.10, 3.5),
    (0.15, 1.0),
    (0.15, 2.0),
    (0.15, 3.5),
)
THIRD_OPTIONS = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)
SOURCE_OVERLAY = ROOT / "data" / "raw" / "assembly_issue_character_overlay.csv"
OUTPUT_DIR = ROOT / "presidential_issue_engine" / "report" / "through2022_rederived"
VARIANT_DIR = OUTPUT_DIR / "overlay_variants"
CONFIG_OUTPUT = ROOT / "data" / "config" / "through2022_rederived_layers.json"


@dataclass(frozen=True)
class LayerConfig:
    ridge_alpha: float = 0.30
    residual_enabled: bool = False
    residual_scale: float = 0.0
    residual_shrinkage: float = 8.0
    neutral_context_scale: float = 0.0
    overlay_gain: float = 0.0
    conversion_scale: float = 0.0
    district_terrain_scale: float = 0.0
    within_bloc_transfer_scale: float = 0.0
    within_bloc_reservoir_gain: float = 1.0
    within_bloc_stronghold_gain: float = 0.0
    regionalism_scale: float = 0.0
    regional_anchor_strength: float = 1.0
    third_competitiveness_gate_enabled: bool = False
    third_character_multiplier_enabled: bool = False
    manual_issue_seed_enabled: bool = False
    automatic_issue_seed_enabled: bool = True

    @property
    def complexity(self) -> tuple[int, float]:
        active = sum(
            (
                self.overlay_gain > 0.0,
                self.conversion_scale > 0.0,
                self.district_terrain_scale > 0.0,
                self.within_bloc_transfer_scale > 0.0,
                self.regionalism_scale > 0.0,
                self.third_competitiveness_gate_enabled,
                self.third_character_multiplier_enabled,
                self.residual_enabled,
                self.neutral_context_scale > 0.0,
            )
        )
        magnitude = (
            self.overlay_gain
            + self.conversion_scale
            + self.district_terrain_scale
            + self.within_bloc_transfer_scale
            + self.within_bloc_stronghold_gain * 0.05
            + self.regionalism_scale
            + (0.025 if self.third_competitiveness_gate_enabled else 0.0)
            + (0.025 if self.third_character_multiplier_enabled else 0.0)
            + (0.05 if self.residual_enabled else 0.0)
            + self.neutral_context_scale * 0.05
        )
        return active, magnitude


NEUTRAL_CONFIG = LayerConfig()
ENV_KEYS = (
    "POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH",
)


def require_scope() -> None:
    if tuple(engine.ORDER) != ALLOWED_ELECTIONS:
        raise RuntimeError(f"Unexpected scored elections: {engine.ORDER}")
    if tuple(engine.ROLLING_WARMUP_ORDER) != WARMUP_ELECTIONS:
        raise RuntimeError(f"Unexpected warmup elections: {engine.ROLLING_WARMUP_ORDER}")
    source = pd.read_csv(SOURCE_OVERLAY, encoding="utf-8-sig")
    if not set(source["election_id"].astype(str)).issubset(ALLOWED_ELECTIONS):
        raise RuntimeError("Overlay source contains an out-of-scope election")


def write_overlay_variants() -> dict[float, Path | None]:
    source = pd.read_csv(SOURCE_OVERLAY, encoding="utf-8-sig")
    required = {"issue_confidence_quality", "character_score", "link_multiplier"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Overlay source missing columns: {sorted(missing)}")
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[float, Path | None] = {0.0: None}
    for gain in OVERLAY_GAINS[1:]:
        out = source.copy()
        raw = 1.0 + gain * out["issue_confidence_quality"] * out["character_score"]
        out["character_multiplier_raw"] = raw.clip(0.88, 1.24)
        election_mean = out.drop_duplicates(["election_id", "issue_name"]).groupby(
            "election_id"
        )["character_multiplier_raw"].mean()
        out["character_multiplier"] = (
            out["character_multiplier_raw"]
            / out["election_id"].map(election_mean).astype(float)
        ).clip(0.88, 1.24)
        out["salience_multiplier"] = out["character_multiplier"]
        out["character_gain"] = gain
        path = VARIANT_DIR / f"overlay_gain_{gain:.3f}.csv"
        out.to_csv(path, index=False, encoding="utf-8-sig")
        paths[gain] = path
    return paths


@contextmanager
def configured(config: LayerConfig, overlay_paths: dict[float, Path | None]):
    previous_env = {key: os.environ.get(key) for key in ENV_KEYS}
    previous_config = dict(engine.THROUGH_2022_REDERIVED_LAYER_CONFIG)
    try:
        engine.THROUGH_2022_REDERIVED_LAYER_CONFIG = asdict(config)
        overlay = overlay_paths[config.overlay_gain]
        os.environ["POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH"] = str(overlay) if overlay else "off"
        yield
    finally:
        engine.THROUGH_2022_REDERIVED_LAYER_CONFIG = previous_config
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_frames(
    overlay_paths: dict[float, Path | None],
) -> dict[float, tuple[pd.DataFrame, pd.DataFrame]]:
    frames: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for gain in OVERLAY_GAINS:
        config = replace(NEUTRAL_CONFIG, overlay_gain=gain)
        with configured(config, overlay_paths):
            all_rows = engine.assemble()
        scored = robustness.competition_frame(all_rows)
        warmup = robustness.rolling_warmup_frame(all_rows)
        frames[gain] = (scored, warmup)
    return frames


def improvement_threshold(n_elections: int) -> float:
    if n_elections <= 2:
        return 0.10
    if n_elections == 3:
        return 0.075
    return 0.05


def within_bloc_effective_elections(
    frame: pd.DataFrame,
    order: tuple[str, ...],
) -> tuple[str, ...]:
    """Return elections with enough ex-ante split activation to tune the layer."""

    if "within_bloc_transfer_activation" not in frame.columns:
        return ()
    maxima = (
        frame.loc[frame["election_id"].isin(order)]
        .groupby("election_id")["within_bloc_transfer_activation"]
        .max()
    )
    return tuple(
        election_id
        for election_id in order
        if float(maxima.get(election_id, 0.0)) >= WITHIN_BLOC_MIN_EFFECTIVE_ACTIVATION
    )


def candidates_for_step(config: LayerConfig, step: str) -> list[LayerConfig]:
    if step == "ridge_alpha":
        return [replace(config, ridge_alpha=value) for value in RIDGE_ALPHAS]
    if step == "residual":
        return [
            replace(
                config,
                residual_enabled=enabled,
                residual_scale=scale,
                residual_shrinkage=shrinkage,
            )
            for enabled, scale, shrinkage in RESIDUAL_OPTIONS
        ]
    if step == "neutral_context":
        return [
            replace(config, neutral_context_scale=value)
            for value in NEUTRAL_CONTEXT_SCALES
        ]
    if step == "overlay":
        return [replace(config, overlay_gain=value) for value in OVERLAY_GAINS]
    if step == "conversion":
        return [replace(config, conversion_scale=value) for value in CONVERSION_SCALES]
    if step == "district_terrain":
        return [
            replace(config, district_terrain_scale=value)
            for value in DISTRICT_TERRAIN_SCALES
        ]
    if step == "within_bloc_transfer":
        return [
            replace(
                config,
                within_bloc_transfer_scale=scale,
                within_bloc_stronghold_gain=(gain if scale > 0.0 else 0.0),
            )
            for scale in WITHIN_BLOC_TRANSFER_SCALES
            for gain in (
                WITHIN_BLOC_STRONGHOLD_GAINS if scale > 0.0 else (0.0,)
            )
        ]
    if step == "regionalism":
        return [
            replace(config, regionalism_scale=scale, regional_anchor_strength=anchor)
            for scale, anchor in REGIONAL_OPTIONS
        ]
    if step == "third":
        return [
            replace(
                config,
                third_competitiveness_gate_enabled=gate,
                third_character_multiplier_enabled=character,
            )
            for gate, character in THIRD_OPTIONS
        ]
    raise ValueError(step)


def main() -> None:
    require_scope()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_paths = write_overlay_variants()
    frames = build_frames(overlay_paths)
    cache: dict[tuple[LayerConfig, tuple[str, ...]], tuple[float, dict[str, float]]] = {}
    trace: list[dict[str, object]] = []

    def evaluate(config: LayerConfig, order: tuple[str, ...]) -> tuple[float, dict[str, float]]:
        key = (config, order)
        if key in cache:
            return cache[key]
        frame, warmup = frames[config.overlay_gain]
        subset = frame.loc[frame["election_id"].isin(order)].copy()
        with configured(config, overlay_paths):
            result = engine.rolling_origin_cv(
                subset,
                engine.PREDICTORS,
                alpha=config.ridge_alpha,
                election_order=list(order),
                warmup=warmup,
                warmup_order=list(WARMUP_ELECTIONS),
            )
        cache[key] = result
        return result

    def select(prior_order: tuple[str, ...], label: str) -> LayerConfig:
        if len(prior_order) < 2:
            return NEUTRAL_CONFIG
        current = NEUTRAL_CONFIG
        threshold = improvement_threshold(len(prior_order))
        for step in (
            "ridge_alpha",
            "residual",
            "neutral_context",
            "overlay",
            "conversion",
            "district_terrain",
            "regionalism",
            "third",
            "within_bloc_transfer",
        ):
            current_mae, _ = evaluate(current, prior_order)
            if step == "within_bloc_transfer":
                frame, _ = frames[current.overlay_gain]
                effective = within_bloc_effective_elections(frame, prior_order)
                if len(effective) < WITHIN_BLOC_MIN_EFFECTIVE_ELECTIONS:
                    current = replace(
                        current,
                        within_bloc_transfer_scale=0.0,
                        within_bloc_stronghold_gain=0.0,
                    )
                    trace.append(
                        {
                            "selection_label": label,
                            "tuning_elections": "|".join(prior_order),
                            "step": step,
                            **asdict(current),
                            "rolling_row_mae_pp": current_mae,
                            "current_before_step_mae_pp": current_mae,
                            "required_improvement_pp": threshold,
                            "selection_eligible": False,
                            "effective_activation_elections": "|".join(effective),
                        }
                    )
                    continue
            options = candidates_for_step(current, step)
            scored_options: list[tuple[float, tuple[int, float], LayerConfig]] = []
            for option in options:
                mae, _ = evaluate(option, prior_order)
                scored_options.append((mae, option.complexity, option))
                trace.append(
                    {
                        "selection_label": label,
                        "tuning_elections": "|".join(prior_order),
                        "step": step,
                        **asdict(option),
                        "rolling_row_mae_pp": mae,
                        "current_before_step_mae_pp": current_mae,
                        "required_improvement_pp": threshold,
                        "selection_eligible": True,
                        "effective_activation_elections": "",
                    }
                )
            best_mae, _, best = min(scored_options, key=lambda item: (item[0], item[1]))
            if np.isfinite(best_mae) and current_mae - best_mae >= threshold:
                current = best
        return current

    nested_rows: list[dict[str, object]] = []
    row_counts = {
        election_id: len(engine.scored_contest_rows(frames[0.0][0].loc[frames[0.0][0]["election_id"].eq(election_id)]))
        for election_id in ALLOWED_ELECTIONS
    }
    for index, target in enumerate(ALLOWED_ELECTIONS):
        prior_order = ALLOWED_ELECTIONS[:index]
        selected = select(prior_order, f"outer_{target}")
        evaluation_order = ALLOWED_ELECTIONS[: index + 1]
        _, by_election = evaluate(selected, evaluation_order)
        nested_rows.append(
            {
                "target_election": target,
                "tuning_elections": "|".join(prior_order),
                **asdict(selected),
                "outer_row_mae_pp": by_election[target],
                "outer_rows": row_counts[target],
            }
        )

    final_config = select(ALLOWED_ELECTIONS, "final_deployment")
    baseline_rolling, baseline_by = evaluate(NEUTRAL_CONFIG, ALLOWED_ELECTIONS)
    final_rolling, final_by = evaluate(final_config, ALLOWED_ELECTIONS)
    final_frame, _ = frames[final_config.overlay_gain]
    effective_transfer_elections = within_bloc_effective_elections(
        final_frame,
        ALLOWED_ELECTIONS,
    )
    transfer_deployment_eligible = (
        len(effective_transfer_elections) >= WITHIN_BLOC_MIN_EFFECTIVE_ELECTIONS
    )
    with configured(final_config, overlay_paths):
        final_loeo = engine.loeo_cv(
            final_frame, engine.PREDICTORS, alpha=final_config.ridge_alpha
        )
    baseline_frame, _ = frames[0.0]
    with configured(NEUTRAL_CONFIG, overlay_paths):
        baseline_loeo = engine.loeo_cv(
            baseline_frame, engine.PREDICTORS, alpha=NEUTRAL_CONFIG.ridge_alpha
        )

    transfer_experiment_rows: list[dict[str, object]] = []
    for scale in WITHIN_BLOC_TRANSFER_SCALES:
        gains = WITHIN_BLOC_STRONGHOLD_GAINS if scale > 0.0 else (0.0,)
        for stronghold_gain in gains:
            experimental = replace(
                final_config,
                within_bloc_transfer_scale=scale,
                within_bloc_stronghold_gain=stronghold_gain,
            )
            experimental_mae, experimental_by = evaluate(experimental, ALLOWED_ELECTIONS)
            transfer_experiment_rows.append(
                {
                    "within_bloc_transfer_scale": scale,
                    "within_bloc_stronghold_gain": stronghold_gain,
                    "selection_sample_rolling_row_mae_pp": experimental_mae,
                    **{
                        f"{election_id}_row_mae_pp": experimental_by[election_id]
                        for election_id in ALLOWED_ELECTIONS
                    },
                }
            )
    transfer_experiment = pd.DataFrame(transfer_experiment_rows)
    transfer_experiment.to_csv(
        OUTPUT_DIR / "within_bloc_transfer_experiment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_experimental_row = transfer_experiment.sort_values(
        [
            "selection_sample_rolling_row_mae_pp",
            "within_bloc_transfer_scale",
            "within_bloc_stronghold_gain",
        ]
    ).iloc[0]
    best_experimental_config = replace(
        final_config,
        within_bloc_transfer_scale=float(best_experimental_row["within_bloc_transfer_scale"]),
        within_bloc_stronghold_gain=float(best_experimental_row["within_bloc_stronghold_gain"]),
    )
    with configured(best_experimental_config, overlay_paths):
        best_experimental_loeo = engine.loeo_cv(
            final_frame,
            engine.PREDICTORS,
            alpha=best_experimental_config.ridge_alpha,
        )

    nested = pd.DataFrame(nested_rows)
    nested_mae = float(
        np.average(nested["outer_row_mae_pp"], weights=nested["outer_rows"])
    )
    nested.to_csv(OUTPUT_DIR / "nested_outer_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trace).to_csv(
        OUTPUT_DIR / "selection_trace.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "policy": {
            "allowed_elections": list(ALLOWED_ELECTIONS),
            "warmup_elections": list(WARMUP_ELECTIONS),
            "selection_method": "fixed-order coordinate search inside nested rolling origin",
            "selection_steps": [
                "ridge_alpha",
                "residual",
                "neutral_context",
                "overlay",
                "conversion",
                "district_terrain",
                "regionalism",
                "third",
                "within_bloc_transfer",
            ],
            "minimum_improvement_pp": {
                "2_elections": 0.10,
                "3_elections": 0.075,
                "4_or_more_elections": 0.05,
            },
            "within_bloc_transfer_guard": {
                "minimum_activation": WITHIN_BLOC_MIN_EFFECTIVE_ACTIVATION,
                "minimum_effective_elections": WITHIN_BLOC_MIN_EFFECTIVE_ELECTIONS,
                "observed_effective_elections": list(
                    effective_transfer_elections
                ),
            },
        },
        "neutral_baseline": {
            "config": asdict(NEUTRAL_CONFIG),
            "rolling_row_mae_pp": baseline_rolling,
            "rolling_by_election": baseline_by,
            "loeo_row_mae_pp": baseline_loeo,
        },
        "nested_evaluation": {
            "row_weighted_mae_pp": nested_mae,
            "outer_folds": nested_rows,
        },
        "final_deployment_selection": {
            "config": asdict(final_config),
            "selection_sample_rolling_row_mae_pp": final_rolling,
            "selection_sample_rolling_by_election": final_by,
            "selection_sample_loeo_row_mae_pp": final_loeo,
            "warning": "selection-sample metrics are not an external holdout",
        },
        "within_bloc_transfer_experiment": {
            "deployment_eligible": transfer_deployment_eligible,
            "reason": (
                "at least two pre-2022 elections have non-trivial split activation"
                if transfer_deployment_eligible
                else "fewer than two effective pre-2022 bloc-split elections"
            ),
            "best_selection_sample_scale": float(
                best_experimental_row["within_bloc_transfer_scale"]
            ),
            "best_selection_sample_stronghold_gain": float(
                best_experimental_row["within_bloc_stronghold_gain"]
            ),
            "best_selection_sample_rolling_row_mae_pp": float(
                best_experimental_row["selection_sample_rolling_row_mae_pp"]
            ),
            "best_selection_sample_loeo_row_mae_pp": best_experimental_loeo,
            "warning": "selection-sample diagnostic; nested outer performance remains primary",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CONFIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OUTPUT.write_text(
        json.dumps(
            {
                "provenance": "rederived only from rolling elections through 2022",
                "config": asdict(final_config),
                "registered_layers": {
                    "issue_character_overlay": {
                        "enabled": final_config.overlay_gain > 0.0,
                        "gain": final_config.overlay_gain,
                        "source": "data/raw/assembly_issue_character_overlay.csv",
                    },
                    "third_candidate_competitiveness_gate": {
                        "enabled": final_config.third_competitiveness_gate_enabled,
                        "source": "derived pre-election third-candidate profile",
                    },
                    "third_candidate_character_multiplier": {
                        "enabled": final_config.third_character_multiplier_enabled,
                        "source": "derived pre-election third-regime character",
                    },
                    "candidate_vote_conversion_context": {
                        "enabled": final_config.conversion_scale > 0.0,
                        "scale": final_config.conversion_scale,
                        "source": "data/raw/candidate_vote_conversion_context.csv",
                    },
                    "district_terrain": {
                        "enabled": final_config.district_terrain_scale > 0.0,
                        "scale": final_config.district_terrain_scale,
                        "source": "presidential_issue_engine/fixed_dataset/bloc_history_results.csv",
                        "source_types": [
                            "assembly_district",
                            "metro_council_district",
                            "local_council_district",
                            "metro_governor",
                            "local_governor",
                        ],
                        "candidate_mapping": "pre-election assembly political landscape",
                    },
                    "within_bloc_regional_transfer": {
                        "enabled": final_config.within_bloc_transfer_scale > 0.0,
                        "scale": final_config.within_bloc_transfer_scale,
                        "source": "pre-election bloc-split character plus candidate regional and district terrain",
                        "constraint": "region-zero-sum transfer among orientation-affine candidates",
                        "reservoir_gain": final_config.within_bloc_reservoir_gain,
                        "reservoir_source": "orientation-affine donor partisan prior with pre-election evidence reliability",
                        "stronghold_gain": final_config.within_bloc_stronghold_gain,
                        "stronghold_source": "documented pre-election candidate regional base",
                        "status": (
                            "selected from eligible through-2022 elections"
                            if transfer_deployment_eligible
                            else "insufficient effective elections for deployment selection"
                        ),
                        "minimum_effective_elections": WITHIN_BLOC_MIN_EFFECTIVE_ELECTIONS,
                        "observed_effective_elections": list(
                            effective_transfer_elections
                        ),
                    },
                    "manual_issue_seed": {
                        "enabled": final_config.manual_issue_seed_enabled,
                        "elections": list(ALLOWED_ELECTIONS),
                        "sources": [
                            "data/raw/candidate_issue_profile.csv",
                            "data/raw/mega_issue_axis.csv",
                            "data/raw/mega_issue_attribution.csv",
                        ],
                        "status": "disabled for forecasting",
                    },
                    "automatic_issue_seed": {
                        "enabled": final_config.automatic_issue_seed_enabled,
                        "elections": list(ALLOWED_ELECTIONS),
                        "builder": "scripts/build_through2022_automatic_issue_seeds.py",
                        "sources": [
                            "data/issue_salience_assembly.csv",
                            "data/candidate_issue_link.csv",
                            "data/raw/assembly_issue_character_overlay.csv",
                            "data/raw/candidate_public_treatment.csv",
                            "data/raw/candidate_party_tone_gap.csv",
                        ],
                        "outputs": [
                            "data/raw/auto_issue_seed/candidate_issue_profile.csv",
                            "data/raw/auto_issue_seed/mega_issue_axis.csv",
                            "data/raw/auto_issue_seed/mega_issue_attribution.csv",
                        ],
                        "outcome_fields_used": [],
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
