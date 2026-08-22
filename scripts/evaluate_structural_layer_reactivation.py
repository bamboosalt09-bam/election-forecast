"""Audit forecast-safe structural layers that are dormant in early outer folds.

This is a development comparison over the through-2022 presidential sample.
Every target prediction remains point-in-time safe, but the comparison table is
not an untouched holdout because the same five outcomes are used to decide
which bounded structural policy should be promoted.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import electorate_layers as layers  # noqa: E402
from presidential_issue_engine import mega_issue_adjustment  # noqa: E402
from scripts import evaluate_preliminary_slot_shadow_nested as nested  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "structural_layer_reactivation"
POLICY_PATH = ROOT / "data" / "config" / "active_presidential_model_v16.json"
PROFILE_PATH = ROOT / "data" / "raw" / "auto_issue_seed" / "candidate_issue_profile.csv"
INTENSITY_PATH = ROOT / "data" / "raw" / "mega_issue_intensity.csv"
ACTIVE_VARIANT = "slot_free_hierarchy_no_neutral"
EARLY_TARGETS = frozenset({"pres_2002", "pres_2007"})


@dataclass(frozen=True)
class Experiment:
    name: str
    preliminary_share_min1: bool = False
    config_overrides: tuple[tuple[str, object], ...] = ()
    override_scope: str = "all"
    terrain_anchor_gain: float = 0.0
    terrain_scope: str = "none"
    terrain_gain_by_target: tuple[tuple[str, float], ...] = ()
    preference_gain_floor: float = 0.0
    preference_scope: str = "none"
    layer_separation: float = 0.0
    direct_mega_minimum_intensity: float = 1.0

    @property
    def override_dict(self) -> dict[str, object]:
        return dict(self.config_overrides)


def _experiments() -> tuple[Experiment, ...]:
    rows: list[Experiment] = [
        Experiment("baseline"),
        Experiment("preliminary_share_min1", preliminary_share_min1=True),
    ]
    rows.extend(
        Experiment(
            f"candidate_conversion_{value:.3f}",
            config_overrides=(("conversion_scale", value),),
        )
        for value in (0.01, 0.02, 0.035, 0.05)
    )
    rows.extend(
        Experiment(
            f"candidate_regionalism_{value:.2f}",
            config_overrides=(("regionalism_scale", value),),
        )
        for value in (0.05, 0.10, 0.15)
    )
    rows.extend(
        Experiment(
            f"within_bloc_{value:.2f}",
            config_overrides=(
                ("within_bloc_transfer_scale", value),
                ("within_bloc_stronghold_gain", 0.25 if value else 0.0),
            ),
        )
        for value in (0.20, 0.35, 0.50)
    )
    rows.extend(
        [
            Experiment(
                "third_gate_only",
                config_overrides=(("third_competitiveness_gate_enabled", True),),
            ),
            Experiment(
                "third_character_only",
                config_overrides=(("third_character_multiplier_enabled", True),),
            ),
            Experiment(
                "third_gate_and_character",
                config_overrides=(
                    ("third_competitiveness_gate_enabled", True),
                    ("third_character_multiplier_enabled", True),
                ),
            ),
        ]
    )
    rows.extend(
        Experiment(
            f"terrain_anchor_early_{value:.2f}",
            terrain_anchor_gain=value,
            terrain_scope="early",
        )
        for value in (0.10, 0.25, 0.50)
    )
    rows.extend(
        Experiment(
            f"terrain_anchor_all_{value:.2f}",
            terrain_anchor_gain=value,
            terrain_scope="all",
        )
        for value in (0.10, 0.25, 0.50)
    )
    rows.extend(
        Experiment(
            f"preference_floor_early_{value:.3f}",
            preference_gain_floor=value,
            preference_scope="early",
        )
        for value in (0.01, 0.02, 0.04)
    )
    rows.extend(
        [
            Experiment("direct_mega_intensity_ge_1", direct_mega_minimum_intensity=0.999999),
            Experiment("direct_mega_intensity_ge_0_5", direct_mega_minimum_intensity=0.499999),
            Experiment(
                "structural_bundle",
                config_overrides=(
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                    ("third_competitiveness_gate_enabled", True),
                    ("third_character_multiplier_enabled", True),
                ),
                terrain_anchor_gain=0.25,
                terrain_scope="early",
            ),
            Experiment(
                "structural_bundle_with_preliminary",
                preliminary_share_min1=True,
                config_overrides=(
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                    ("third_competitiveness_gate_enabled", True),
                    ("third_character_multiplier_enabled", True),
                ),
                terrain_anchor_gain=0.25,
                terrain_scope="early",
            ),
            Experiment(
                "accepted_layers_without_terrain",
                config_overrides=(
                    ("conversion_scale", 0.05),
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                ),
                preference_gain_floor=0.04,
                preference_scope="all",
            ),
            Experiment(
                "accepted_layers_terrain_early_0_25",
                config_overrides=(
                    ("conversion_scale", 0.05),
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                ),
                terrain_anchor_gain=0.25,
                terrain_scope="early",
                preference_gain_floor=0.04,
                preference_scope="all",
            ),
            Experiment(
                "accepted_layers_terrain_all_0_10",
                config_overrides=(
                    ("conversion_scale", 0.05),
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                ),
                terrain_anchor_gain=0.10,
                terrain_scope="all",
                preference_gain_floor=0.04,
                preference_scope="all",
            ),
            Experiment(
                "accepted_layers_terrain_all_0_25",
                config_overrides=(
                    ("conversion_scale", 0.05),
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                ),
                terrain_anchor_gain=0.25,
                terrain_scope="all",
                preference_gain_floor=0.04,
                preference_scope="all",
            ),
            Experiment(
                "accepted_layers_evidence_shock_terrain",
                config_overrides=(
                    ("conversion_scale", 0.05),
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                ),
                terrain_gain_by_target=(
                    ("pres_2002", 0.1666666666666667),
                    ("pres_2007", 0.25),
                    ("pres_2012", 0.25),
                    ("pres_2017", 0.125),
                    ("pres_2022", 0.25),
                ),
                preference_gain_floor=0.04,
                preference_scope="all",
            ),
            Experiment(
                "accepted_layers_evidence_shock_separation_0_25",
                config_overrides=(
                    ("conversion_scale", 0.05),
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                ),
                terrain_gain_by_target=(
                    ("pres_2002", 0.1666666666666667),
                    ("pres_2007", 0.25),
                    ("pres_2012", 0.25),
                    ("pres_2017", 0.125),
                    ("pres_2022", 0.25),
                ),
                preference_gain_floor=0.04,
                preference_scope="all",
                layer_separation=0.25,
            ),
            Experiment(
                "accepted_layers_evidence_shock_separation_0_50",
                config_overrides=(
                    ("conversion_scale", 0.05),
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                ),
                terrain_gain_by_target=(
                    ("pres_2002", 0.1666666666666667),
                    ("pres_2007", 0.25),
                    ("pres_2012", 0.25),
                    ("pres_2017", 0.125),
                    ("pres_2022", 0.25),
                ),
                preference_gain_floor=0.04,
                preference_scope="all",
                layer_separation=0.50,
            ),
            Experiment(
                "accepted_layers_evidence_shock_separation_1_00",
                config_overrides=(
                    ("conversion_scale", 0.05),
                    ("regionalism_scale", 0.15),
                    ("within_bloc_transfer_scale", 0.50),
                    ("within_bloc_stronghold_gain", 0.25),
                ),
                terrain_gain_by_target=(
                    ("pres_2002", 0.1666666666666667),
                    ("pres_2007", 0.25),
                    ("pres_2012", 0.25),
                    ("pres_2017", 0.125),
                    ("pres_2022", 0.25),
                ),
                preference_gain_floor=0.04,
                preference_scope="all",
                layer_separation=1.0,
            ),
        ]
    )
    return tuple(rows)


def _scope_applies(scope: str, target: str) -> bool:
    if scope == "all":
        return True
    if scope == "early":
        return target in EARLY_TARGETS
    if scope == "none":
        return False
    raise ValueError(f"unknown experiment scope: {scope}")


def _outer_predictions(full: pd.DataFrame, experiment: Experiment) -> pd.DataFrame:
    original_config_loader = nested.base_eval._layer_config_from_row
    original_predictor_selector = nested._predictors_for_fold

    def config_loader(row: pd.Series):
        config = original_config_loader(row)
        target = str(row["target_election"])
        if experiment.override_dict and _scope_applies(experiment.override_scope, target):
            config = replace(config, **experiment.override_dict)
        return config

    def predictor_selector(
        frame: pd.DataFrame,
        variant: str,
        target_index: int,
        target: str,
    ) -> tuple[str, ...]:
        if experiment.preliminary_share_min1 and target_index >= 1:
            return tuple(nested.SHARE_PRELIMINARY)
        return original_predictor_selector(frame, variant, target_index, target)

    nested.base_eval._layer_config_from_row = config_loader
    nested._predictors_for_fold = predictor_selector
    try:
        outer, _ = nested._build_outer_predictions(full, ACTIVE_VARIANT)
    finally:
        nested.base_eval._layer_config_from_row = original_config_loader
        nested._predictors_for_fold = original_predictor_selector
    return outer


def _electorate_predictions(
    layered: pd.DataFrame,
    experiment: Experiment,
) -> pd.DataFrame:
    _, selected = nested._apply_nested_preference(layered, ACTIVE_VARIANT)
    gains = {str(row["target_election"]): float(row["preference_gain"]) for row in selected}
    parts: list[pd.DataFrame] = []
    for target in nested.ELECTIONS:
        target_frame = layered.loc[layered["election_id"].eq(target)].copy()
        preference_gain = gains[target]
        if _scope_applies(experiment.preference_scope, target):
            preference_gain = max(preference_gain, experiment.preference_gain_floor)
        target_terrain = dict(experiment.terrain_gain_by_target)
        terrain_gain = target_terrain.get(
            target,
            experiment.terrain_anchor_gain
            if _scope_applies(experiment.terrain_scope, target)
            else 0.0,
        )
        parts.append(
            nested.base_eval.apply_config(
                target_frame,
                layers.ElectorateLayerConfig(
                    terrain_anchor_gain=terrain_gain,
                    preference_gain=preference_gain,
                    layer_separation=experiment.layer_separation,
                    mass_profile="direct_party_layers",
                ),
            )
        )
    return pd.concat(parts, ignore_index=True)


def _apply_direct_mega(predictions: pd.DataFrame, experiment: Experiment) -> pd.DataFrame:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    postprocess: Mapping[str, object] = policy["postprocess"]
    scores = mega_issue_adjustment.compile_direct_mega_scores(
        pd.read_csv(PROFILE_PATH, encoding="utf-8-sig"),
        pd.read_csv(INTENSITY_PATH, encoding="utf-8-sig"),
        nested.engine.ELECTION_DATES,
        minimum_intensity=experiment.direct_mega_minimum_intensity,
        score_cap=float(postprocess["direct_mega_score_cap"]),
    )
    return mega_issue_adjustment.apply_direct_mega_shift(
        predictions,
        scores,
        prediction_column="layer_pred",
        gain=float(postprocess["direct_mega_logit_gain"]),
        log_shift_cap=float(postprocess["direct_mega_log_shift_cap"]),
    )


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = nested._prepare_rows()
    base = nested._base_layer_frame(require_frozen_reproduction=False)
    summary_rows: list[dict[str, object]] = []
    election_rows: list[pd.DataFrame] = []
    national_rows: list[pd.DataFrame] = []
    outer_cache: dict[tuple[bool, tuple[tuple[str, object], ...], str], pd.DataFrame] = {}

    for experiment in _experiments():
        outer_key = (
            experiment.preliminary_share_min1,
            experiment.config_overrides,
            experiment.override_scope,
        )
        if outer_key not in outer_cache:
            outer_cache[outer_key] = _outer_predictions(full, experiment)
        layered = nested._attach_layers(base, outer_cache[outer_key])
        predictions = _electorate_predictions(layered, experiment)
        predictions = _apply_direct_mega(predictions, experiment)
        summary, by_election, national = nested._metrics(
            predictions,
            "layer_pred",
            experiment.name,
        )
        summary_rows.append(summary)
        election_rows.append(by_election)
        national_rows.append(national)

    summary = pd.DataFrame(summary_rows)
    baseline = summary.loc[summary["variant"].eq("baseline")].iloc[0]
    summary["regional_change_vs_baseline_pp"] = (
        summary["regional_equal_election_macro_mae_pp"]
        - float(baseline["regional_equal_election_macro_mae_pp"])
    )
    summary["national_change_vs_baseline_pp"] = (
        summary["national_equal_election_macro_mae_pp"]
        - float(baseline["national_equal_election_macro_mae_pp"])
    )
    by_election = pd.concat(election_rows, ignore_index=True)
    national = pd.concat(national_rows, ignore_index=True)
    return summary, by_election, national


def main() -> None:
    summary, by_election, national = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    national.to_csv(OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig")
    print(
        summary.sort_values("national_equal_election_macro_mae_pp").to_string(index=False)
    )


if __name__ == "__main__":
    main()
