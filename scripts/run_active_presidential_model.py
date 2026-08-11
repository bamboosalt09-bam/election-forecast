"""Run the promoted outcome-blind strict nested presidential model."""

from __future__ import annotations

import json
import hashlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import build_preliminary_slot_assignments as assignment_builder  # noqa: E402
from scripts import build_through2022_automatic_issue_seeds as issue_seed_builder  # noqa: E402
from scripts import evaluate_preliminary_slot_shadow_nested as nested  # noqa: E402
from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine import chungcheong_identity  # noqa: E402
from presidential_issue_engine import direct_party_center  # noqa: E402
from presidential_issue_engine import incumbent_shock_adjustment  # noqa: E402
from presidential_issue_engine import mega_issue_adjustment  # noqa: E402
from presidential_issue_engine import regional_identity  # noqa: E402
from presidential_issue_engine import regional_swing_elasticity  # noqa: E402
from presidential_issue_engine import fully_nested_policy  # noqa: E402
from presidential_issue_engine import strategic_lane_transfer  # noqa: E402
from presidential_issue_engine import rejection_beneficiary_routing  # noqa: E402
from presidential_issue_engine.point_in_time import filter_available_by_election  # noqa: E402


CONFIG_PATH = ROOT / "data" / "config" / "active_presidential_model.json"
OUTPUT_DIR = ROOT / "outputs" / "active_presidential_nested_v16"
ASSIGNMENT_DIR = ROOT / "outputs" / "preliminary_slot_assignment"
EXPECTED_VARIANT = "slot_free_hierarchy_no_neutral"
CANDIDATE_ISSUE_PROFILE = ROOT / "data" / "raw" / "auto_issue_seed" / "candidate_issue_profile.csv"
MEGA_ISSUE_INTENSITY = ROOT / "data" / "raw" / "mega_issue_intensity.csv"
CONVERSION_CONTEXT = ROOT / "data" / "raw" / "candidate_vote_conversion_context.csv"
CHUNGCHEONG_ALIGNMENT = ROOT / "data" / "raw" / "chungcheong_identity_alignment.csv"
CANDIDATE_REGIONAL_BASE = ROOT / "data" / "raw" / "candidate_regional_base.csv"


@contextmanager
def strict_input_policy() -> Iterator[None]:
    """Disable undated curated weights for the whole active computation."""

    key = nested.engine.STRICT_UNDATED_CURATED_INPUTS_ENV
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def track_csv_inputs() -> Iterator[dict[str, dict[str, object]]]:
    """Collect every local CSV read by pandas during one active run."""

    records: dict[str, dict[str, object]] = {}
    original = pd.read_csv

    def tracked(source, *args, **kwargs):
        if isinstance(source, (str, os.PathLike)):
            candidate = Path(source)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if candidate.exists():
                resolved = candidate.resolve()
                payload = resolved.read_bytes()
                try:
                    display = str(resolved.relative_to(ROOT))
                except ValueError:
                    if resolved.parent.name.startswith("slot_assignment_"):
                        display = (
                            "generated:preliminary_slot_assignment/"
                            f"{resolved.name}"
                        )
                    else:
                        display = str(resolved)
                records[str(resolved)] = {
                    "path": display.replace("\\", "/"),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
        return original(source, *args, **kwargs)

    pd.read_csv = tracked
    try:
        yield records
    finally:
        pd.read_csv = original


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_policy(policy: dict[str, object]) -> dict[str, object]:
    if not policy.get("active"):
        raise RuntimeError("active presidential policy is disabled")
    if policy.get("variant") != EXPECTED_VARIANT:
        raise RuntimeError(f"unsupported active variant: {policy.get('variant')}")
    predictors = tuple(policy.get("predictors", []))
    if predictors != nested.BASE_PREDICTORS:
        raise RuntimeError("active predictor set drifted from validated slot-free predictors")
    forbidden = set(policy.get("forbidden_predictors", []))
    if not nested.OLD_SLOT_PREDICTORS.issubset(forbidden):
        raise RuntimeError("active policy does not forbid every realized-slot predictor")
    postprocess = policy.get("postprocess", {})
    if postprocess.get("neutral_context_direct_adjustment") is not False:
        raise RuntimeError("active policy must disable signed neutral-context adjustment")
    if postprocess.get("withdrawn_candidate_transfer_adjustment") is not True:
        raise RuntimeError("active policy must retain final withdrawn-candidate transfer")
    if postprocess.get("direct_mega_issue_shift") is not True:
        raise RuntimeError("active policy must retain the bounded direct mega-issue shift")
    if postprocess.get("incumbent_shock_response") is not True:
        raise RuntimeError("active policy must retain the incumbent-shock response")
    if (
        float(postprocess.get("government_burden_gain", -1.0)) != 1.0
        or float(postprocess.get("rupture_extra_gain", -1.0)) != 0.40
        or float(postprocess.get("incumbent_shock_log_shift_cap", -1.0)) != 0.15
    ):
        raise RuntimeError("active incumbent-shock response parameters drifted")
    if postprocess.get("contest_regime_response") is not True:
        raise RuntimeError("active policy must retain the contest-regime response")
    if postprocess.get("cumulative_regime_rejection") is not True:
        raise RuntimeError("active policy must retain cumulative regime rejection")
    if (
        float(postprocess.get("contest_regime_expansion_gain", -1.0)) != 0.50
        or float(postprocess.get("contest_regime_log_shift_cap", -1.0)) != 0.40
        or float(postprocess.get("contest_regime_critical_elasticity", -1.0)) != 0.75
        or float(postprocess.get("contest_regime_swing_elasticity", -1.0)) != 1.25
        or float(postprocess.get("contest_regime_swing_log_shift_cap", -1.0)) != 0.50
        or postprocess.get("contest_regime_rejection_double_discount") is not False
        or int(postprocess.get("cumulative_rejection_breadth_reference", -1)) != 4
        or float(postprocess.get("cumulative_rejection_party_erosion_width", -1.0)) != 0.08
        or float(postprocess.get("cumulative_rejection_conversion_buffer", -1.0)) != 0.15
        or float(postprocess.get("cumulative_rejection_rupture_score_reference", -1.0)) != 0.25
    ):
        raise RuntimeError("active contest-regime response parameters drifted")
    structural = policy.get("structural_layers", {})
    required_overrides = {
        "conversion_scale": 0.05,
        "regionalism_scale": 0.15,
        "within_bloc_transfer_scale": 0.50,
        "within_bloc_stronghold_gain": 0.25,
    }
    if structural.get("outer_config_overrides") != required_overrides:
        raise RuntimeError("active policy structural layer overrides drifted")
    electorate = structural.get("electorate_response", {})
    if float(electorate.get("preference_gain_floor", -1.0)) != 0.04:
        raise RuntimeError("active policy preference gain floor drifted")
    terrain = electorate.get("terrain_anchor", {})
    if (
        float(terrain.get("reliability_multiplier", -1.0)) != 0.50
        or float(terrain.get("gain_cap", -1.0)) != 0.25
        or terrain.get("mega_shock_attenuation") is not True
    ):
        raise RuntimeError("active policy terrain anchor formula drifted")
    regional_accent = electorate.get("regional_accent", {})
    if (
        regional_accent.get("source")
        != "strictly_prior_direct_party_multiaxis_history"
        or float(regional_accent.get("reliability_multiplier", -1.0)) != 0.30
        or float(regional_accent.get("gain_cap", -1.0)) != 0.20
        or float(regional_accent.get("signal_width", -1.0)) != 0.10
        or regional_accent.get("core_policy") != "inverse_core_share_mobility"
        or regional_accent.get("mega_shock_attenuation") is not False
    ):
        raise RuntimeError("active regional-accent policy drifted")
    regional_offset = electorate.get("regional_swing_offset", {})
    if (
        regional_offset.get("enabled") is not True
        or regional_offset.get("source")
        != "rolling_nonpresidential_direct_party_ballots"
        or regional_offset.get("method") != "hierarchical_logit_offset"
        or float(regional_offset.get("base_gain", -1.0)) != 0.25
        or float(regional_offset.get("prior_strength", -1.0)) != 2.0
        or int(regional_offset.get("minimum_prior_scored_elections", -1)) != 2
        or float(regional_offset.get("vif_threshold", -1.0)) != 20.0
        or regional_offset.get("activation_gate")
        != "minimum_two_scored_elections_and_max_finite_vif_gt_20"
        or regional_offset.get("third_candidate_mass_preserved") is not True
        or regional_offset.get("outcome_fields_used") != []
    ):
        raise RuntimeError("active regional-swing offset policy drifted")
    identity = structural.get("chungcheong_regional_identity", {})
    identity_source = identity.get("reservoir_source")
    identity_evidence = identity.get("routing_evidence")
    legacy_identity_schema = (
        identity_source == "strictly_prior_regional_third_bloc_excess"
        and identity_evidence
        == ["candidate_regional_base", "dated_pre_election_alignment"]
    )
    automatic_identity_schema = (
        identity_source == "strictly_prior_full_election_history"
        and identity_evidence
        == [
            "footprint_candidate_regional_base",
            "dated_pre_election_alignment",
            "automatic_regional_party_candidate_fit",
        ]
        and identity.get("automatic_alignment_schema")
        == "automatic_regional_party_alignment_v11"
    )
    if (
        identity.get("enabled") is not True
        or not (legacy_identity_schema or automatic_identity_schema)
        or float(identity.get("gain", -1.0)) != 0.50
        or float(identity.get("regional_shift_cap", -1.0)) != 0.08
        or float(identity.get("half_life_years", -1.0)) != 12.0
        or float(identity.get("prior_strength", -1.0)) != 1.5
        or identity.get("unrouted_mass_policy") != "remain_critical_or_swing"
        or identity.get("outcome_fields_used") != []
    ):
        raise RuntimeError("active Chungcheong regional-identity policy drifted")
    general_identity = structural.get("general_regional_identity", {})
    if (
        general_identity.get("enabled") is not True
        or general_identity.get("region_scope")
        != "non_chungcheong_with_dated_candidate_base_only"
        or general_identity.get("distinctiveness_source")
        != "strictly_prior_direct_party_and_downweighted_presidential_ballots"
        or general_identity.get("routing_evidence") != ["candidate_regional_base"]
        or general_identity.get("donor_policy")
        != "least_compatible_prior_regional_camp_first"
        or float(general_identity.get("gain", -1.0)) != 0.10
        or float(general_identity.get("regional_shift_cap", -1.0)) != 0.04
        or float(general_identity.get("half_life_years", -1.0)) != 12.0
        or float(general_identity.get("prior_strength", -1.0)) != 1.5
        or general_identity.get("outcome_fields_used") != []
    ):
        raise RuntimeError("active general regional-identity policy drifted")
    concrete = electorate.get("concrete_support", {})
    if (
        concrete.get("eligible_lineages") != ["국민의힘", "더불어민주당"]
        or concrete.get("matching")
        != "exact_pre_normalization_party_lineage"
        or float(concrete.get("other_lineage_core", -1.0)) != 0.0
        or concrete.get("other_stable_support_reclassification")
        != "critical_support"
        or concrete.get("cross_candidate_core_sharing") is not False
    ):
        raise RuntimeError("active major-party concrete-support policy drifted")
    strategic_transfer = structural.get("strategic_lane_transfer", {})
    if (
        strategic_transfer.get("reservoir") != "nonmajor_effective_critical_support"
        or strategic_transfer.get("recipient_pool") != "aligned_major_party_candidates"
        or strategic_transfer.get("viability_source") != "preliminary_expected_share"
        or strategic_transfer.get("outcome_fields_used") != []
        or float(strategic_transfer.get("affinity_power", -1.0)) != 2.0
    ):
        raise RuntimeError("active strategic-lane transfer policy drifted")
    party_context = structural.get("party_context_cohesion", {})
    if (
        party_context.get("mode") != "supporter_retention"
        or party_context.get("direct_vote_adjustment") is not False
        or float(party_context.get("core_defection_cap", -1.0))
        != nested.engine.PARTY_CONTEXT_CORE_DEFECTION_CAP
        or float(party_context.get("critical_defection_cap", -1.0))
        != nested.engine.PARTY_CONTEXT_CRITICAL_DEFECTION_CAP
        or party_context.get("released_mass_allocation")
        != "regional_pre_adjustment_prediction"
        or party_context.get("candidate_conversion_direct_input")
        != "nonparty_candidate_stature"
        or party_context.get("same_orientation_use")
        != "within_bloc_dispersion_only"
    ):
        raise RuntimeError("active party-context cohesion policy drifted")
    selection = policy.get("strict_nested_selection", {})
    if selection.get("enabled") is not True:
        raise RuntimeError("active policy must enable fully nested stage selection")
    if tuple(selection.get("ordered_stages", [])) != tuple(
        stage.name for stage in fully_nested_policy.ORDERED_STAGES
    ):
        raise RuntimeError("active fully nested stage order drifted")
    if int(selection.get("minimum_prior_scored_elections", -1)) != 2:
        raise RuntimeError("active fully nested minimum history drifted")
    if selection.get("mode") != "fixed_universal_evidence_gated_pipeline":
        raise RuntimeError("active deployment pipeline mode drifted")
    if selection.get("fixed_deployment_stage") != "structural_mega_shock_regime":
        raise RuntimeError("active fixed deployment stage drifted")
    if selection.get("undated_issue_importance") != "neutral_default_0.5":
        raise RuntimeError("undated issue importance is not neutralized")
    if selection.get("undated_region_issue_sensitivity") != "neutral_default_0.3":
        raise RuntimeError("undated regional sensitivity is not neutralized")
    return policy


def load_policy(path: Path = CONFIG_PATH) -> dict[str, object]:
    return validate_policy(json.loads(path.read_text(encoding="utf-8")))


def regenerate_assignments() -> None:
    assignments, audit, summary = assignment_builder.build()
    ASSIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        ASSIGNMENT_DIR / "candidate_slot_assignments_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(ASSIGNMENT_DIR / "fold_audit.csv", index=False, encoding="utf-8-sig")
    (ASSIGNMENT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def regenerate_issue_seeds() -> None:
    """Keep lightweight issue seeds synchronized with the active interpretation overlay."""

    issue_seed_builder.write_outputs(
        issue_seed_builder.OUTPUT_DIR,
        issue_seed_builder.DEFAULT_ELECTIONS,
        verbose=False,
    )


def structural_terrain_gain_by_target(
    frame: pd.DataFrame,
    intensity: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Derive PIT-safe terrain strength from party-ballot evidence and shock size."""

    required = {"election_id", "direct_party_reliability"}
    if not required.issubset(frame.columns):
        raise RuntimeError("electorate frame lacks direct-party reliability")
    eligible_intensity = filter_available_by_election(
        intensity.copy(),
        nested.engine.ELECTION_DATES,
        source_name="structural_terrain_mega_intensity",
    )
    eligible_intensity["mega_issue_intensity"] = pd.to_numeric(
        eligible_intensity["mega_issue_intensity"], errors="coerce"
    ).fillna(1.0).clip(lower=0.0)
    latest_intensity = (
        eligible_intensity.sort_values("available_date")
        .drop_duplicates("election_id", keep="last")
        .set_index("election_id")["mega_issue_intensity"]
    )
    multiplier = float(config["reliability_multiplier"])
    cap = float(config["gain_cap"])
    attenuate = bool(config["mega_shock_attenuation"])
    gains: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for target in nested.ELECTIONS:
        reliability = pd.to_numeric(
            frame.loc[frame["election_id"].eq(target), "direct_party_reliability"],
            errors="coerce",
        ).fillna(0.0)
        positive = reliability.loc[reliability.gt(0.0)]
        mean_reliability = float(positive.mean()) if not positive.empty else 0.0
        intensity_value = float(latest_intensity.get(target, 1.0))
        raw_gain = min(max(multiplier * mean_reliability, 0.0), cap)
        shock_divisor = max(intensity_value, 1.0) if attenuate else 1.0
        gain = raw_gain / shock_divisor
        gains[target] = gain
        rows.append(
            {
                "target_election": target,
                "direct_party_reliability": mean_reliability,
                "mega_issue_intensity": intensity_value,
                "raw_terrain_anchor_gain": raw_gain,
                "shock_divisor": shock_divisor,
                "terrain_anchor_gain": gain,
            }
        )
    return gains, pd.DataFrame(rows)


def regional_accent_gain_by_target(
    frame: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Derive one conservative accent gain from strictly prior party evidence."""

    multiplier = float(config["reliability_multiplier"])
    cap = float(config["gain_cap"])
    gains: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for target in nested.ELECTIONS:
        target_rows = frame.loc[frame["election_id"].eq(target)]
        reliability = pd.to_numeric(
            target_rows.get("regional_accent_reliability", 0.0),
            errors="coerce",
        )
        if not isinstance(reliability, pd.Series):
            reliability = pd.Series(float(reliability), index=target_rows.index)
        positive = reliability.fillna(0.0).loc[lambda value: value.gt(0.0)]
        mean_reliability = float(positive.mean()) if not positive.empty else 0.0
        gain = min(max(multiplier * mean_reliability, 0.0), cap)
        gains[target] = gain
        rows.append(
            {
                "target_election": target,
                "regional_accent_reliability": mean_reliability,
                "regional_accent_gain": gain,
            }
        )
    return gains, pd.DataFrame(rows)


def _candidate_metrics(
    candidates: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    election_rows: list[pd.DataFrame] = []
    national_rows: list[pd.DataFrame] = []
    for stage in fully_nested_policy.ORDERED_STAGES:
        summary, by_election, national = nested._metrics(
            candidates[stage.name], "layer_pred", stage.name
        )
        summaries.append(summary)
        election_rows.append(by_election)
        national_rows.append(national)
    return (
        pd.DataFrame(summaries),
        pd.concat(election_rows, ignore_index=True),
        pd.concat(national_rows, ignore_index=True),
    )


def _compose_fully_nested_predictions(
    candidates: Mapping[str, pd.DataFrame],
    stage_by_election: pd.DataFrame,
    minimum_history: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for target in nested.ELECTIONS:
        selected, losses, prior = fully_nested_policy.select_stage_from_prior_folds(
            target,
            nested.ELECTIONS,
            stage_by_election,
            minimum_selection_elections=minimum_history,
        )
        target_rows = candidates[selected].loc[
            candidates[selected]["election_id"].eq(target)
        ].copy()
        target_rows["selected_stage"] = selected
        parts.append(target_rows)
        audit_rows.append(
            {
                "target_election": target,
                "selection_training_elections": "|".join(prior),
                "selection_election_count": len(prior),
                "minimum_selection_elections": minimum_history,
                "selected_stage": selected,
                "target_excluded_from_selection": target not in prior,
                "selection_fallback": len(prior) < minimum_history,
                **{f"inner_mae_{name}": value for name, value in losses.items()},
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit_rows)


def _compose_fixed_pipeline_predictions(
    candidates: Mapping[str, pd.DataFrame],
    stage_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one evidence-gated pipeline to every historical and future target."""

    if stage_name not in candidates:
        raise ValueError(f"unknown fixed deployment stage: {stage_name}")
    predictions = candidates[stage_name].copy()
    predictions["selected_stage"] = stage_name
    audit = pd.DataFrame(
        [
            {
                "target_election": target,
                "selection_training_elections": "",
                "selection_election_count": 0,
                "minimum_selection_elections": 0,
                "selected_stage": stage_name,
                "target_excluded_from_selection": True,
                "selection_fallback": False,
                "selection_policy": "fixed_universal_evidence_gated_pipeline",
            }
            for target in nested.ELECTIONS
        ]
    )
    return predictions, audit


def _input_manifest(records: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    rows = [dict(record) for record in records.values()]
    config_payload = CONFIG_PATH.read_bytes()
    rows.append(
        {
            "path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "bytes": len(config_payload),
            "sha256": hashlib.sha256(config_payload).hexdigest(),
        }
    )
    return pd.DataFrame(rows).drop_duplicates("path").sort_values("path")


def run(
    *,
    output_dir: Path | None = None,
    predictor_orthogonalization_pairs: Sequence[tuple[str, str]] = (),
    direct_party_center_base_gain: float = 0.0,
    regional_offset_base_gain: float | None = None,
    regional_offset_vif_threshold: float | None = None,
    regional_offset_gate_mode: str = "vif",
    chungcheong_identity_gain: float | None = None,
    chungcheong_identity_shift_cap: float | None = None,
    general_regional_identity_gain: float | None = None,
    general_regional_identity_shift_cap: float | None = None,
    rejection_beneficiary_routing_enabled: bool = False,
) -> dict[str, object]:
    destination = OUTPUT_DIR if output_dir is None else Path(output_dir)
    policy = load_policy()
    selection_policy = policy["strict_nested_selection"]
    with strict_input_policy(), track_csv_inputs() as input_paths:
        regenerate_issue_seeds()
        regenerate_assignments()
        full = nested._prepare_rows()
        base = nested._base_layer_frame(require_frozen_reproduction=False)

        strict_outer, strict_outer_audit = nested._build_outer_predictions(
            full,
            EXPECTED_VARIANT,
            predictor_orthogonalization_pairs=predictor_orthogonalization_pairs,
        )
        strict_layered = nested._attach_layers(base, strict_outer)
        strict_predictions, strict_preferences = nested._apply_nested_preference(
            strict_layered, EXPECTED_VARIANT
        )

        structural = policy["structural_layers"]
        identity_policy = structural["chungcheong_regional_identity"]
        effective_identity_gain = (
            float(identity_policy["gain"])
            if chungcheong_identity_gain is None
            else float(chungcheong_identity_gain)
        )
        effective_identity_shift_cap = (
            float(identity_policy["regional_shift_cap"])
            if chungcheong_identity_shift_cap is None
            else float(chungcheong_identity_shift_cap)
        )
        general_identity_policy = structural["general_regional_identity"]
        effective_general_identity_gain = (
            float(general_identity_policy["gain"])
            if general_regional_identity_gain is None
            else float(general_regional_identity_gain)
        )
        effective_general_identity_shift_cap = (
            float(general_identity_policy["regional_shift_cap"])
            if general_regional_identity_shift_cap is None
            else float(general_regional_identity_shift_cap)
        )
        outer, outer_audit = nested._build_outer_predictions(
            full,
            EXPECTED_VARIANT,
            layer_config_overrides=structural["outer_config_overrides"],
            predictor_orthogonalization_pairs=predictor_orthogonalization_pairs,
        )
        layered = nested._attach_layers(base, outer)
        electorate = structural["electorate_response"]
        regional_offset_policy = electorate["regional_swing_offset"]
        effective_regional_offset_gain = (
            float(regional_offset_policy["base_gain"])
            if regional_offset_base_gain is None
            else float(regional_offset_base_gain)
        )
        effective_regional_offset_vif_threshold = (
            float(regional_offset_policy["vif_threshold"])
            if regional_offset_vif_threshold is None
            else float(regional_offset_vif_threshold)
        )
        intensity = pd.read_csv(MEGA_ISSUE_INTENSITY, encoding="utf-8-sig")
        profile = pd.read_csv(CANDIDATE_ISSUE_PROFILE, encoding="utf-8-sig")
        terrain_gains, terrain_audit = structural_terrain_gain_by_target(
            layered, intensity, electorate["terrain_anchor"]
        )
        accent_gains, accent_audit = regional_accent_gain_by_target(
            layered,
            electorate["regional_accent"],
        )
        terrain_audit = terrain_audit.merge(
            accent_audit,
            on="target_election",
            how="left",
        )
        structural_predictions, preference_configs = nested._apply_nested_preference(
            layered,
            EXPECTED_VARIANT,
            preference_gain_floor=float(electorate["preference_gain_floor"]),
            terrain_gain_by_target=terrain_gains,
            regional_accent_gain_by_target=accent_gains,
            regional_accent_signal_width=float(
                electorate["regional_accent"]["signal_width"]
            ),
        )
        direct_party_center_gains: dict[str, float] = {}
        if direct_party_center_base_gain > 0.0:
            for row in terrain_audit.itertuples(index=False):
                direct_party_center_gains[str(row.target_election)] = float(
                    np.clip(
                        direct_party_center_base_gain
                        * float(row.direct_party_reliability)
                        / max(float(row.shock_divisor), 1.0),
                        0.0,
                        0.35,
                    )
                )
            structural_predictions = direct_party_center.apply_direct_party_center(
                structural_predictions,
                prediction_column="layer_pred",
                gain_by_election=direct_party_center_gains,
            )
        regional_offset_gains: dict[str, float] = {}
        if effective_regional_offset_gain > 0.0:
            if regional_offset_gate_mode not in {"vif", "all_available"}:
                raise ValueError(
                    f"unknown regional offset gate mode: {regional_offset_gate_mode}"
                )
            outer_audit_frame = pd.DataFrame(outer_audit)
            shock_divisor_by_target = dict(
                zip(terrain_audit["target_election"], terrain_audit["shock_divisor"])
            )
            for target_index, target in enumerate(nested.ELECTIONS):
                row = outer_audit_frame.loc[
                    outer_audit_frame["target_election"].eq(target)
                ].iloc[0]
                max_vif = float(row["raw_max_predictor_vif"])
                if regional_offset_gate_mode == "all_available":
                    activation = 1.0
                elif target_index < 2 or not np.isfinite(max_vif):
                    activation = 0.0
                else:
                    activation = float(
                        np.clip(
                            np.log(max(max_vif, effective_regional_offset_vif_threshold) / effective_regional_offset_vif_threshold)
                            / np.log(100.0 / effective_regional_offset_vif_threshold),
                            0.0,
                            1.0,
                        )
                    )
                regional_offset_gains[target] = float(
                    effective_regional_offset_gain
                    * activation
                    / max(float(shock_divisor_by_target.get(target, 1.0)), 1.0)
                )
            regional_history = pd.read_csv(
                nested.base_eval.HISTORY_PATH, encoding="utf-8-sig"
            )
            regional_events = regional_swing_elasticity.build_event_frame(
                regional_history
            )
            structural_predictions = regional_swing_elasticity.apply_regional_offset(
                structural_predictions,
                regional_events,
                prediction_column="layer_pred",
                gain_by_election=regional_offset_gains,
                prior_strength=float(regional_offset_policy["prior_strength"]),
            )
        strategic_transfer_policy = structural["strategic_lane_transfer"]
        if bool(strategic_transfer_policy["enabled"]):
            conversion_context = pd.read_csv(CONVERSION_CONTEXT, encoding="utf-8-sig")
            structural_predictions = strategic_lane_transfer.attach_conversion_context(
                structural_predictions,
                conversion_context,
                nested.engine.ELECTION_DATES,
            )
            structural_predictions = strategic_lane_transfer.apply_strategic_lane_transfer(
                structural_predictions,
                prediction_column="layer_pred",
                affinity_power=float(strategic_transfer_policy["affinity_power"]),
            )
        identity_audit = pd.DataFrame()
        if effective_identity_gain > 0.0:
            identity_history = pd.read_csv(
                nested.base_eval.HISTORY_PATH, encoding="utf-8-sig"
            )
            identity_events = chungcheong_identity.build_identity_events(identity_history)
            candidate_regional_base = pd.read_csv(
                CANDIDATE_REGIONAL_BASE, encoding="utf-8-sig"
            )
            identity_alignment = pd.read_csv(
                CHUNGCHEONG_ALIGNMENT, encoding="utf-8-sig"
            )
            structural_predictions, identity_audit = (
                chungcheong_identity.apply_identity_routing(
                    structural_predictions,
                    identity_events,
                    candidate_regional_base,
                    identity_alignment,
                    prediction_column="layer_pred",
                    gain=effective_identity_gain,
                    shift_cap=effective_identity_shift_cap,
                    half_life_years=float(identity_policy["half_life_years"]),
                    prior_strength=float(identity_policy["prior_strength"]),
                )
            )
        regional_identity_audit = pd.DataFrame()
        if effective_general_identity_gain > 0.0:
            general_identity_history = pd.read_csv(
                nested.base_eval.HISTORY_PATH, encoding="utf-8-sig"
            )
            general_identity_events = regional_identity.build_distinctiveness_events(
                general_identity_history
            )
            general_candidate_regional_base = pd.read_csv(
                CANDIDATE_REGIONAL_BASE, encoding="utf-8-sig"
            )
            structural_predictions, regional_identity_audit = (
                regional_identity.apply_regional_identity_routing(
                    structural_predictions,
                    general_identity_events,
                    general_candidate_regional_base,
                    prediction_column="layer_pred",
                    gain=effective_general_identity_gain,
                    shift_cap=effective_general_identity_shift_cap,
                    half_life_years=float(general_identity_policy["half_life_years"]),
                    prior_strength=float(general_identity_policy["prior_strength"]),
                )
            )
        postprocess = policy["postprocess"]
        direct_mega_scores = mega_issue_adjustment.compile_direct_mega_scores(
            profile,
            intensity,
            nested.engine.ELECTION_DATES,
            minimum_intensity=float(postprocess["direct_mega_minimum_intensity"]),
            score_cap=float(postprocess["direct_mega_score_cap"]),
        )
        mega_predictions = mega_issue_adjustment.apply_direct_mega_shift(
            structural_predictions,
            direct_mega_scores,
            prediction_column="layer_pred",
            gain=float(postprocess["direct_mega_logit_gain"]),
            log_shift_cap=float(postprocess["direct_mega_log_shift_cap"]),
        )
        government_burden_scores = (
            incumbent_shock_adjustment.compile_government_burden_scores(
                profile, nested.engine.ELECTION_DATES
            )
        )
        shock_predictions = incumbent_shock_adjustment.apply_incumbent_shock_response(
            mega_predictions,
            government_burden_scores,
            intensity,
            nested.engine.ELECTION_DATES,
            prediction_column="layer_pred",
            government_burden_gain=float(postprocess["government_burden_gain"]),
            rupture_extra_gain=float(postprocess["rupture_extra_gain"]),
            conversion_buffer=float(postprocess["incumbent_conversion_buffer"]),
            log_shift_cap=float(postprocess["incumbent_shock_log_shift_cap"]),
        )
        contest_regimes = contest_regime.derive_contest_regimes(
            shock_predictions,
            prediction_column="layer_pred",
            rejection_double_discount=bool(
                postprocess["contest_regime_rejection_double_discount"]
            ),
        )
        regime_predictions = contest_regime.apply_contest_regime_response(
            shock_predictions,
            contest_regimes,
            prediction_column="layer_pred",
            expansion_gain=float(postprocess["contest_regime_expansion_gain"]),
            log_shift_cap=float(postprocess["contest_regime_log_shift_cap"]),
            critical_elasticity=float(
                postprocess["contest_regime_critical_elasticity"]
            ),
            swing_elasticity=float(postprocess["contest_regime_swing_elasticity"]),
            swing_log_shift_cap=float(
                postprocess["contest_regime_swing_log_shift_cap"]
            ),
        )
        rejection_routing_audit = pd.DataFrame()
        if rejection_beneficiary_routing_enabled:
            regime_predictions, rejection_routing_audit = (
                rejection_beneficiary_routing.apply_rejection_beneficiary_routing(
                    regime_predictions,
                    contest_regimes,
                    prediction_column="layer_pred",
                )
            )
        candidates = {
            "strict_base": strict_predictions,
            "structural": structural_predictions,
            "structural_mega": mega_predictions,
            "structural_mega_shock": shock_predictions,
            "structural_mega_shock_regime": regime_predictions,
        }
        stage_summary, stage_by_election, stage_national = _candidate_metrics(candidates)
        predictions, selection_audit = _compose_fixed_pipeline_predictions(
            candidates,
            str(selection_policy["fixed_deployment_stage"]),
        )
        summary, by_election, national = nested._metrics(
            predictions, "layer_pred", "fully_nested_postprocess"
        )

    audit = pd.concat(
        [
            pd.DataFrame(strict_outer_audit).assign(candidate_stage="strict_base"),
            pd.DataFrame(outer_audit).assign(candidate_stage="structural_stack"),
        ],
        ignore_index=True,
    )
    required_true = {
        "target_excluded_from_fit",
        "consistent_scored_denominator",
        "withdrawn_candidate_direct_adjustment",
    }
    for column in required_true:
        if column not in audit or not audit[column].astype(bool).all():
            raise RuntimeError(f"active nested audit failed: {column}")
    if audit["old_slot_predictors_used"].astype(bool).any():
        raise RuntimeError("realized slot predictors leaked into active model")
    if audit["neutral_context_direct_adjustment"].astype(bool).any():
        raise RuntimeError("neutral-context direct adjustment leaked into active model")

    deployment_stage = str(selection_policy["fixed_deployment_stage"])
    _, deployment_losses = fully_nested_policy.deployment_stage_from_completed_folds(
        stage_by_election, nested.ELECTIONS
    )
    selected_stage_by_election = dict(
        zip(selection_audit["target_election"], selection_audit["selected_stage"])
    )
    payload = {
        "policy_version": policy["policy_version"],
        "variant": EXPECTED_VARIANT,
        "status": "active",
        "strict_nested": True,
        "strict_nested_base": True,
        "strict_nested_postprocess_selection": False,
        "fixed_universal_evidence_gated_pipeline": True,
        "strict_nested_execution": True,
        "untouched_historical_holdout": False,
        "target_excluded_from_each_outer_fit": True,
        "target_excluded_from_each_stage_selection": bool(
            selection_audit["target_excluded_from_selection"].all()
        ),
        "post_2022_outcomes_used": False,
        "predictor_orthogonalization_pairs": [
            list(pair) for pair in predictor_orthogonalization_pairs
        ],
        "direct_party_center_base_gain": float(direct_party_center_base_gain),
        "direct_party_center_gain_by_election": direct_party_center_gains,
        "regional_offset_base_gain": effective_regional_offset_gain,
        "regional_offset_vif_threshold": effective_regional_offset_vif_threshold,
        "regional_offset_gate_mode": regional_offset_gate_mode,
        "regional_offset_gain_by_election": regional_offset_gains,
        "chungcheong_identity_gain": float(effective_identity_gain),
        "chungcheong_identity_shift_cap": float(effective_identity_shift_cap),
        "chungcheong_identity_outcome_fields_used": [],
        "general_regional_identity_gain": float(effective_general_identity_gain),
        "general_regional_identity_shift_cap": float(
            effective_general_identity_shift_cap
        ),
        "general_regional_identity_outcome_fields_used": [],
        "metrics": summary,
        "transfer_adjustment_applied_after_ridge": True,
        "legacy_outcome_aligned_slots_active": False,
        "automatic_issue_seed_schema": issue_seed_builder.SCHEMA_VERSION,
        "automatic_issue_seed_regenerated": True,
        "selected_stage_by_election": selected_stage_by_election,
        "future_deployment_stage": deployment_stage,
        "future_deployment_inner_losses": deployment_losses,
        "minimum_stage_selection_elections": 0,
        "direct_mega_issue_shift": "candidate_stage_only",
        "direct_mega_stage_nested_selected": False,
        "direct_mega_stage_fixed_for_deployment": True,
        "direct_mega_gain_nested_selected": False,
        "direct_mega_gain_development_selected": True,
        "candidate_numeric_parameters_historically_development_selected": True,
        "direct_mega_logit_gain": float(postprocess["direct_mega_logit_gain"]),
        "direct_mega_log_shift_cap": float(postprocess["direct_mega_log_shift_cap"]),
        "direct_mega_minimum_intensity": float(
            postprocess["direct_mega_minimum_intensity"]
        ),
        "direct_mega_score_rows": int(len(direct_mega_scores)),
        "incumbent_shock_response": True,
        "government_burden_gain": float(postprocess["government_burden_gain"]),
        "rupture_extra_gain": float(postprocess["rupture_extra_gain"]),
        "incumbent_conversion_buffer": float(
            postprocess["incumbent_conversion_buffer"]
        ),
        "incumbent_shock_log_shift_cap": float(
            postprocess["incumbent_shock_log_shift_cap"]
        ),
        "government_burden_score_rows": int(len(government_burden_scores)),
        "contest_regime_response": True,
        "contest_regime_expansion_gain": float(
            postprocess["contest_regime_expansion_gain"]
        ),
        "contest_regime_log_shift_cap": float(
            postprocess["contest_regime_log_shift_cap"]
        ),
        "contest_regime_critical_elasticity": float(
            postprocess["contest_regime_critical_elasticity"]
        ),
        "contest_regime_swing_elasticity": float(
            postprocess["contest_regime_swing_elasticity"]
        ),
        "contest_regime_swing_log_shift_cap": float(
            postprocess["contest_regime_swing_log_shift_cap"]
        ),
        "contest_regime_rejection_double_discount": False,
        "contest_regime_rows": int(len(contest_regimes)),
        "rejection_beneficiary_routing_enabled": bool(
            rejection_beneficiary_routing_enabled
        ),
        "rejection_beneficiary_routing_gain": None,
        "rejection_beneficiary_routing_formula": (
            "cumulative_rejection_advantage * rejection_activation * certainty "
            "* runner flexible mass"
        ),
        "rejection_beneficiary_routing_rows": int(len(rejection_routing_audit)),
        "contest_regime_core_floor": "min_effective_direct_core_times_reliability",
        "cumulative_regime_rejection": True,
        "cumulative_rejection_formula": (
            "negative_direction * negative_share * sqrt(min(unique_negative_issues/4,1)) "
            "* max(party_erosion_route, rupture_route) * reliability"
        ),
        "structural_layers_reactivated": True,
        "structural_outer_config_overrides": structural["outer_config_overrides"],
        "electorate_preference_gain_floor": float(electorate["preference_gain_floor"]),
        "terrain_anchor_policy": electorate["terrain_anchor"],
        "regional_accent_policy": electorate["regional_accent"],
        "concrete_support_policy": electorate["concrete_support"],
        "strategic_lane_transfer_policy": structural["strategic_lane_transfer"],
        "chungcheong_regional_identity_policy": identity_policy,
        "general_regional_identity_policy": general_identity_policy,
        "party_context_cohesion_policy": structural["party_context_cohesion"],
        "undated_issue_importance_policy": selection_policy[
            "undated_issue_importance"
        ],
        "undated_region_issue_sensitivity_policy": selection_policy[
            "undated_region_issue_sensitivity"
        ],
        "historical_candidate_set_warning": (
            "Nested execution blocks target-fold stage-selection leakage, but the "
            "stage definitions and numeric gains were historically developed on the "
            "through-2022 sample and cannot become an untouched holdout retroactively."
        ),
        "frozen_reproduction_difference": float(
            pd.to_numeric(
                layered["frozen_reproduction_difference"], errors="coerce"
            ).max()
        ),
        "frozen_reproduction_guard": (
            "measured but not required for a versioned upstream issue-layer change"
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_csv(predictions, destination / "nested_predictions.csv")
    _atomic_csv(by_election, destination / "by_election.csv")
    _atomic_csv(national, destination / "national_predictions.csv")
    _atomic_csv(audit, destination / "fold_audit.csv")
    _atomic_csv(selection_audit, destination / "stage_selection_audit.csv")
    _atomic_csv(stage_summary, destination / "candidate_stage_summary.csv")
    _atomic_csv(stage_by_election, destination / "candidate_stage_by_election.csv")
    _atomic_csv(stage_national, destination / "candidate_stage_national.csv")
    _atomic_csv(pd.DataFrame(strict_preferences), destination / "strict_preference_gain_by_fold.csv")
    _atomic_csv(pd.DataFrame(preference_configs), destination / "preference_gain_by_fold.csv")
    _atomic_csv(terrain_audit, destination / "terrain_anchor_by_fold.csv")
    _atomic_csv(identity_audit, destination / "chungcheong_identity_audit.csv")
    _atomic_csv(regional_identity_audit, destination / "regional_identity_audit.csv")
    _atomic_csv(direct_mega_scores, destination / "direct_mega_issue_scores.csv")
    _atomic_csv(government_burden_scores, destination / "government_burden_scores.csv")
    _atomic_csv(contest_regimes, destination / "contest_regimes.csv")
    _atomic_csv(
        rejection_routing_audit,
        destination / "rejection_beneficiary_routing_audit.csv",
    )
    _atomic_csv(_input_manifest(input_paths), destination / "input_manifest.csv")
    _atomic_json(payload, destination / "summary.json")
    return payload


def main() -> None:
    payload = run()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
