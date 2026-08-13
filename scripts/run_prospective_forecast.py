"""Run an outcome-free prospective presidential forecast through frozen policy code.

The runner wraps the existing V22/V23 execution stack.  It supplies an
outcome-free candidate skeleton for the forecast election, appends the D-1
Assembly inputs, disables every scoring sink, and writes only prospective
artifacts.  No prediction formula is reimplemented here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Iterator

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import automatic_contest_response  # noqa: E402
from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine import unified_lineage_identity  # noqa: E402
from presidential_issue_engine.election_scope import (  # noqa: E402
    ELECTION_DATES,
    SCORED_ELECTIONS,
    assert_election_scope,
)
from presidential_issue_engine.electorate_layers import (  # noqa: E402
    MAJOR_PARTY_CORE_BLOCS,
)
from presidential_issue_engine.forecast_only_inputs import (  # noqa: E402
    FORBIDDEN_OUTCOME_COLUMNS,
    load_forecast_only_assembly_inputs,
)
from presidential_issue_engine.point_in_time import forecast_cutoff  # noqa: E402
from scripts import build_preliminary_slot_assignments as assignment_builder  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


TARGET_ELECTION = "pres_2025"
FORECAST_CUTOFF = "2025-06-02"
REGISTRY = ROOT / "data/raw/official_sources/pres_2025_candidate_registry.csv"
CONTEXT_DIR = ROOT / "data/raw/official_sources/assembly_pres_2025_context"
REGIONS = ROOT / "presidential_issue_engine/fixed_dataset/regions_master.csv"
RESULTS = (
    ROOT
    / "presidential_issue_engine/fixed_dataset/presidential_results_standardized.csv"
)
HISTORY = ROOT / "presidential_issue_engine/fixed_dataset/bloc_history_results.csv"
OUTER_CONFIG = (
    ROOT
    / "presidential_issue_engine/report/through2022_rederived/nested_outer_results.csv"
)
DEPLOYMENT_CONFIG = ROOT / "data/config/through2022_rederived_layers.json"
V23_CONFIG = ROOT / "data/config/active_presidential_model_v23.json"
V24_CONFIG = ROOT / "data/config/active_presidential_model_v24.json"
V23_ASSIGNMENTS = (
    ROOT / "outputs/preliminary_slot_assignment_v23/candidate_slot_assignments_v2.csv"
)
AUTOMATIC_DIR = ROOT / "outputs/automatic_controls_v23"
FOOTPRINT_BASE = (
    ROOT / "outputs/footprint_candidate_base_v9/candidate_regional_base.csv"
)
PARTY_TRANSITIONS = ROOT / "data/raw/party_lineage_transitions.csv"
CANDIDATE_LINK_HISTORY = ROOT / "data/candidate_issue_link.csv"
CANDIDATE_CONVERSION_HISTORY = ROOT / "data/raw/candidate_vote_conversion_context.csv"

OUTPUT_COLUMNS = (
    "election_id",
    "region_id",
    "slot",
    "candidate_name",
    "predicted_share",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_path(version: str) -> Path:
    path = V23_CONFIG if version == "v23" else V24_CONFIG
    if not path.exists():
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        raise RuntimeError(
            f"{version} has no human-promoted config at {display_path}; "
            "V24 ablations are measurement-only and cannot be selected automatically"
        )
    return path


def _validate_registry(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    required = {
        "election_id",
        "candidate_id",
        "candidate_name",
        "party_name",
        "ballot_number",
        "available_date",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate registry is missing columns: {missing}")
    forbidden = sorted(set(frame.columns) & FORBIDDEN_OUTCOME_COLUMNS)
    if forbidden:
        raise RuntimeError(f"candidate registry contains outcome columns: {forbidden}")
    if set(frame["election_id"].astype(str)) != {TARGET_ELECTION}:
        raise RuntimeError("candidate registry contains another election")
    out = frame.copy()
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    if out["available_date"].isna().any() or out["available_date"].gt(cutoff).any():
        raise RuntimeError("candidate registry is not fully available by forecast cutoff")
    return out


def _select_model_candidates(
    registry: pd.DataFrame,
    candidate_link: pd.DataFrame,
    engine,
) -> pd.DataFrame:
    """Select the two major-camp nominees and strongest non-major lineage.

    The ordering uses the latest strictly prior direct-party national result
    (2024 Assembly PR).  Candidate attention only resolves candidates within a
    shared bloc and never reads polling or the target result.
    """

    candidates = registry.copy()
    candidates["bloc"] = (
        candidates["party_name"].map(engine.party_bloc).map(engine.normalize_bloc)
    )
    attention = candidate_link.groupby("candidate_id")["emphasis_volume"].sum()
    candidates["candidate_attention"] = (
        candidates["candidate_id"].map(attention).fillna(0.0)
    )
    history = pd.read_csv(HISTORY, encoding="utf-8-sig")
    prior = (
        history.loc[history["election_id"].eq("assembly_2024_pr")]
        .groupby("bloc")["vote_share"]
        .mean()
    )
    candidates["prior_bloc_share"] = candidates["bloc"].map(prior).fillna(0.0)

    major_rows: list[pd.Series] = []
    for bloc in MAJOR_PARTY_CORE_BLOCS:
        eligible = candidates.loc[candidates["bloc"].eq(bloc)].sort_values(
            ["candidate_attention", "ballot_number"], ascending=[False, True]
        )
        if eligible.empty:
            raise RuntimeError(f"candidate registry lacks a nominee for major bloc {bloc}")
        major_rows.append(eligible.iloc[0])
    major = pd.DataFrame(major_rows).sort_values(
        ["prior_bloc_share", "candidate_attention"], ascending=False
    )
    nonmajor = candidates.loc[
        ~candidates["bloc"].isin(MAJOR_PARTY_CORE_BLOCS)
        & candidates["candidate_id"].isin(candidate_link["candidate_id"])
    ].sort_values(
        ["prior_bloc_share", "candidate_attention", "ballot_number"],
        ascending=[False, False, True],
    )
    if nonmajor.empty:
        raise RuntimeError("no outcome-free non-major candidate has Assembly context")
    selected = pd.concat([major, nonmajor.head(1)], ignore_index=True)
    selected["slot"] = ["A", "B", "C"]
    return selected


def _candidate_strength_context(
    selected: pd.DataFrame,
    candidate_link: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Project the existing speech-derived candidate-weight scale to 2025.

    The historical target is the frozen, outcome-free candidate weight rather
    than vote share.  A small ridge model maps issue-link attention to that
    scale and is fitted only on elections through 2022.  This is a deployment
    adapter for an input that is unavailable in the compact D-1 context, not a
    new election-outcome model.
    """

    historical_link = pd.read_csv(CANDIDATE_LINK_HISTORY, encoding="utf-8-sig")
    historical_context = pd.read_csv(
        CANDIDATE_CONVERSION_HISTORY, encoding="utf-8-sig"
    )

    def aggregate(
        frame: pd.DataFrame,
        identity_columns: list[str],
    ) -> pd.DataFrame:
        out = (
            frame.groupby(identity_columns, as_index=False)
            .agg(
                mentions=("mentions", "sum"),
                emphasis_volume=("emphasis_volume", "sum"),
            )
        )
        denominator = out.groupby("election_id")["emphasis_volume"].transform(
            "sum"
        )
        out["attention_share"] = np.divide(
            out["emphasis_volume"],
            denominator,
            out=np.zeros(len(out), dtype=float),
            where=denominator.to_numpy(float) > 0.0,
        )
        out["log_attention"] = np.log1p(out["emphasis_volume"])
        return out

    historical = aggregate(historical_link, ["election_id", "slot"]).merge(
        historical_context[
            ["election_id", "slot", "candidate_weight", "confidence"]
        ],
        on=["election_id", "slot"],
        how="inner",
        validate="one_to_one",
    )
    if TARGET_ELECTION in set(historical["election_id"].astype(str)):
        raise RuntimeError("candidate-strength training includes the forecast election")
    historical["is_nonmajor_slot"] = historical["slot"].astype(str).eq("C").astype(float)
    target = aggregate(
        candidate_link,
        ["election_id", "candidate_id"],
    ).merge(
        selected[["candidate_id", "candidate_name", "slot"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    target["is_nonmajor_slot"] = target["slot"].astype(str).eq("C").astype(float)

    feature_names = ["log_attention", "attention_share", "is_nonmajor_slot"]
    x = historical[feature_names].to_numpy(float)
    y = pd.to_numeric(historical["candidate_weight"], errors="raise").to_numpy(float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    alpha = 1.0
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ y,
    )
    target_design = np.column_stack(
        [np.ones(len(target)), (target[feature_names].to_numpy(float) - mean) / scale]
    )
    target["candidate_weight"] = np.clip(target_design @ coefficients, 0.08, 1.0)

    historical_confidence = pd.to_numeric(
        historical["confidence"], errors="coerce"
    ).fillna(0.0)
    target["confidence"] = float(historical_confidence.median())
    target["available_date"] = FORECAST_CUTOFF
    target["notes"] = (
        "Projected from through-2022 speech-derived candidate weights using "
        "D-1 Assembly issue-link attention; no election outcomes or polling"
    )
    for column in [
        "coalition_cohesion",
        "coalition_mobilization_score",
        "wasted_vote_resistance",
        "major_party_gravity",
        "third_character_constraint",
        "third_candidate_overexposure_risk",
        "attention_to_support_gap",
        "conversion_capacity",
    ]:
        target[column] = 0.0
    columns = list(historical_context.columns)
    target_context = target.reindex(columns=columns)
    combined = pd.concat([historical_context, target_context], ignore_index=True)
    diagnostics = {
        "method": "ridge_projection_to_speech_derived_candidate_weight",
        "training_elections": list(SCORED_ELECTIONS),
        "target_outcomes_used": False,
        "polling_used": False,
        "ridge_alpha": alpha,
        "feature_names": feature_names,
        "training_rows": int(len(historical)),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients_intercept_first": coefficients.tolist(),
        "projected_candidates": target[
            [
                "candidate_id",
                "candidate_name",
                "slot",
                "mentions",
                "emphasis_volume",
                "attention_share",
                "candidate_weight",
                "confidence",
            ]
        ].to_dict("records"),
    }
    return combined, diagnostics


def _prospective_sources(
    temp: Path,
    selected: pd.DataFrame,
    salience: pd.DataFrame,
    candidate_link: pd.DataFrame,
) -> tuple[dict[str, Path], dict[str, object]]:
    regions = pd.read_csv(REGIONS, encoding="utf-8-sig")
    skeleton = selected.merge(regions, how="cross")
    skeleton["election_id"] = TARGET_ELECTION
    skeleton["is_active_slot"] = True
    # Structural placeholders create rows only.  The target is excluded from
    # every fit and no code path may score these values.
    skeleton["votes"] = 0.0
    skeleton["vote_share"] = 0.0
    skeleton = skeleton[
        [
            "election_id",
            "region_id",
            "region_name",
            "province",
            "slot",
            "candidate_name",
            "party_name",
            "is_active_slot",
            "votes",
            "vote_share",
        ]
    ]
    results = pd.concat(
        [pd.read_csv(RESULTS, encoding="utf-8-sig"), skeleton],
        ignore_index=True,
    )

    historical_salience = pd.read_csv(ROOT / active.nested.engine.SALIENCE)
    salience = salience.copy()
    for column in ("period", "available_date"):
        salience[column] = pd.to_datetime(salience[column]).dt.strftime("%Y-%m-%d")
    combined_salience = pd.concat(
        [historical_salience, salience[historical_salience.columns]],
        ignore_index=True,
    )

    link = candidate_link.merge(
        selected[["candidate_id", "slot"]], on="candidate_id", how="inner"
    )
    link = link[
        [
            "election_id",
            "slot",
            "issue_name",
            "mentions",
            "emphasis_volume",
            "emphasis_within",
            "available_date",
        ]
    ].copy()
    link["available_date"] = pd.to_datetime(link["available_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    historical_link = pd.read_csv(ROOT / active.nested.engine.LINK)
    combined_link = pd.concat([historical_link, link], ignore_index=True)
    candidate_context, candidate_context_diagnostics = _candidate_strength_context(
        selected, candidate_link
    )

    outer = pd.read_csv(OUTER_CONFIG, encoding="utf-8-sig")
    deployment = json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))["config"]
    target_config = {column: np.nan for column in outer.columns}
    target_config.update(deployment)
    target_config.update(
        {
            "target_election": TARGET_ELECTION,
            "tuning_elections": "|".join(SCORED_ELECTIONS),
            # The active variant is the frozen no-neutral path.  Its nested
            # overlay registry intentionally exposes only the zero overlay.
            "overlay_gain": 0.0,
            "outer_row_mae_pp": np.nan,
            "outer_rows": 0,
        }
    )
    outer = pd.concat([outer, pd.DataFrame([target_config])], ignore_index=True)

    paths = {
        "results": temp / "outcome_free_presidential_rows.csv",
        "salience": temp / "issue_salience_through_cutoff.csv",
        "link": temp / "candidate_issue_link_through_cutoff.csv",
        "candidate_context": temp / "candidate_vote_conversion_context.csv",
        "outer_config": temp / "nested_outer_with_prospective_target.csv",
    }
    for key, frame in (
        ("results", results),
        ("salience", combined_salience),
        ("link", combined_link),
        ("candidate_context", candidate_context),
        ("outer_config", outer),
    ):
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    return paths, candidate_context_diagnostics


def _prior_region_volume() -> pd.Series:
    results = pd.read_csv(RESULTS, encoding="utf-8-sig")
    prior = results.loc[results["election_id"].eq("pres_2022")].copy()
    return pd.to_numeric(prior["votes"], errors="coerce").fillna(0.0).groupby(
        prior["region_id"]
    ).sum()


def _target_base(target: pd.DataFrame, historical_base: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    prior_volume = _prior_region_volume()
    out["contest_votes"] = out["region_id"].map(prior_volume).fillna(0.0)
    out["actual"] = np.nan
    out["official_pred"] = np.nan
    out["replacement_base_pred"] = 1.0 / 3.0
    out["pred"] = 1.0 / 3.0
    out["frozen_reproduction_difference"] = 0.0
    out["frozen_reproduction_guard_required"] = False
    if "candidate_name_x" in historical_base.columns:
        out["candidate_name_x"] = out["candidate_name"]
    if "candidate_name_y" in historical_base.columns:
        out["candidate_name_y"] = out["candidate_name"]
    # The assembled target already contains the same electorate and issue
    # feature contract.  Missing diagnostic-only columns are inert.
    for column in historical_base.columns:
        if column not in out.columns:
            out[column] = np.nan if column == "actual" else 0.0
    return out.reindex(columns=historical_base.columns)


def _target_full(
    target: pd.DataFrame,
    historical_full: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    assignment_columns = [
        "election_id",
        "candidate_name",
        "assigned_slot",
        "preliminary_mean_share",
        "pre_withdrawal_mean_share",
        "post_withdrawal_mean_share",
        "withdrawal_event_applied",
    ]
    out = target.merge(
        assignments[assignment_columns],
        on=["election_id", "candidate_name"],
        how="left",
        validate="many_to_one",
    )
    if out["assigned_slot"].isna().any():
        raise RuntimeError("prospective candidates lack preliminary assignments")
    out["source_slot"] = out["slot"]
    out["slot"] = out["assigned_slot"]
    out["prelim_slot_A"] = out["slot"].eq("A").astype(float)
    out["prelim_slot_B"] = out["slot"].eq("B").astype(float)
    out["prelim_slotA_prior"] = out["prelim_slot_A"] * out["partisan_prior"]
    out["prelim_slotB_prior"] = out["prelim_slot_B"] * out["partisan_prior"]
    event = out["withdrawal_event_applied"].fillna(False).astype(bool)
    out["prelim_withdrawal_event"] = event.astype(float)
    out["prelim_withdrawal_share"] = np.where(
        event,
        pd.to_numeric(out["post_withdrawal_mean_share"], errors="coerce").fillna(0.0),
        0.0,
    )
    out["_order"] = len(SCORED_ELECTIONS) + 1
    for column in historical_full.columns:
        if column not in out.columns:
            out[column] = 0.0
    return pd.concat(
        [historical_full, out.reindex(columns=historical_full.columns)],
        ignore_index=True,
        sort=False,
    )


@contextmanager
def _v23_runtime(config_path: Path, selected: pd.DataFrame) -> Iterator[None]:
    """Apply the existing V22/V23 wrappers without rebuilding frozen inputs."""

    history = pd.read_csv(HISTORY, encoding="utf-8-sig")
    assembly = pd.read_csv(
        ROOT / "data/raw/official_sources/nec_assembly_district_history.csv",
        encoding="utf-8-sig",
    )
    events = unified_lineage_identity.build_exact_lineage_events(history, assembly)
    transitions = pd.read_csv(PARTY_TRANSITIONS, encoding="utf-8-sig")
    candidate_parties = (
        pd.read_csv(
            RESULTS,
            encoding="utf-8-sig",
            usecols=["election_id", "slot", "candidate_name", "party_name"],
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )
    candidate_parties = pd.concat(
        [
            candidate_parties,
            selected[["slot", "candidate_name", "party_name"]].assign(
                election_id=TARGET_ELECTION
            ),
        ],
        ignore_index=True,
    )
    original_response = contest_regime.apply_contest_regime_response

    def automatic_apply(frame, regimes, *, prediction_column, slot_column="source_slot", output_column=None, critical_elasticity=0.75, swing_elasticity=1.25, **_):
        return automatic_contest_response.apply_prior_selected_contest_response(
            frame,
            regimes,
            prediction_column=prediction_column,
            apply_response=original_response,
            election_order=active.nested.ELECTIONS,
            slot_column=slot_column,
            output_column=output_column,
            critical_elasticity=critical_elasticity,
            swing_elasticity=swing_elasticity,
        )[0]

    def unified_prior(frame, _history, election_order):
        return unified_lineage_identity.attach_lineage_projected_prior(
            frame, events, candidate_parties, election_order
        )

    def unified_apply(frame, _events, candidate_regional_base, alignment, *, prediction_column, gain, shift_cap=0.08, half_life_years=12.0, prior_strength=1.5):
        adjusted, audit, _ = unified_lineage_identity.apply_unified_lineage_routing(
            frame,
            events,
            candidate_regional_base,
            alignment,
            candidate_parties,
            transitions,
            prediction_column=prediction_column,
            gain=gain,
            shift_cap=shift_cap,
            half_life_years=half_life_years,
            prior_strength=prior_strength,
            include_direct_lineage_score=True,
            direct_lineage_scope="non_major",
        )
        return adjusted, audit

    original_load = active.load_policy

    def load_selected_policy(path=config_path):
        return original_load(config_path)

    engines = {active.nested.engine, active.assignment_builder.engine}
    attributes: list[tuple[object, str, object]] = [
        (active, "CONFIG_PATH", config_path),
        (active, "load_policy", load_selected_policy),
        (active.contest_regime, "apply_contest_regime_response", automatic_apply),
        (active.nested.engine, "attach_bloc_prior", unified_prior),
        (active.assignment_builder.engine, "attach_bloc_prior", unified_prior),
        (active.chungcheong_identity, "build_identity_events", lambda _: events),
        (active.chungcheong_identity, "apply_identity_routing", unified_apply),
        (active, "MEGA_ISSUE_INTENSITY", AUTOMATIC_DIR / "mega_issue_intensity.csv"),
    ]
    source_overrides = {
        "WITHDRAWAL_TRANSFER_REGISTRY": AUTOMATIC_DIR / "withdrawal_transfer_registry.csv",
        "ELECTION_GENERATION_WEIGHTS": AUTOMATIC_DIR / "election_generation_weights.csv",
        "ENHANCED_MEGA_ISSUE_INTENSITY": AUTOMATIC_DIR / "mega_issue_intensity.csv",
        "MEGA_ISSUE_TAXONOMY": AUTOMATIC_DIR / "mega_issue_taxonomy.csv",
        "ECONOMIC_SLOT_ALIGNMENT": AUTOMATIC_DIR / "economic_slot_alignment.csv",
        "HOUSING_SLOT_ALIGNMENT": AUTOMATIC_DIR / "housing_slot_alignment.csv",
        "CANDIDATE_POLITICAL_LANDSCAPE": AUTOMATIC_DIR / "candidate_political_landscape.csv",
    }
    for engine in engines:
        attributes.extend(
            (engine, name, str(path)) for name, path in source_overrides.items()
        )
    attributes.extend(
        [
            (active, "CANDIDATE_REGIONAL_BASE", FOOTPRINT_BASE),
            (
                active,
                "CHUNGCHEONG_ALIGNMENT",
                AUTOMATIC_DIR / "regional_alignment_with_policy.csv",
            ),
            (
                active.nested.engine,
                "THIRD_CANDIDATE_PROFILE",
                str(AUTOMATIC_DIR / "third_candidate_profile.csv"),
            ),
            (
                active.assignment_builder.engine,
                "THIRD_CANDIDATE_PROFILE",
                str(AUTOMATIC_DIR / "third_candidate_profile.csv"),
            ),
        ]
    )
    with patching.patched(attributes):
        yield


def _execute_existing_pipeline(
    config_path: Path,
    sources: dict[str, Path],
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_elections = (*SCORED_ELECTIONS, TARGET_ELECTION)
    engines = {active.nested.engine, active.assignment_builder.engine}
    captures: dict[str, pd.DataFrame] = {}

    with _v23_runtime(config_path, selected):
        historical_full = active.nested._prepare_rows()
        historical_base = active.nested._base_layer_frame(
            require_frozen_reproduction=False
        )
        source_attributes: list[tuple[object, str, object]] = []
        for engine in engines:
            source_attributes.extend(
                [
                    (engine, "RESULTS", str(sources["results"])),
                    (engine, "SALIENCE", str(sources["salience"])),
                    (engine, "LINK", str(sources["link"])),
                    (engine, "ORDER", list(all_elections)),
                    (
                        engine,
                        "REGIONAL_BASE_ORDER",
                        ["pres_1992", "pres_1997", *all_elections],
                    ),
                    (
                        engine,
                        "CANDIDATE_VOTE_CONVERSION_CONTEXT",
                        str(sources["candidate_context"]),
                    ),
                ]
            )
        with patching.patched(source_attributes):
            target = active.nested.engine.assemble()
            target = target.loc[target["election_id"].eq(TARGET_ELECTION)].copy()
            if len(target) != 51:
                raise RuntimeError(f"expected 51 prospective rows, found {len(target)}")
            assignment_attributes = [
                (assignment_builder, "ELECTIONS", all_elections),
                (active.assignment_builder, "ELECTIONS", all_elections),
            ]
            with patching.patched(assignment_attributes):
                assignments, assignment_audit, _ = assignment_builder.build()
            target_assignments = assignments.loc[
                assignments["election_id"].eq(TARGET_ELECTION)
            ].copy()
            full = _target_full(target, historical_full, target_assignments)
            base = pd.concat(
                [historical_base, _target_base(target, historical_base)],
                ignore_index=True,
                sort=False,
            )

            original_atomic_csv = active._atomic_csv

            def capture_csv(frame: pd.DataFrame, path: Path) -> None:
                captures[Path(path).name] = frame.copy()

            def no_metrics(candidates):
                del candidates
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            def no_score(frame, prediction_column, variant):
                del frame, prediction_column, variant
                return {"performance_metrics_computed": False}, pd.DataFrame(), pd.DataFrame()

            def no_deployment_losses(stage_by_election, election_order):
                del stage_by_election, election_order
                return "structural_mega_shock_regime", {}

            runtime_attributes = [
                (active, "regenerate_issue_seeds", lambda: None),
                (active, "regenerate_assignments", lambda: None),
                (active, "_candidate_metrics", no_metrics),
                (active, "_atomic_csv", capture_csv),
                (active, "_atomic_json", lambda payload, path: None),
                (active.nested, "_prepare_rows", lambda: full),
                (
                    active.nested,
                    "_base_layer_frame",
                    lambda require_frozen_reproduction=False: base,
                ),
                (active.nested, "_metrics", no_score),
                (active.nested, "ELECTIONS", all_elections),
                (active.nested.base_eval, "ALLOWED_ELECTIONS", all_elections),
                (active.nested, "CONFIG_PATH", sources["outer_config"]),
                (
                    active.fully_nested_policy,
                    "deployment_stage_from_completed_folds",
                    no_deployment_losses,
                ),
            ]
            try:
                with patching.patched(runtime_attributes):
                    active.run(output_dir=Path(tempfile.gettempdir()) / "prospective_sink")
            finally:
                active._atomic_csv = original_atomic_csv

    predictions = captures.get("nested_predictions.csv")
    input_manifest = captures.get("input_manifest.csv", pd.DataFrame())
    if predictions is None:
        raise RuntimeError("existing pipeline did not emit prospective predictions")
    target_predictions = predictions.loc[
        predictions["election_id"].eq(TARGET_ELECTION)
    ].copy()
    target_predictions = target_predictions.rename(
        columns={"source_slot": "slot", "layer_pred": "predicted_share"}
    )
    name_column = (
        "candidate_name_x"
        if "candidate_name_x" in target_predictions.columns
        else "candidate_name"
    )
    target_predictions = target_predictions.rename(columns={name_column: "candidate_name"})
    target_predictions = target_predictions.loc[:, ~target_predictions.columns.duplicated()]
    return target_predictions[list(OUTPUT_COLUMNS)], input_manifest


def _national_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    volume = _prior_region_volume()
    work = predictions.copy()
    work["prior_election_vote_weight"] = work["region_id"].map(volume).fillna(0.0)
    rows: list[dict[str, object]] = []
    for (slot, candidate_name), group in work.groupby(
        ["slot", "candidate_name"], sort=True
    ):
        weights = group["prior_election_vote_weight"].to_numpy(float)
        rows.append(
            {
                "election_id": TARGET_ELECTION,
                "slot": slot,
                "candidate_name": candidate_name,
                "predicted_share": float(
                    np.average(group["predicted_share"], weights=weights)
                ),
                "region_weight_source": "pres_2022_valid_vote_volume",
            }
        )
    out = pd.DataFrame(rows)
    out["predicted_share"] /= out["predicted_share"].sum()
    return out


def _input_manifest(
    captured: pd.DataFrame,
    config_path: Path,
) -> pd.DataFrame:
    explicit = [
        config_path,
        REGISTRY,
        CONTEXT_DIR / "model_issue_salience.csv",
        CONTEXT_DIR / "model_candidate_issue_link.csv",
        CONTEXT_DIR / "manifest.json",
        RESULTS,
        HISTORY,
        REGIONS,
        OUTER_CONFIG,
        DEPLOYMENT_CONFIG,
        CANDIDATE_LINK_HISTORY,
        CANDIDATE_CONVERSION_HISTORY,
        PARTY_TRANSITIONS,
        ROOT / "data/raw/official_sources/nec_assembly_district_history.csv",
        ROOT / active.nested.engine.SALIENCE,
    ]
    rows = []
    if not captured.empty and {"path", "bytes", "sha256"}.issubset(captured.columns):
        rows.extend(captured[["path", "bytes", "sha256"]].to_dict("records"))
    rows.extend(
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in explicit
    )
    out = pd.DataFrame(rows).drop_duplicates("path", keep="last")
    # Temporary composition files are deterministic intermediates rather than
    # repository inputs and are intentionally not exposed as machine paths.
    absolute = out["path"].astype(str).map(lambda value: Path(value).is_absolute())
    out = out.loc[~absolute].sort_values("path").reset_index(drop=True)
    return out


def run(version: str = "v23") -> Path:
    assert_election_scope()
    cutoff = forecast_cutoff(TARGET_ELECTION, ELECTION_DATES)
    if cutoff is None or cutoff.date().isoformat() != FORECAST_CUTOFF:
        raise RuntimeError("pres_2025 forecast cutoff drifted from D-1")
    config_path = _config_path(version)
    registry = _validate_registry(
        pd.read_csv(REGISTRY, encoding="utf-8-sig"), cutoff
    )
    salience, candidate_link, context_manifest = load_forecast_only_assembly_inputs(
        TARGET_ELECTION, CONTEXT_DIR
    )
    selected = _select_model_candidates(
        registry, candidate_link, active.nested.engine
    )

    output_dir = ROOT / "outputs" / f"prospective_pres_2025_{version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prospective_pres_2025_") as temp_dir:
        sources, candidate_context_diagnostics = _prospective_sources(
            Path(temp_dir), selected, salience, candidate_link
        )
        predictions, captured_manifest = _execute_existing_pipeline(
            config_path, sources, selected
        )

    forbidden_output = {
        column
        for column in predictions.columns
        if column.casefold() in FORBIDDEN_OUTCOME_COLUMNS
        or "actual" in column.casefold()
        or "error" in column.casefold()
        or "mae" in column.casefold()
    }
    if forbidden_output:
        raise RuntimeError(f"prospective output contains outcome columns: {forbidden_output}")
    if not np.allclose(
        predictions.groupby(["election_id", "region_id"])["predicted_share"].sum(),
        1.0,
    ):
        raise RuntimeError("prospective regional shares do not sum to one")

    national = _national_summary(predictions)
    manifest = _input_manifest(captured_manifest, config_path)
    predictions.to_csv(
        output_dir / "prospective_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    national.to_csv(
        output_dir / "national_summary.csv", index=False, encoding="utf-8-sig"
    )
    manifest.to_csv(
        output_dir / "input_manifest.csv", index=False, encoding="utf-8-sig"
    )
    run_manifest = {
        "schema": "prospective_presidential_forecast_v1",
        "status": "forecast_only_not_scored",
        "election_id": TARGET_ELECTION,
        "version": version,
        "forecast_cutoff": FORECAST_CUTOFF,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "training_scored_elections": list(SCORED_ELECTIONS),
        "training_latest_election": "pres_2022",
        "candidate_selection": selected[
            [
                "candidate_id",
                "candidate_name",
                "party_name",
                "slot",
                "prior_bloc_share",
                "candidate_attention",
            ]
        ].to_dict("records"),
        "candidate_selection_outcome_fields_used": [],
        "final_model_slot_assignment": predictions[
            ["slot", "candidate_name"]
        ].drop_duplicates().sort_values("slot").to_dict("records"),
        "candidate_strength_projection": candidate_context_diagnostics,
        "national_region_weight_source": "pres_2022_valid_vote_volume",
        "outcome_columns_used": [],
        "performance_metrics_computed": False,
        "pres_2025_outcome_present": False,
        "assembly_context_manifest_sha256": _sha256(CONTEXT_DIR / "manifest.json"),
        "assembly_context_certification": context_manifest.get("status"),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v23", "v24"), default="v23")
    args = parser.parse_args()
    output_dir = run(args.version)
    print(output_dir.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
