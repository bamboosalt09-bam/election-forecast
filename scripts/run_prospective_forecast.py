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
import os
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
from presidential_issue_engine.automatic_controls_v22 import (  # noqa: E402
    SHOCK_CLASS_INTENSITY,
    build_automatic_mega_taxonomy,
)
from election_forecast.features.issue_matcher import match_issue_weights  # noqa: E402
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
from presidential_issue_engine.speech_derived_mega_intensity import (  # noqa: E402
    build_automatic_mega_issue_intensity,
)
from scripts import build_preliminary_slot_assignments as assignment_builder  # noqa: E402
from scripts import build_speech_derived_candidate_context_v2 as context_builder  # noqa: E402
from scripts import build_speech_derived_issue_context as issue_context_builder  # noqa: E402
from scripts import extract_assembly_speaker_issue_matches as assembly_match_builder  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402
from scripts import run_active_presidential_model_v24 as active_v24  # noqa: E402
from scripts import run_active_presidential_model_v25 as active_v25  # noqa: E402


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
V24_CONFIG = active_v24.CONFIG_PATH
V25_CONFIG = active_v25.CONFIG_PATH
V23_ASSIGNMENTS = (
    ROOT / "outputs/preliminary_slot_assignment_v23/candidate_slot_assignments_v2.csv"
)
AUTOMATIC_DIR = ROOT / "outputs/automatic_controls_v23"
FOOTPRINT_BASE = (
    ROOT / "outputs/footprint_candidate_base_v9/candidate_regional_base.csv"
)
PARTY_TRANSITIONS = ROOT / "data/raw/party_lineage_transitions.csv"
CANDIDATE_LINK_HISTORY = ROOT / "data/candidate_issue_link.csv"
FORECAST_CANDIDATE_CONTEXT_DIR = CONTEXT_DIR / "candidate_context_v2"
CANDIDATE_CONVERSION_HISTORY = (
    FORECAST_CANDIDATE_CONTEXT_DIR / "candidate_vote_conversion_context.csv"
)
FORECAST_CANDIDATE_LANDSCAPE = (
    FORECAST_CANDIDATE_CONTEXT_DIR / "candidate_political_landscape.csv"
)
FORECAST_THIRD_CANDIDATE_PROFILE = (
    FORECAST_CANDIDATE_CONTEXT_DIR
    / "auto_candidate_role"
    / "third_candidate_profile.csv"
)
FORECAST_ISSUE_SEED_DIR = FORECAST_CANDIDATE_CONTEXT_DIR / "auto_issue_seed"
EXPLICIT_TARGET_CONTEXT = CONTEXT_DIR / "explicit_target_context_weekly.csv"
CANDIDATE_TARGET_CONTEXT = CONTEXT_DIR / "candidate_target_context_weekly.csv"
OFFICIAL_2025_MINUTES = (
    ROOT
    / "data/raw/official_sources/assembly_pres_2025_minutes"
    / "assembly_stance_rows_2025_h1.csv"
)
ISSUE_CONTEXT_RULES = (
    ROOT / "presidential_issue_engine/fixed_dataset/issue_context_rules.csv"
)

OUTPUT_COLUMNS = (
    "election_id",
    "region_id",
    "slot",
    "candidate_name",
    "predicted_share",
)

V24_POSTPROCESS_ORDER = (
    "strong_incumbent_veto",
    "third_candidate_lineage_ceiling",
    "weak_same_lane_refusal",
)

V24_AUDIT_COLUMNS = {
    "strong_incumbent_veto": (
        "election_id",
        "region_id",
        "beneficiary_slot",
        "burdened_slot",
        "projected_margin",
        "government_rejection_strength",
        "dominance_activation",
        "regime_certainty",
        "veto_rate",
        "base_runner_core_floor",
        "rupture_floor_activation",
        "theoretical_floor",
        "effective_runner_floor",
        "runner_flexible_mass",
        "transfer",
    ),
    "third_candidate_lineage_ceiling": (
        "election_id",
        "region_id",
        "candidate_name",
        "before",
        "ceiling",
        "excess_redistributed",
    ),
    "weak_same_lane_refusal": (
        "election_id",
        "region_id",
        "donor_slot",
        "recipient_slots",
        "candidate_ballot_recent_base",
        "floor_mode",
        "recipient_weight_mode",
        "protected_floor",
        "before",
        "reservoir",
        "gain",
        "transfer",
        "after",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_path(version: str) -> Path:
    paths = {"v23": V23_CONFIG, "v24": V24_CONFIG, "v25": V25_CONFIG}
    if version not in paths:
        raise ValueError(f"unsupported prospective model version: {version}")
    path = paths[version]
    if not path.exists():
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        raise RuntimeError(f"{version} model config is unavailable at {display_path}")
    return path


def _historical_results_path(version: str) -> Path:
    return (
        active_v24.V24_DATA / "presidential_results_standardized.csv"
        if version in {"v24", "v25"}
        else RESULTS
    )


def _historical_conversion_path(version: str) -> Path:
    if version in {"v24", "v25"}:
        return active_v24.V24_DATA / "candidate_vote_conversion_context.csv"
    return CANDIDATE_CONVERSION_HISTORY


def _historical_speech_context_path(version: str) -> Path:
    if version in {"v24", "v25"}:
        return active_v24.V24_DATA / "candidate_party_speech_context.csv"
    return FORECAST_CANDIDATE_CONTEXT_DIR / "candidate_party_speech_context.csv"


def _runtime_policy_path(version: str, declared_config_path: Path) -> Path:
    if version in {"v23", "v25"}:
        return declared_config_path
    defaults = active.load_policy.__defaults__ or ()
    if not defaults:
        raise RuntimeError("active policy loader has no default runtime path")
    return Path(defaults[0])


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


def _assert_target_input_coverage(
    selected: pd.DataFrame,
    salience: pd.DataFrame,
    candidate_link: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> None:
    if salience.empty:
        raise RuntimeError("forecast issue salience is empty")
    selected_ids = set(selected["candidate_id"].astype(str))
    linked_ids = set(candidate_link["candidate_id"].astype(str))
    missing = sorted(selected_ids - linked_ids)
    if missing:
        raise RuntimeError(f"selected candidates lack issue-link rows: {missing}")
    for name, frame in (("salience", salience), ("candidate_link", candidate_link)):
        available = pd.to_datetime(frame["available_date"], errors="coerce")
        if available.isna().any() or available.gt(cutoff).any():
            raise RuntimeError(f"forecast {name} crosses the D-1 cutoff")
    selected_link = candidate_link.loc[
        candidate_link["candidate_id"].astype(str).isin(selected_ids)
    ]
    directional = pd.to_numeric(
        selected_link.get("emphasis_within", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    if not directional.abs().gt(0.0).any():
        raise RuntimeError("selected candidates have no signed issue-link evidence")


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


def _augment_current_government_context(
    version: str,
    registry: pd.DataFrame,
    selected: pd.DataFrame,
    base_candidate_link: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    """Link current-Assembly government evidence to the prior winner's nominee.

    The explicit-target extractor deliberately leaves ``government`` rows
    unassigned because it does not read election results.  At forecast time the
    governing party is nevertheless observable from the latest *prior*
    presidential result.  This bridge makes that PIT-safe identity link, while
    leaving every model formula and coefficient untouched.
    """

    cutoff = pd.Timestamp(FORECAST_CUTOFF)
    results = pd.read_csv(_historical_results_path(version), encoding="utf-8-sig")
    prior_elections = sorted(
        {
            str(election_id)
            for election_id in results["election_id"].astype(str).unique()
            if election_id in ELECTION_DATES
            and pd.Timestamp(ELECTION_DATES[election_id])
            < pd.Timestamp(ELECTION_DATES[TARGET_ELECTION])
        },
        key=lambda election_id: pd.Timestamp(ELECTION_DATES[election_id]),
    )
    if not prior_elections:
        raise RuntimeError("no prior presidential result can identify the government")
    prior_election = prior_elections[-1]
    prior = results.loc[results["election_id"].astype(str).eq(prior_election)].copy()
    prior["votes"] = pd.to_numeric(prior["votes"], errors="raise")
    winner = (
        prior.groupby(["candidate_name", "party_name"], as_index=False)["votes"]
        .sum()
        .sort_values(["votes", "candidate_name"], ascending=[False, True])
        .iloc[0]
    )
    governing_party = str(winner["party_name"])
    nominee = selected.loc[selected["party_name"].astype(str).eq(governing_party)].copy()
    if len(nominee) != 1:
        raise RuntimeError(
            "latest prior presidential winner party does not identify exactly one "
            "selected 2025 candidate"
        )
    nominee_row = nominee.iloc[0]
    registry_nominee = registry.loc[
        registry["candidate_id"].astype(str).eq(str(nominee_row["candidate_id"]))
    ]
    if len(registry_nominee) != 1:
        raise RuntimeError("governing nominee is not unique in the candidate registry")
    registry_nominee_row = registry_nominee.iloc[0]

    explicit = pd.read_csv(EXPLICIT_TARGET_CONTEXT, encoding="utf-8-sig")
    observed = pd.to_datetime(explicit["available_date"], errors="coerce")
    if observed.isna().any() or observed.gt(cutoff).any():
        raise RuntimeError("explicit government context crosses the D-1 cutoff")
    assembly_number = pd.to_numeric(explicit["assembly_daesu"], errors="raise")
    current_assembly = int(assembly_number.max())
    government = explicit.loc[
        explicit["target_type"].astype(str).eq("government")
        & assembly_number.eq(current_assembly)
    ].copy()
    if government.empty:
        raise RuntimeError("current Assembly has no explicit government target evidence")

    candidate_target = pd.read_csv(CANDIDATE_TARGET_CONTEXT, encoding="utf-8-sig")
    existing_government = candidate_target["source_target_type"].astype(str).eq(
        "government"
    )
    if existing_government.any():
        raise RuntimeError("candidate target context already contains government links")
    registry_available = pd.Timestamp(registry_nominee_row["available_date"])
    government["candidate_id"] = str(registry_nominee_row["candidate_id"])
    government["candidate_name"] = str(registry_nominee_row["candidate_name"])
    government["candidate_party_name"] = str(registry_nominee_row["party_name"])
    government["candidate_ballot_number"] = int(registry_nominee_row["ballot_number"])
    government["candidate_registry_available_date"] = registry_available.strftime(
        "%Y-%m-%d"
    )
    government["candidate_link_eligible"] = True
    government["candidate_linkage_basis"] = (
        "current_assembly_government_to_latest_prior_presidential_winner_party"
    )
    government["source_target_type"] = government["target_type"]
    government["source_target_name"] = government["target_name"]
    government["source_observed_available_date"] = government["available_date"]
    government["available_date"] = pd.concat(
        [
            pd.to_datetime(government["available_date"], errors="raise"),
            pd.Series(registry_available, index=government.index),
        ],
        axis=1,
    ).max(axis=1).dt.strftime("%Y-%m-%d")
    government = government.reindex(columns=candidate_target.columns)
    augmented_target = pd.concat(
        [candidate_target, government], ignore_index=True, sort=False
    )

    # Government-target rows describe the incumbent administration, not the
    # nominee's own speech attention or personal candidate strength. Keep the
    # original person/party candidate link for candidate-level features, and
    # expose the augmented target table only to the issue-character bridge.
    # Rebuilding this link from ``augmented_target`` counted 11k+ government
    # sentences as nominee attention and contaminated the preliminary tier.
    selected_ids = set(selected["candidate_id"].astype(str))
    model_link = base_candidate_link.loc[
        base_candidate_link["candidate_id"].astype(str).isin(selected_ids)
    ].copy()
    if set(model_link["candidate_id"].astype(str)) != selected_ids:
        raise RuntimeError("person/party candidate link coverage drifted")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "candidate_target_context_weekly.csv"
    link_path = output_dir / "model_candidate_issue_link.csv"
    augmented_target.to_csv(
        target_path, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    model_link.to_csv(
        link_path, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    directional = pd.to_numeric(
        government["absolute_directional_weight"], errors="coerce"
    ).fillna(0.0)
    diagnostics = {
        "method": "current_assembly_government_to_latest_prior_winner_party",
        "prior_election": prior_election,
        "prior_winner_candidate": str(winner["candidate_name"]),
        "prior_winner_party": governing_party,
        "mapped_candidate_id": str(registry_nominee_row["candidate_id"]),
        "mapped_candidate_name": str(registry_nominee_row["candidate_name"]),
        "mapped_slot": str(nominee_row["slot"]),
        "current_assembly": current_assembly,
        "government_aggregate_rows": int(len(government)),
        "government_sentence_count": int(
            pd.to_numeric(government["sentence_count"], errors="coerce").fillna(0).sum()
        ),
        "directional_aggregate_rows": int(directional.gt(0.0).sum()),
        "signed_weight": float(
            pd.to_numeric(government["signed_weight"], errors="coerce").fillna(0.0).sum()
        ),
        "absolute_directional_weight": float(directional.sum()),
        "candidate_attention_source_types": ["person", "party"],
        "government_evidence_destination": "issue_character_burden_only",
        "government_rows_excluded_from_candidate_attention": int(len(government)),
        "candidate_issue_link_rows": int(len(model_link)),
        "target_outcomes_used": False,
        "prior_outcome_used_only_for_governing_party_identity": True,
        "forecast_cutoff": FORECAST_CUTOFF,
    }
    return model_link, target_path, diagnostics


def _build_target_candidate_context(
    temp: Path,
    registry: pd.DataFrame,
    selected: pd.DataFrame,
    candidate_link: pd.DataFrame,
    *,
    version: str,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, object]]:
    augmented_link, target_context_path, diagnostics = (
        _augment_current_government_context(
            version,
            registry,
            selected,
            candidate_link,
            temp / "government_linked_inputs",
        )
    )
    link_path = target_context_path.parent / "model_candidate_issue_link.csv"
    ballot_to_slot = {
        int(row.ballot_number): str(row.slot)
        for row in selected[["ballot_number", "slot"]].itertuples(index=False)
    }
    if set(ballot_to_slot.values()) != {"A", "B", "C"}:
        raise RuntimeError("selected candidates do not define exactly three model slots")
    output_dir = temp / "candidate_context_v2"
    with issue_context_builder.patched(
        [
            (issue_context_builder, "FORECAST_LINKS", link_path),
            (
                issue_context_builder,
                "FORECAST_CANDIDATE_TARGET_CONTEXT",
                target_context_path,
            ),
            (issue_context_builder, "PRES_2025_BALLOT_TO_SLOT", ballot_to_slot),
        ]
    ):
        built = context_builder.build_context(
            output_dir=output_dir,
            assembly_matches=issue_context_builder.DEFAULT_MATCHES,
            candidates=REGISTRY,
            speaker_profile=(
                CONTEXT_DIR / "assembly_speaker_influence_pres_2025.csv"
            ),
        )

    # Candidate/party target evidence and government-responsibility evidence
    # have different consumers.  The augmented character overlay above is
    # retained for the incumbent-burden compiler.  Rebuild only the direct
    # candidate profile from the original person/party target table so that a
    # government row cannot also become candidate-strength evidence.
    direct_output_dir = temp / "candidate_direct_context_v2"
    with issue_context_builder.patched(
        [
            (issue_context_builder, "FORECAST_LINKS", link_path),
            (
                issue_context_builder,
                "FORECAST_CANDIDATE_TARGET_CONTEXT",
                CANDIDATE_TARGET_CONTEXT,
            ),
            (issue_context_builder, "PRES_2025_BALLOT_TO_SLOT", ballot_to_slot),
        ]
    ):
        direct_built = context_builder.build_context(
            output_dir=direct_output_dir,
            assembly_matches=issue_context_builder.DEFAULT_MATCHES,
            candidates=REGISTRY,
            speaker_profile=(
                CONTEXT_DIR / "assembly_speaker_influence_pres_2025.csv"
            ),
        )
    paths = {
        "context_dir": output_dir,
        "candidate_context": Path(built["conversion"]),
        "candidate_party_speech_context": Path(built["speech"]),
        "candidate_party_tone_gap": Path(built["tone"]),
        "candidate_public_treatment": Path(built["treatment"]),
        "political_landscape": output_dir / "candidate_political_landscape.csv",
        "third_candidate_profile": (
            output_dir / "auto_candidate_role" / "third_candidate_profile.csv"
        ),
        "candidate_issue_profile": Path(built["profile"]),
        "candidate_direct_issue_profile": Path(direct_built["profile"]),
        "mega_issue_axis": Path(built["axis"]),
        "mega_issue_attribution": Path(built["attribution"]),
    }
    diagnostics["model_slot_by_ballot"] = {
        str(ballot): slot for ballot, slot in sorted(ballot_to_slot.items())
    }
    diagnostics["generated_context_output_rows"] = built["manifest"]["outputs"]
    diagnostics["generated_candidate_direct_profile_rows"] = int(
        direct_built["manifest"]["outputs"]["candidate_issue_profile.csv"]
    )

    burden_profile = pd.read_csv(paths["candidate_issue_profile"], encoding="utf-8-sig")
    direct_profile = pd.read_csv(
        paths["candidate_direct_issue_profile"], encoding="utf-8-sig"
    )
    burden_target = burden_profile.loc[
        burden_profile["election_id"].astype(str).eq(TARGET_ELECTION)
    ].copy()
    direct_target = direct_profile.loc[
        direct_profile["election_id"].astype(str).eq(TARGET_ELECTION)
    ].copy()
    burden_government = burden_target["target_source_types"].fillna("").astype(str).str.contains(
        r"(?:^|\|)government(?:\||$)", regex=True
    )
    direct_government = direct_target["target_source_types"].fillna("").astype(str).str.contains(
        r"(?:^|\|)government(?:\||$)", regex=True
    )
    if not burden_government.any():
        raise RuntimeError("government-burden profile lost government target evidence")
    if direct_government.any():
        raise RuntimeError("candidate direct profile contains government target evidence")
    diagnostics["government_burden_profile_government_rows"] = int(
        burden_government.sum()
    )
    diagnostics["candidate_direct_profile_government_rows"] = int(
        direct_government.sum()
    )
    diagnostics["generated_context_sha256"] = {
        key: _sha256(path)
        for key, path in paths.items()
        if key != "context_dir"
    }
    return augmented_link, paths, diagnostics


def _candidate_strength_context(
    selected: pd.DataFrame,
    candidate_link: pd.DataFrame,
    *,
    historical_context_path: Path | None = None,
    target_context_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load direct 2025 candidate context or use the historical ridge fallback.

    Direct rows take precedence when all selected candidates have point-in-time
    context. Otherwise, a small ridge model maps issue-link attention to the
    frozen, outcome-free candidate-weight scale using elections through 2022.
    Neither path fits or reads the forecast-election result.
    """

    history_path = historical_context_path or CANDIDATE_CONVERSION_HISTORY
    historical_context = pd.read_csv(history_path, encoding="utf-8-sig")
    historical_context = historical_context.loc[
        ~historical_context["election_id"].astype(str).eq(TARGET_ELECTION)
    ].copy()
    direct_context_path = target_context_path or CANDIDATE_CONVERSION_HISTORY
    target_context = pd.read_csv(direct_context_path, encoding="utf-8-sig")
    direct = target_context.loc[
        target_context["election_id"].astype(str).eq(TARGET_ELECTION)
    ].copy()
    selected_names = set(selected["candidate_name"].astype(str))
    direct_names = set(direct["candidate_name"].astype(str))
    if len(direct) == len(selected) and direct_names == selected_names:
        observed = pd.to_datetime(direct["available_date"], errors="coerce")
        if observed.isna().any() or observed.gt(pd.Timestamp(FORECAST_CUTOFF)).any():
            raise RuntimeError("direct candidate context is not available by forecast cutoff")
        direct = direct.drop(columns=["slot"]).merge(
            selected[["candidate_id", "candidate_name", "slot"]],
            on="candidate_name",
            how="inner",
            validate="one_to_one",
        )
        direct = direct.reindex(columns=[*historical_context.columns, "candidate_id"])
        combined = pd.concat(
            [
                historical_context,
                direct[historical_context.columns],
            ],
            ignore_index=True,
        )
        try:
            context_source = str(direct_context_path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            context_source = "generated_prospective_candidate_context"
        diagnostics = {
            "method": "direct_speech_derived_candidate_context",
            "training_elections": [],
            "target_outcomes_used": False,
            "polling_used": False,
            "source": context_source,
            "projected_candidates": direct[
                [
                    "candidate_id",
                    "candidate_name",
                    "slot",
                    "candidate_weight",
                    "confidence",
                    "available_date",
                ]
            ].to_dict("records"),
        }
        return combined, diagnostics

    historical_link = pd.read_csv(CANDIDATE_LINK_HISTORY, encoding="utf-8-sig")

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


def _combine_historical_and_target_rows(
    historical_path: Path,
    target_path: Path,
    *,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    historical = pd.read_csv(historical_path, encoding="utf-8-sig")
    historical = historical.loc[
        ~historical["election_id"].astype(str).eq(TARGET_ELECTION)
    ].copy()
    target = pd.read_csv(target_path, encoding="utf-8-sig")
    target = target.loc[target["election_id"].astype(str).eq(TARGET_ELECTION)].copy()
    expected = set(selected["candidate_name"].astype(str))
    observed = set(target["candidate_name"].astype(str))
    if observed != expected:
        raise RuntimeError(
            f"target context candidate coverage mismatch: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    missing = sorted(set(historical.columns) - set(target.columns))
    if missing:
        raise RuntimeError(f"target context is missing historical columns: {missing}")
    return pd.concat(
        [historical, target.reindex(columns=historical.columns)],
        ignore_index=True,
        sort=False,
    )


def _historical_compatible_target_matches(
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rebuild target issue matches at the historical speech-row granularity.

    The 16th-22nd Assembly archive matcher receives one complete speech row.
    The prospective stance source stores one row per issue per sentence.  Using
    those already-collapsed issue labels directly loses context when, for
    example, an issue term is in one sentence and impeachment or martial-law
    responsibility language is in another sentence of the same speech.  This
    adapter reconstructs each original speech row and runs the same matcher,
    term weights, boosts, and context rules as the historical extractor.
    """

    required = {
        "source_id",
        "source_row_id",
        "sentence_index",
        "period",
        "speaker",
        "text_excerpt",
    }
    missing = sorted(required - set(target.columns))
    if missing:
        raise RuntimeError(
            f"official target source cannot reconstruct historical matches: {missing}"
        )

    sentences = target[list(required)].copy()
    sentences["sentence_index"] = pd.to_numeric(
        sentences["sentence_index"], errors="coerce"
    )
    sentences = sentences.loc[
        sentences["sentence_index"].notna()
        & sentences["text_excerpt"].astype(str).str.strip().ne("")
    ].copy()
    sentences = (
        sentences.sort_values(["source_id", "source_row_id", "sentence_index"])
        .drop_duplicates(
            ["source_id", "source_row_id", "sentence_index"], keep="first"
        )
        .reset_index(drop=True)
    )
    if sentences.empty:
        raise RuntimeError("official target source has no reconstructable speech rows")

    keyword_maps, term_weights, issue_boosts, context_rules = (
        assembly_match_builder.build_keyword_inputs()
    )
    rows: list[dict[str, object]] = []
    for _, speech in sentences.groupby(
        ["source_id", "source_row_id"], sort=False
    ):
        speech = speech.sort_values("sentence_index")
        full_text = " ".join(speech["text_excerpt"].astype(str))
        weights = match_issue_weights(
            full_text,
            keyword_maps[TARGET_ELECTION],
            term_weights=term_weights.get(TARGET_ELECTION),
            issue_boosts=issue_boosts.get(TARGET_ELECTION),
            context_rules=context_rules.get(TARGET_ELECTION),
        )
        first = speech.iloc[0]
        for issue_name, issue_weight in weights.items():
            rows.append(
                {
                    "election_id": TARGET_ELECTION,
                    "period": first["period"],
                    "speaker": first["speaker"],
                    "issue_name": issue_name,
                    "issue_weight": float(issue_weight),
                    # Historical 16th-22nd extraction emits one issue row per
                    # speech row and therefore uses one matched unit here.
                    "matched_term_count": 1,
                }
            )
    matches = pd.DataFrame(rows)
    if matches.empty:
        raise RuntimeError("historical-compatible target issue rematch is empty")
    diagnostics = {
        "sentence_issue_rows": int(len(target)),
        "unique_sentence_rows": int(len(sentences)),
        "reconstructed_speech_rows": int(
            sentences[["source_id", "source_row_id"]].drop_duplicates().shape[0]
        ),
        "historical_compatible_match_rows": int(len(matches)),
    }
    return matches, diagnostics


def _automatic_target_mega_controls(
    temp: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    """Build V25 target shock controls from PIT-safe official proceedings.

    The historical automatic controls receive complete speech rows, so the
    target's sentence-level source is first reconstructed to that granularity.
    A dated official institutional proceeding then supplies the same universal
    categorical event identity used by the class-level shock scale.  Frequency
    diagnostics remain in the audit rather than being discarded.
    """

    source = pd.read_csv(OFFICIAL_2025_MINUTES, encoding="utf-8-sig").fillna("")
    lowered = {str(column).strip().casefold() for column in source.columns}
    forbidden = sorted(lowered & {value.casefold() for value in FORBIDDEN_OUTCOME_COLUMNS})
    forbidden.extend(
        sorted(
            column
            for column in lowered
            if "actual" in column or "vote_share" in column or "winner" in column
        )
    )
    if forbidden:
        raise RuntimeError(
            f"official target mega-control source contains outcome columns: {sorted(set(forbidden))}"
        )
    required = {
        "election_id",
        "assembly_daesu",
        "source_id",
        "source_row_id",
        "sentence_index",
        "meeting_date",
        "available_date",
        "period",
        "speaker",
        "committee",
        "agenda",
        "issue_name",
        "issue_weight",
        "text_excerpt",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"official target mega-control source is missing: {missing}")

    meeting = pd.to_datetime(source["meeting_date"], errors="coerce")
    available = pd.to_datetime(source["available_date"], errors="coerce")
    cutoff = pd.Timestamp(FORECAST_CUTOFF)
    election_date = pd.Timestamp(ELECTION_DATES[TARGET_ELECTION])
    eligible = (
        source["election_id"].astype(str).eq(TARGET_ELECTION)
        & source["assembly_daesu"].astype(str).eq("22")
        & meeting.notna()
        & available.notna()
        & meeting.lt(election_date)
        & available.le(cutoff)
    )
    target = source.loc[eligible].copy()
    if target.empty:
        raise RuntimeError("official target mega-control source has no PIT-eligible rows")

    matches, match_diagnostics = _historical_compatible_target_matches(target)
    _, diagnostics = build_automatic_mega_issue_intensity(matches, ELECTION_DATES)
    taxonomy, intensity, audit = build_automatic_mega_taxonomy(diagnostics)
    if len(taxonomy) != 1 or len(intensity) != 1:
        raise RuntimeError("automatic target mega controls did not produce one target row")

    rules = pd.read_csv(ISSUE_CONTEXT_RULES, encoding="utf-8-sig").fillna("")
    mega_rules = rules.loc[
        rules["rule_id"].astype(str).str.startswith("mega_")
        & rules["target_issue"].astype(str).eq("regime_change")
    ].copy()
    crisis_terms = sorted(
        {
            term.strip()
            for value in mega_rules["context_terms"].astype(str)
            for term in value.split("|")
            if term.strip()
        }
    )
    if not crisis_terms:
        raise RuntimeError("automatic target mega-control vocabulary is empty")
    title = target["committee"].astype(str) + " " + target["agenda"].astype(str)
    title_has_crisis = title.apply(lambda value: any(term in value for term in crisis_terms))
    institutional_markers = ("국정조사", "특별위원회", "탄핵소추", "헌법재판")
    title_is_official_proceeding = title.apply(
        lambda value: any(marker in value for marker in institutional_markers)
    )
    institutional = target.loc[title_has_crisis & title_is_official_proceeding].copy()
    semantic_gate = not institutional.empty
    frequency_shock_type = str(taxonomy.iloc[0]["shock_type"])
    frequency_intensity = float(intensity.iloc[0]["mega_issue_intensity"])
    selected_shock_type = frequency_shock_type
    selected_intensity = frequency_intensity
    semantic_adjustment_applied = False
    if semantic_gate:
        selected_shock_type = "institutional_crisis"
        selected_intensity = SHOCK_CLASS_INTENSITY[selected_shock_type]
        # The gate asserts the class whether or not the frequency path already
        # reached it, but the reported flag must not claim credit for a no-op:
        # once pres_2025 crisis vocabulary is registered the frequency path
        # arrives at the same class and intensity on its own, and a flag that
        # still reads True hides that the gate has stopped deciding anything.
        semantic_adjustment_applied = (
            selected_shock_type != frequency_shock_type
            or selected_intensity != frequency_intensity
        )
        taxonomy.loc[:, "shock_type"] = selected_shock_type
        taxonomy.loc[:, "notes"] = (
            "Assembly speech-row rematch plus universal official-proceeding "
            "semantic event class; no election outcome"
        )
        intensity.loc[:, "mega_issue_intensity"] = selected_intensity
        intensity.loc[:, "notes"] = (
            "Universal institutional-crisis class intensity after historical-"
            "compatible Assembly speech-row rematch"
        )

    audit["frequency_automatic_shock_type"] = frequency_shock_type
    audit["frequency_automatic_mega_issue_intensity"] = frequency_intensity
    audit["selected_shock_type"] = selected_shock_type
    audit["selected_mega_issue_intensity"] = selected_intensity
    audit["semantic_institutional_gate"] = semantic_gate
    audit["semantic_gate_adjustment_applied"] = semantic_adjustment_applied
    audit["semantic_source_rows"] = int(len(institutional))
    audit["semantic_source_meetings"] = int(institutional["source_id"].nunique())
    audit["semantic_source_speakers"] = int(institutional["speaker"].nunique())
    audit["semantic_vocabulary_source"] = ISSUE_CONTEXT_RULES.relative_to(ROOT).as_posix()
    for key, value in match_diagnostics.items():
        audit[key] = value
    audit["target_match_granularity"] = "reconstructed_historical_speech_row"
    audit["outcome_columns_used"] = ""

    historical_intensity = pd.read_csv(
        AUTOMATIC_DIR / "mega_issue_intensity.csv", encoding="utf-8-sig"
    )
    historical_taxonomy = pd.read_csv(
        AUTOMATIC_DIR / "mega_issue_taxonomy.csv", encoding="utf-8-sig"
    )
    combined_intensity = pd.concat(
        [
            historical_intensity.loc[
                ~historical_intensity["election_id"].astype(str).eq(TARGET_ELECTION)
            ],
            intensity.reindex(columns=historical_intensity.columns),
        ],
        ignore_index=True,
    )
    combined_taxonomy = pd.concat(
        [
            historical_taxonomy.loc[
                ~historical_taxonomy["election_id"].astype(str).eq(TARGET_ELECTION)
            ],
            taxonomy.reindex(columns=historical_taxonomy.columns),
        ],
        ignore_index=True,
    )
    for name, frame in {
        "mega_issue_intensity": combined_intensity,
        "mega_issue_taxonomy": combined_taxonomy,
    }.items():
        target_rows = frame.loc[frame["election_id"].astype(str).eq(TARGET_ELECTION)]
        dates = pd.to_datetime(target_rows["available_date"], errors="coerce")
        if len(target_rows) != 1 or dates.isna().any() or dates.gt(cutoff).any():
            raise RuntimeError(f"{name} failed target coverage or PIT validation")

    paths = {
        "mega_issue_intensity": temp / "mega_issue_intensity.csv",
        "mega_issue_taxonomy": temp / "mega_issue_taxonomy.csv",
        "mega_issue_taxonomy_audit": temp / "mega_issue_taxonomy_audit.csv",
    }
    combined_intensity.to_csv(paths["mega_issue_intensity"], index=False, encoding="utf-8-sig")
    combined_taxonomy.to_csv(paths["mega_issue_taxonomy"], index=False, encoding="utf-8-sig")
    audit.to_csv(paths["mega_issue_taxonomy_audit"], index=False, encoding="utf-8-sig")
    metadata = {
        "method": (
            "historical_speech_row_rematch_with_official_institutional_event_gate"
        ),
        "source": OFFICIAL_2025_MINUTES.relative_to(ROOT).as_posix(),
        "source_rows": int(len(source)),
        "pit_eligible_rows": int(len(target)),
        "semantic_gate": semantic_gate,
        "semantic_gate_adjustment_applied": semantic_adjustment_applied,
        "semantic_source_rows": int(len(institutional)),
        "semantic_source_meetings": int(institutional["source_id"].nunique()),
        "semantic_source_speakers": int(institutional["speaker"].nunique()),
        "frequency_shock_type": frequency_shock_type,
        "frequency_mega_issue_intensity": frequency_intensity,
        "shock_type": selected_shock_type,
        "mega_issue_intensity": selected_intensity,
        "available_date": str(intensity.iloc[0]["available_date"]),
        "target_outcomes_used": False,
        "model_parameters_changed": False,
        **match_diagnostics,
    }
    return paths, metadata


def _prospective_sources(
    temp: Path,
    registry: pd.DataFrame,
    selected: pd.DataFrame,
    salience: pd.DataFrame,
    candidate_link: pd.DataFrame,
    *,
    version: str,
) -> tuple[dict[str, Path], dict[str, object]]:
    mega_control_paths: dict[str, Path] = {}
    mega_control_diagnostics: dict[str, object] = {
        "method": "historical_runtime_default",
        "target_outcomes_used": False,
        "model_parameters_changed": False,
    }
    if version == "v25":
        mega_control_paths, mega_control_diagnostics = (
            _automatic_target_mega_controls(temp)
        )
    candidate_link, target_context_paths, government_link_diagnostics = (
        _build_target_candidate_context(
            temp,
            registry,
            selected,
            candidate_link,
            version=version,
        )
    )
    _assert_target_input_coverage(
        selected,
        salience,
        candidate_link,
        pd.Timestamp(FORECAST_CUTOFF),
    )
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
        [
            pd.read_csv(_historical_results_path(version), encoding="utf-8-sig"),
            skeleton,
        ],
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
        selected,
        candidate_link,
        historical_context_path=_historical_conversion_path(version),
        target_context_path=target_context_paths["candidate_context"],
    )
    combined_speech_context = _combine_historical_and_target_rows(
        _historical_speech_context_path(version),
        target_context_paths["candidate_party_speech_context"],
        selected=selected,
    )
    historical_landscape = pd.read_csv(
        AUTOMATIC_DIR / "candidate_political_landscape.csv",
        encoding="utf-8-sig",
    )
    target_landscape = pd.read_csv(
        target_context_paths["political_landscape"],
        encoding="utf-8-sig",
    )
    target_landscape = target_landscape.loc[
        target_landscape["election_id"].astype(str).eq(TARGET_ELECTION)
    ].copy()
    if set(target_landscape["candidate_name"].astype(str)) != set(
        selected["candidate_name"].astype(str)
    ):
        raise RuntimeError("forecast candidate landscape does not match selected candidates")
    combined_landscape = pd.concat(
        [
            historical_landscape.loc[
                ~historical_landscape["election_id"].astype(str).eq(TARGET_ELECTION)
            ],
            target_landscape[historical_landscape.columns],
        ],
        ignore_index=True,
    )
    historical_third_profile = pd.read_csv(
        AUTOMATIC_DIR / "third_candidate_profile.csv",
        encoding="utf-8-sig",
    )
    target_third_profile = pd.read_csv(
        target_context_paths["third_candidate_profile"],
        encoding="utf-8-sig",
    )
    target_third_profile = target_third_profile.loc[
        target_third_profile["election_id"].astype(str).eq(TARGET_ELECTION)
    ].copy()
    expected_third = set(
        selected.loc[selected["slot"].astype(str).eq("C"), "candidate_name"].astype(str)
    )
    observed_third = set(target_third_profile["candidate_name"].astype(str))
    if observed_third != expected_third:
        raise RuntimeError(
            "forecast third-candidate profile does not match the selected C candidate"
        )
    combined_third_profile = pd.concat(
        [
            historical_third_profile.loc[
                ~historical_third_profile["election_id"].astype(str).eq(
                    TARGET_ELECTION
                )
            ],
            target_third_profile[historical_third_profile.columns],
        ],
        ignore_index=True,
    )
    combined_issue_seeds: dict[str, pd.DataFrame] = {}
    for filename in (
        "candidate_issue_profile.csv",
        "mega_issue_axis.csv",
        "mega_issue_attribution.csv",
    ):
        historical_seed = pd.read_csv(
            ROOT / "data/raw/auto_issue_seed" / filename,
            encoding="utf-8-sig",
        )
        target_seed = pd.read_csv(
            target_context_paths[filename.removesuffix(".csv")],
            encoding="utf-8-sig",
        )
        target_seed = target_seed.loc[
            target_seed["election_id"].astype(str).eq(TARGET_ELECTION)
        ].copy()
        if target_seed.empty:
            raise RuntimeError(f"forecast issue seed is empty: {filename}")
        if filename == "candidate_issue_profile.csv":
            observed_candidates = set(target_seed["candidate_name"].astype(str))
            expected_candidates = set(selected["candidate_name"].astype(str))
            if observed_candidates != expected_candidates:
                raise RuntimeError(
                    "forecast candidate issue profile does not cover selected candidates"
                )
        combined_issue_seeds[filename] = pd.concat(
            [
                historical_seed.loc[
                    ~historical_seed["election_id"].astype(str).eq(TARGET_ELECTION)
                ],
                target_seed,
            ],
            ignore_index=True,
            sort=False,
        )

    historical_direct_profile = pd.read_csv(
        ROOT / "data/raw/auto_issue_seed/candidate_issue_profile.csv",
        encoding="utf-8-sig",
    )
    target_direct_profile = pd.read_csv(
        target_context_paths["candidate_direct_issue_profile"],
        encoding="utf-8-sig",
    )
    target_direct_profile = target_direct_profile.loc[
        target_direct_profile["election_id"].astype(str).eq(TARGET_ELECTION)
    ].copy()
    direct_government = target_direct_profile["target_source_types"].fillna("").astype(str).str.contains(
        r"(?:^|\|)government(?:\||$)", regex=True
    )
    if target_direct_profile.empty or direct_government.any():
        raise RuntimeError("forecast direct candidate profile is empty or government-linked")
    combined_direct_profile = pd.concat(
        [
            historical_direct_profile.loc[
                ~historical_direct_profile["election_id"].astype(str).eq(
                    TARGET_ELECTION
                )
            ],
            target_direct_profile,
        ],
        ignore_index=True,
        sort=False,
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
        "candidate_party_speech_context": temp / "candidate_party_speech_context.csv",
        "candidate_party_tone_gap": temp / "candidate_party_tone_gap.csv",
        "candidate_public_treatment": temp / "candidate_public_treatment.csv",
        "political_landscape": temp / "candidate_political_landscape.csv",
        "third_candidate_profile": temp / "third_candidate_profile.csv",
        "candidate_issue_profile": temp / "candidate_issue_profile.csv",
        "candidate_direct_issue_profile": temp / "candidate_direct_issue_profile.csv",
        "mega_issue_axis": temp / "mega_issue_axis.csv",
        "mega_issue_attribution": temp / "mega_issue_attribution.csv",
        "outer_config": temp / "nested_outer_with_prospective_target.csv",
    }
    paths.update(mega_control_paths)
    for key, frame in (
        ("results", results),
        ("salience", combined_salience),
        ("link", combined_link),
        ("candidate_context", candidate_context),
        ("candidate_party_speech_context", combined_speech_context),
        (
            "candidate_party_tone_gap",
            pd.read_csv(
                target_context_paths["candidate_party_tone_gap"],
                encoding="utf-8-sig",
            ),
        ),
        (
            "candidate_public_treatment",
            pd.read_csv(
                target_context_paths["candidate_public_treatment"],
                encoding="utf-8-sig",
            ),
        ),
        ("political_landscape", combined_landscape),
        ("third_candidate_profile", combined_third_profile),
        ("candidate_issue_profile", combined_issue_seeds["candidate_issue_profile.csv"]),
        ("candidate_direct_issue_profile", combined_direct_profile),
        ("mega_issue_axis", combined_issue_seeds["mega_issue_axis.csv"]),
        (
            "mega_issue_attribution",
            combined_issue_seeds["mega_issue_attribution.csv"],
        ),
        ("outer_config", outer),
    ):
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    return paths, {
        "candidate_strength": candidate_context_diagnostics,
        "government_context_link": government_link_diagnostics,
        "mega_issue_controls": mega_control_diagnostics,
    }


def _prior_region_volume(version: str = "v23") -> pd.Series:
    results = pd.read_csv(_historical_results_path(version), encoding="utf-8-sig")
    prior = results.loc[results["election_id"].eq("pres_2022")].copy()
    return pd.to_numeric(prior["votes"], errors="coerce").fillna(0.0).groupby(
        prior["region_id"]
    ).sum()


def _target_base(
    target: pd.DataFrame,
    historical_base: pd.DataFrame,
    version: str = "v23",
) -> pd.DataFrame:
    out = target.copy()
    prior_volume = _prior_region_volume(version)
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

    engines = {
        active.nested.engine,
        active.assignment_builder.engine,
        active.nested.base_eval.engine,
        assignment_builder.engine,
    }
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


@contextmanager
def _model_runtime(
    version: str,
    config_path: Path,
    selected: pd.DataFrame,
) -> Iterator[None]:
    """Install the exact promoted historical runtime for the requested version."""

    if version == "v23":
        with _v23_runtime(config_path, selected):
            yield
        return

    if version == "v25":
        with active_v25.corrected_runtime(
            active,
            assignment_builder,
            active.nested,
            active.nested.base_eval,
            repairs=active_v25.RUNTIME_REPAIRS,
        ):
            yield
        return

    exclusions = active_v24.scored_exclusions()
    engines = {
        active.nested.engine,
        active.assignment_builder.engine,
        active.nested.base_eval.engine,
        assignment_builder.engine,
    }
    attributes: list[tuple[object, str, object]] = [
        (active, "CONFIG_PATH", config_path),
        (
            active.nested,
            "ASSIGNMENT_PATH",
            active_v24.V24_DATA / "candidate_slot_assignments_v2.csv",
        ),
        (active.nested.base_eval, "BASELINE_PATH", active_v24.V24_BASELINE),
    ]
    for engine in engines:
        attributes.extend(
            [
                (
                    engine,
                    "RESULTS",
                    str(active_v24.V24_DATA / "presidential_results_standardized.csv"),
                ),
                (
                    engine,
                    "COALITION_EVENTS",
                    str(active_v24.V24_DATA / "coalition_events.csv"),
                ),
                # The official V24 wrapper leaves the generic registry empty so
                # the ballot-faithful empty coalition table remains authoritative.
                (engine, "WITHDRAWAL_TRANSFER_REGISTRY", ""),
                (
                    engine,
                    "CANDIDATE_PARTY_SPEECH_CONTEXT",
                    str(active_v24.V24_DATA / "candidate_party_speech_context.csv"),
                ),
                (
                    engine,
                    "CANDIDATE_VOTE_CONVERSION_CONTEXT",
                    str(active_v24.V24_DATA / "candidate_vote_conversion_context.csv"),
                ),
                (
                    engine,
                    "_load_scored_contest_scope_exclusions",
                    lambda _excluded=exclusions: set(_excluded),
                ),
            ]
        )

    original_rows = assignment_builder._all_ballot_rows
    original_redistribution = assignment_builder._apply_withdrawal_redistribution
    active_v24.install_ballot_patches(assignment_builder)
    try:
        with patching.patched(attributes):
            yield
    finally:
        assignment_builder._all_ballot_rows = original_rows
        assignment_builder._apply_withdrawal_redistribution = original_redistribution


def _execute_existing_pipeline(
    version: str,
    config_path: Path,
    sources: dict[str, Path],
    selected: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, object],
]:
    all_elections = (*SCORED_ELECTIONS, TARGET_ELECTION)
    engines = {
        active.nested.engine,
        active.assignment_builder.engine,
        active.nested.base_eval.engine,
        assignment_builder.engine,
    }
    captures: dict[str, pd.DataFrame] = {}

    # The promoted runner assembles both the rolling rows and the electorate
    # base inside strict_input_policy().  Building either frame before entering
    # that context silently re-enables undated curated sensitivity inputs and
    # changes the frozen historical Ridge chain.  Keep the prospective target
    # under the identical boundary as well.
    with _model_runtime(version, config_path, selected), active.strict_input_policy():
        strict_key = active.nested.engine.STRICT_UNDATED_CURATED_INPUTS_ENV
        if os.environ.get(strict_key) != "1":
            raise RuntimeError("prospective assembly escaped strict input policy")
        historical_full = active.nested._prepare_rows()
        historical_base = active.nested._base_layer_frame(
            require_frozen_reproduction=False
        )
        source_attributes: list[tuple[object, str, object]] = []
        if "mega_issue_intensity" in sources:
            source_attributes.append(
                (active, "MEGA_ISSUE_INTENSITY", sources["mega_issue_intensity"])
            )
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
                    (
                        engine,
                        "CANDIDATE_POLITICAL_LANDSCAPE",
                        str(sources["political_landscape"]),
                    ),
                    (
                        engine,
                        "THIRD_CANDIDATE_PROFILE",
                        str(sources["third_candidate_profile"]),
                    ),
                    (
                        engine,
                        "AUTO_CANDIDATE_ISSUE_PROFILE",
                        str(sources["candidate_issue_profile"]),
                    ),
                    (
                        engine,
                        "AUTO_MEGA_ISSUE_AXIS",
                        str(sources["mega_issue_axis"]),
                    ),
                    (
                        engine,
                        "AUTO_MEGA_ISSUE_ATTRIBUTION",
                        str(sources["mega_issue_attribution"]),
                    ),
                    (
                        engine,
                        "CANDIDATE_PARTY_SPEECH_CONTEXT",
                        str(sources["candidate_party_speech_context"]),
                    ),
                    (
                        engine,
                        "CANDIDATE_PARTY_TONE_GAP",
                        str(sources["candidate_party_tone_gap"]),
                    ),
                    (
                        engine,
                        "CANDIDATE_PUBLIC_TREATMENT",
                        str(sources["candidate_public_treatment"]),
                    ),
                ]
            )
            if "mega_issue_intensity" in sources:
                source_attributes.extend(
                    [
                        (
                            engine,
                            "ENHANCED_MEGA_ISSUE_INTENSITY",
                            str(sources["mega_issue_intensity"]),
                        ),
                        (
                            engine,
                            "MEGA_ISSUE_TAXONOMY",
                            str(sources["mega_issue_taxonomy"]),
                        ),
                    ]
                )
        with patching.patched(source_attributes):
            target = active.nested.engine.assemble()
            target = target.loc[target["election_id"].eq(TARGET_ELECTION)].copy()
            if len(target) != 51:
                raise RuntimeError(f"expected 51 prospective rows, found {len(target)}")
            target_feature_columns = [
                "election_id",
                "region_id",
                "slot",
                "candidate_name",
                *active.nested.BASE_PREDICTORS,
                "candidate_weight",
            ]
            missing_features = [
                column for column in target_feature_columns if column not in target.columns
            ]
            if missing_features:
                raise RuntimeError(
                    f"prospective target lacks frozen Ridge inputs: {missing_features}"
                )
            target_feature_audit = target[target_feature_columns].copy()
            assignment_attributes = [
                (assignment_builder, "ELECTIONS", all_elections),
                (active.assignment_builder, "ELECTIONS", all_elections),
            ]
            with patching.patched(assignment_attributes):
                assignments, assignment_audit, _ = assignment_builder.build()
            target_assignments = assignments.loc[
                assignments["election_id"].eq(TARGET_ELECTION)
            ].copy()
            target_feature_audit = target_feature_audit.merge(
                target_assignments[
                    [
                        "election_id",
                        "candidate_name",
                        "assigned_slot",
                        "preliminary_mean_share",
                        "pre_withdrawal_mean_share",
                        "post_withdrawal_mean_share",
                        "withdrawal_event_applied",
                    ]
                ],
                on=["election_id", "candidate_name"],
                how="left",
                validate="many_to_one",
            )
            full = _target_full(target, historical_full, target_assignments)
            base = pd.concat(
                [historical_base, _target_base(target, historical_base, version)],
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

            original_attach_layers = active.nested._attach_layers

            def attach_layers_with_coverage_diagnostics(base_frame, outer_frame):
                outer_keys = outer_frame[
                    ["election_id", "region_id", "join_slot"]
                ].drop_duplicates()
                coverage = base_frame[
                    ["election_id", "region_id", "slot"]
                ].merge(
                    outer_keys,
                    left_on=["election_id", "region_id", "slot"],
                    right_on=["election_id", "region_id", "join_slot"],
                    how="left",
                    indicator=True,
                )
                missing = coverage.loc[
                    coverage["_merge"].eq("left_only"),
                    ["election_id", "region_id", "slot"],
                ]
                if not missing.empty:
                    raise RuntimeError(
                        "shadow prediction coverage is incomplete: "
                        + json.dumps(
                            missing.head(20).to_dict("records"),
                            ensure_ascii=False,
                        )
                    )
                return original_attach_layers(base_frame, outer_frame)

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
                (active.nested, "_attach_layers", attach_layers_with_coverage_diagnostics),
                (active.nested, "ELECTIONS", all_elections),
                (active.nested.base_eval, "ALLOWED_ELECTIONS", all_elections),
                (active.nested, "CONFIG_PATH", sources["outer_config"]),
                (active, "CANDIDATE_ISSUE_PROFILE", sources["candidate_issue_profile"]),
                (
                    active.fully_nested_policy,
                    "deployment_stage_from_completed_folds",
                    no_deployment_losses,
                ),
            ]

            direct_profile = pd.read_csv(
                sources["candidate_direct_issue_profile"], encoding="utf-8-sig"
            )
            direct_taxonomy = pd.read_csv(
                sources.get(
                    "mega_issue_taxonomy",
                    AUTOMATIC_DIR / "mega_issue_taxonomy.csv",
                ),
                encoding="utf-8-sig",
            )
            event_aligned_direct_profile = (
                active.mega_issue_adjustment.align_profile_to_event_class(
                    direct_profile,
                    direct_taxonomy,
                    ELECTION_DATES,
                )
            )
            original_compile_direct_mega_scores = (
                active.mega_issue_adjustment.compile_direct_mega_scores
            )

            def compile_candidate_only_direct_mega_scores(
                profile,
                intensity,
                election_dates,
                **kwargs,
            ):
                del profile
                return original_compile_direct_mega_scores(
                    event_aligned_direct_profile,
                    intensity,
                    election_dates,
                    **kwargs,
                )

            runtime_attributes.append(
                (
                    active.mega_issue_adjustment,
                    "compile_direct_mega_scores",
                    compile_candidate_only_direct_mega_scores,
                )
            )
            try:
                with patching.patched(runtime_attributes):
                    active.run(
                        output_dir=Path(tempfile.gettempdir()) / "prospective_sink",
                        rejection_beneficiary_routing_enabled=version in {"v24", "v25"},
                    )
            finally:
                active._atomic_csv = original_atomic_csv

    predictions = captures.get("nested_predictions.csv")
    input_manifest = captures.get("input_manifest.csv", pd.DataFrame())
    if predictions is None:
        raise RuntimeError("existing pipeline did not emit prospective predictions")
    v24_audits: dict[str, pd.DataFrame] = {}
    historical_reproduction: dict[str, object] = {
        "required": False,
        "passed": True,
        "rows": 0,
        "maximum_absolute_difference": 0.0,
    }
    if version in {"v24", "v25"}:
        from presidential_issue_engine import strong_incumbent_veto
        from presidential_issue_engine import third_candidate_lineage_constraint
        from presidential_issue_engine import weak_same_lane_refusal

        predictions["v24_pre_extension_pred"] = predictions["layer_pred"]
        predictions, all_v24_veto = (
            strong_incumbent_veto.apply_strong_incumbent_veto(predictions)
        )
        predictions["v24_post_strong_veto_pred"] = predictions["layer_pred"]
        predictions, all_v24_lineage = (
            third_candidate_lineage_constraint.apply_lineage_ceiling(
                predictions
            )
        )
        predictions["v24_post_lineage_ceiling_pred"] = predictions[
            "layer_pred"
        ]
        predictions, all_v24_refusal = (
            weak_same_lane_refusal.apply_weak_same_lane_refusal(predictions)
        )
        predictions["v24_post_weak_lane_refusal_pred"] = predictions[
            "layer_pred"
        ]
        all_audits = {
            "strong_incumbent_veto": all_v24_veto,
            "third_candidate_lineage_ceiling": all_v24_lineage,
            "weak_same_lane_refusal": all_v24_refusal,
        }
        v24_audits = {
            name: frame.loc[
                frame["election_id"].astype(str).eq(TARGET_ELECTION)
            ].copy()
            if "election_id" in frame.columns
            else frame.iloc[0:0].copy()
            for name, frame in all_audits.items()
        }

        canonical_output = (
            active_v25.DEFAULT_OUTPUT if version == "v25" else active_v24.DEFAULT_OUTPUT
        )
        canonical = pd.read_csv(
            canonical_output / "nested_predictions.csv",
            encoding="utf-8-sig",
            low_memory=False,
        )
        reproduced = predictions.loc[
            predictions["election_id"].astype(str).ne(TARGET_ELECTION)
        ].copy()
        keys = ["election_id", "region_id", "source_slot"]
        canonical_keys = canonical[keys].astype(str)
        reproduced_keys = reproduced[keys].astype(str)
        if set(map(tuple, canonical_keys.to_numpy())) != set(
            map(tuple, reproduced_keys.to_numpy())
        ):
            raise RuntimeError("prospective harness changed the frozen V24 row keys")
        expected = canonical[keys + ["layer_pred"]].copy()
        observed = reproduced[keys + ["layer_pred"]].copy()
        compared = expected.merge(
            observed,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_canonical", "_prospective_harness"),
        )
        difference = (
            pd.to_numeric(compared["layer_pred_canonical"], errors="raise")
            - pd.to_numeric(
                compared["layer_pred_prospective_harness"], errors="raise"
            )
        ).abs()
        maximum_difference = float(difference.max()) if len(difference) else 0.0
        if not np.allclose(difference.to_numpy(float), 0.0, rtol=0.0, atol=1e-12):
            debug_dir = ROOT / "outputs" / f"prospective_pres_2025_{version}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(
                debug_dir / "rejected_historical_harness_debug.csv",
                index=False,
                encoding="utf-8-sig",
                lineterminator="\n",
            )
            compared["absolute_difference"] = difference
            largest = compared.nlargest(8, "absolute_difference")[
                [
                    *keys,
                    "layer_pred_canonical",
                    "layer_pred_prospective_harness",
                    "absolute_difference",
                ]
            ]
            raise RuntimeError(
                f"prospective harness does not reproduce frozen {version.upper()} history: "
                f"maximum absolute difference={maximum_difference:.16g}; "
                + json.dumps(largest.to_dict("records"), ensure_ascii=False)
            )
        historical_reproduction = {
            "required": True,
            "passed": True,
            "rows": int(len(compared)),
            "maximum_absolute_difference": maximum_difference,
            "canonical_predictions_sha256": _sha256(
                canonical_output / "nested_predictions.csv"
            ),
        }

    target_predictions = predictions.loc[
        predictions["election_id"].eq(TARGET_ELECTION)
    ].copy()
    if (
        "candidate_name" not in target_predictions.columns
        and "candidate_name_x" in target_predictions.columns
    ):
        target_predictions["candidate_name"] = target_predictions["candidate_name_x"]
    target_predictions["predicted_share"] = target_predictions["layer_pred"]
    stage_audit = _safe_stage_audit(target_predictions)
    # ``rename`` alone leaves two columns named ``slot`` - the preliminary
    # assignment and the renamed registry slot - and the duplicate drop below
    # keeps the first, silently discarding the rename. Every structural layer
    # keys on ``source_slot``, so the reported label must be that one.
    target_predictions = target_predictions.drop(
        columns=["slot"], errors="ignore"
    ).rename(columns={"source_slot": "slot"})
    name_column = (
        "candidate_name_x"
        if "candidate_name_x" in target_predictions.columns
        else "candidate_name"
    )
    target_predictions = target_predictions.rename(columns={name_column: "candidate_name"})
    target_predictions = target_predictions.loc[:, ~target_predictions.columns.duplicated()]
    return (
        target_predictions[list(OUTPUT_COLUMNS)],
        input_manifest,
        stage_audit,
        target_feature_audit,
        v24_audits,
        historical_reproduction,
    )


def _safe_stage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain model diagnostics while excluding every outcome-shaped field."""

    out = frame.copy()
    if "pred" in out.columns:
        out = out.rename(columns={"pred": "base_stage_prediction"})
    forbidden = []
    for column in out.columns:
        folded = str(column).casefold()
        if (
            folded in FORBIDDEN_OUTCOME_COLUMNS
            or "actual" in folded
            or "error" in folded
            or "mae" in folded
            or folded == "winner"
        ):
            forbidden.append(column)
    out = out.drop(columns=forbidden, errors="ignore")
    return out.loc[:, ~out.columns.duplicated()].copy()


def _national_summary(predictions: pd.DataFrame, version: str = "v23") -> pd.DataFrame:
    volume = _prior_region_volume(version)
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
    version: str,
) -> pd.DataFrame:
    explicit = [
        config_path,
        _runtime_policy_path(version, config_path),
        REGISTRY,
        CONTEXT_DIR / "model_issue_salience.csv",
        CONTEXT_DIR / "model_candidate_issue_link.csv",
        EXPLICIT_TARGET_CONTEXT,
        CANDIDATE_TARGET_CONTEXT,
        CONTEXT_DIR / "manifest.json",
        OFFICIAL_2025_MINUTES,
        ISSUE_CONTEXT_RULES,
        CONTEXT_DIR / "assembly22_speaker_roster.csv",
        CONTEXT_DIR / "assembly22_speaker_roster.manifest.json",
        CONTEXT_DIR / "assembly_speaker_influence_pres_2025.csv",
        CONTEXT_DIR / "assembly_speaker_influence_pres_2025_diagnostics.csv",
        _historical_results_path(version),
        HISTORY,
        REGIONS,
        OUTER_CONFIG,
        DEPLOYMENT_CONFIG,
        CANDIDATE_LINK_HISTORY,
        FORECAST_CANDIDATE_CONTEXT_DIR / "candidate_party_speech_context.csv",
        FORECAST_CANDIDATE_CONTEXT_DIR / "candidate_party_tone_gap.csv",
        FORECAST_CANDIDATE_CONTEXT_DIR / "candidate_public_treatment.csv",
        FORECAST_CANDIDATE_CONTEXT_DIR / "candidate_vote_conversion_context.csv",
        FORECAST_CANDIDATE_LANDSCAPE,
        FORECAST_THIRD_CANDIDATE_PROFILE,
        FORECAST_ISSUE_SEED_DIR / "candidate_issue_profile.csv",
        FORECAST_ISSUE_SEED_DIR / "mega_issue_axis.csv",
        FORECAST_ISSUE_SEED_DIR / "mega_issue_attribution.csv",
        PARTY_TRANSITIONS,
        ROOT / "data/raw/official_sources/nec_assembly_district_history.csv",
        ROOT / active.nested.engine.SALIENCE,
        ROOT / "scripts/build_speech_derived_issue_context.py",
        ROOT / "scripts/build_speech_derived_candidate_context_v2.py",
        ROOT / "scripts/build_candidate_party_speech_context.py",
        ROOT / "scripts/extract_assembly_speaker_issue_matches.py",
        ROOT / "scripts/run_prospective_forecast.py",
        ROOT / "src/election_forecast/features/issue_matcher.py",
        ROOT / "presidential_issue_engine/automatic_controls_v22.py",
        ROOT / "presidential_issue_engine/speech_derived_mega_intensity.py",
        ROOT / "presidential_issue_engine/mega_issue_adjustment.py",
        ROOT / "presidential_issue_engine/issue_vote_engine.py",
    ]
    if version in {"v24", "v25"}:
        explicit.extend(
            [
                active_v24.V24_BASELINE,
                active_v24.V24_DATA / "candidate_slot_assignments_v2.csv",
                active_v24.V24_DATA / "coalition_events.csv",
                active_v24.V24_DATA / "candidate_party_speech_context.csv",
                active_v24.V24_DATA / "candidate_vote_conversion_context.csv",
                active_v24.V24_DATA / "scored_contest_scope.csv",
                active_v24.V24_DATA / "third_candidate_lineage.csv",
                ROOT / "scripts/run_active_presidential_model_v24.py",
                ROOT / "presidential_issue_engine/strong_incumbent_veto.py",
                ROOT
                / "presidential_issue_engine/third_candidate_lineage_constraint.py",
                ROOT / "presidential_issue_engine/weak_same_lane_refusal.py",
            ]
        )
        if version == "v25":
            explicit.extend(
                [
                    ROOT / "scripts/run_active_presidential_model_v25.py",
                    active_v25.DEFAULT_OUTPUT / "nested_predictions.csv",
                    active_v25.DEFAULT_OUTPUT / "summary.json",
                ]
            )
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


def run(version: str = "v23", *, output_dir_override: Path | None = None) -> Path:
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
    _assert_target_input_coverage(selected, salience, candidate_link, cutoff)
    if version in {"v24", "v25"}:
        from presidential_issue_engine import third_candidate_lineage_constraint

        lineage = third_candidate_lineage_constraint.load_lineage()
        target_lineage = lineage.loc[
            lineage["election_id"].astype(str).eq(TARGET_ELECTION)
        ].copy()
        expected_third = set(
            selected.loc[
                selected["slot"].astype(str).eq("C"), "candidate_name"
            ].astype(str)
        )
        if set(target_lineage["candidate_name"].astype(str)) != expected_third:
            raise RuntimeError("V24 lineage table does not match the selected C candidate")
        lineage_date = pd.to_datetime(target_lineage["available_date"], errors="coerce")
        if lineage_date.isna().any() or lineage_date.gt(cutoff).any():
            raise RuntimeError("V24 lineage evidence crosses the D-1 cutoff")

    output_dir = (
        Path(output_dir_override)
        if output_dir_override is not None
        else ROOT / "outputs" / f"prospective_pres_2025_{version}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mega_control_outputs: dict[str, pd.DataFrame] = {}
    with tempfile.TemporaryDirectory(prefix="prospective_pres_2025_") as temp_dir:
        sources, candidate_context_diagnostics = _prospective_sources(
            Path(temp_dir),
            registry,
            selected,
            salience,
            candidate_link,
            version=version,
        )
        (
            predictions,
            captured_manifest,
            stage_audit,
            target_feature_audit,
            v24_audits,
            historical_reproduction,
        ) = (
            _execute_existing_pipeline(version, config_path, sources, selected)
        )
        if "mega_issue_intensity" in sources:
            for key in (
                "mega_issue_intensity",
                "mega_issue_taxonomy",
                "mega_issue_taxonomy_audit",
            ):
                frame = pd.read_csv(sources[key], encoding="utf-8-sig")
                if "election_id" in frame.columns:
                    frame = frame.loc[
                        frame["election_id"].astype(str).eq(TARGET_ELECTION)
                    ].copy()
                mega_control_outputs[key] = frame

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
    government_columns = {
        "government_evidence_count",
        "government_evidence_weight",
        "government_rejection_strength",
    }
    if not government_columns.issubset(stage_audit.columns):
        raise RuntimeError("prospective stage audit lacks government-evidence fields")
    government_evidence = stage_audit[list(government_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    if not government_evidence.to_numpy(float).any():
        raise RuntimeError("current-government Assembly evidence did not reach the model")

    national = _national_summary(predictions, version)
    manifest = _input_manifest(captured_manifest, config_path, version)
    predictions.to_csv(
        output_dir / "prospective_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    national.to_csv(
        output_dir / "national_summary.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    manifest.to_csv(
        output_dir / "input_manifest.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    stage_audit.to_csv(
        output_dir / "prediction_stage_audit.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    target_feature_audit.to_csv(
        output_dir / "target_feature_audit.csv",
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    for name, frame in mega_control_outputs.items():
        frame.to_csv(
            output_dir / f"prospective_{name}.csv",
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
    for name, audit in v24_audits.items():
        schema = list(V24_AUDIT_COLUMNS[name])
        extra = [column for column in audit.columns if column not in schema]
        audit = audit.reindex(columns=[*schema, *extra])
        audit.to_csv(
            output_dir / f"{name}_audit.csv",
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
    run_manifest = {
        "schema": "prospective_presidential_forecast_v1",
        "status": "forecast_only_not_scored",
        "election_id": TARGET_ELECTION,
        "version": version,
        "forecast_cutoff": FORECAST_CUTOFF,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256(config_path),
        "runtime_policy_path": _runtime_policy_path(
            version, config_path
        ).relative_to(ROOT).as_posix(),
        "runtime_policy_sha256": _sha256(
            _runtime_policy_path(version, config_path)
        ),
        "runtime_policy_matches_declared_config": (
            _runtime_policy_path(version, config_path).resolve()
            == config_path.resolve()
        ),
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
        "candidate_strength_projection": candidate_context_diagnostics[
            "candidate_strength"
        ],
        "government_context_link": candidate_context_diagnostics[
            "government_context_link"
        ],
        "mega_issue_controls": candidate_context_diagnostics[
            "mega_issue_controls"
        ],
        "national_region_weight_source": "pres_2022_valid_vote_volume",
        "outcome_columns_used": [],
        "performance_metrics_computed": False,
        "pres_2025_outcome_present": False,
        "model_selection_performed": False,
        "model_parameters_changed": False,
        "frozen_historical_reproduction": historical_reproduction,
        "prediction_stage_audit_rows": int(len(stage_audit)),
        "target_feature_audit_rows": int(len(target_feature_audit)),
        "target_feature_columns": target_feature_audit.columns.tolist(),
        "v24_postprocess_order": list(V24_POSTPROCESS_ORDER)
        if version in {"v24", "v25"}
        else [],
        "v24_postprocess_audit_rows": {
            name: int(len(frame)) for name, frame in v24_audits.items()
        },
        "assembly_context_manifest_sha256": _sha256(CONTEXT_DIR / "manifest.json"),
        "assembly_context_certification": context_manifest.get("status"),
        "generated_candidate_context_sha256": candidate_context_diagnostics[
            "government_context_link"
        ]["generated_context_sha256"],
        "assembly22_roster_manifest_sha256": _sha256(
            CONTEXT_DIR / "assembly22_speaker_roster.manifest.json"
        ),
        "assembly_speaker_profile_sha256": _sha256(
            CONTEXT_DIR / "assembly_speaker_influence_pres_2025.csv"
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v23", "v24", "v25"), default="v23")
    args = parser.parse_args()
    output_dir = run(args.version)
    print(output_dir.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
