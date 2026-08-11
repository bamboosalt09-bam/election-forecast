"""Build V23 unified candidate, generation, and withdrawal controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.automatic_controls_v22 import (  # noqa: E402
    build_automatic_generation_weights,
    build_third_pressure_v22,
)
from presidential_issue_engine.automatic_withdrawal_v23 import (  # noqa: E402
    build_candidate_landscape_from_profiles,
    build_unified_candidate_profiles,
    compile_withdrawal_transfer_registry,
)
from scripts import build_automatic_controls_v22 as v22  # noqa: E402
from scripts import build_election_derived_third_candidate_profile_v15 as third_v15  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_controls_v23"
RAW = ROOT / "data" / "raw"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _primary_third_profiles(
    profiles: pd.DataFrame,
    active_profile: pd.DataFrame,
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    alias_to_id = {
        str(row.alias).strip().casefold(): str(row.candidate_id)
        for row in aliases.itertuples(index=False)
    }
    keys = {
        (
            str(row.election_id),
            str(row.slot),
            alias_to_id.get(str(row.candidate_name).strip().casefold(), ""),
        )
        for row in active_profile.itertuples(index=False)
    }
    selected = profiles.loc[
        profiles.apply(
            lambda row: (
                str(row["election_id"]),
                str(row["slot"]),
                str(row["candidate_id"]),
            )
            in keys,
            axis=1,
        )
    ].copy()
    if len(selected) != len(active_profile):
        raise RuntimeError("canonical profile did not resolve one active third row per election")
    compatibility_names = active_profile[
        ["election_id", "slot", "candidate_name"]
    ].rename(columns={"candidate_name": "engine_candidate_name"})
    selected = selected.merge(
        compatibility_names,
        on=["election_id", "slot"],
        how="left",
        validate="one_to_one",
    )
    selected["candidate_name"] = selected["engine_candidate_name"]
    return selected[
        [
            "election_id",
            "slot",
            "candidate_name",
            "viability",
            "centrist_appeal",
            "anti_major_party_appeal",
            "regional_base_overlap",
            "available_date",
            "confidence",
            "notes",
        ]
    ].sort_values(["election_id", "slot"])


def build(*, status: str = "shadow_inputs_only", active_model_changed: bool = False) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    v22.build(
        status="v22_dependency_for_v23",
        active_model_changed=False,
    )
    aliases = _read(RAW / "candidate_identity_aliases.csv")
    events = _read(RAW / "withdrawal_events.csv")
    active_profile = _read(v22.OUTPUT_DIR / "third_candidate_profile.csv")
    common_profile_args = (
        active_profile,
        _read(third_v15.OUTPUT_DIR / "third_candidate_profile.csv"),
        _read(
            ROOT
            / "outputs"
            / "automatic_preliminary_candidate_profile_v21"
            / "automatic_preliminary_profile.csv"
        ),
    )
    profiles, profile_audit = build_unified_candidate_profiles(
        *common_profile_args,
        pd.DataFrame(),
        events,
        aliases,
        active.nested.engine.ELECTION_DATES,
        _read(
            RAW
            / "official_sources"
            / "nec_assembly_district_history.csv"
        ),
        _read(
            RAW
            / "official_sources"
            / "assembly_candidate_attention_history.csv"
        ),
    )
    profiles_path = _write(profiles, "candidate_political_profiles.csv")
    _write(profile_audit, "candidate_political_profile_audit.csv")
    third_profile = _primary_third_profiles(profiles, active_profile, aliases)
    third_profile_path = _write(third_profile, "third_candidate_profile.csv")

    landscape, landscape_audit = build_candidate_landscape_from_profiles(
        _read(v22.OUTPUT_DIR / "candidate_political_landscape.csv"), profiles, aliases
    )
    landscape_path = _write(landscape, "candidate_political_landscape.csv")
    _write(landscape_audit, "candidate_political_landscape_audit.csv")

    transfer_registry, transfer_audit = compile_withdrawal_transfer_registry(
        events, profiles, active.nested.engine.ELECTION_DATES
    )
    transfer_path = _write(transfer_registry, "withdrawal_transfer_registry.csv")
    _write(transfer_audit, "withdrawal_transfer_audit.csv")

    generation, generation_audit = build_automatic_generation_weights(
        _read(RAW / "official_sources" / "nec_age_turnout_composition_history.csv"),
        active.nested.engine.ELECTION_DATES,
    )
    generation_path = _write(generation, "election_generation_weights.csv")
    _write(generation_audit, "generation_weights_audit.csv")

    pressure, pressure_audit = build_third_pressure_v22(
        third_profile,
        _read(RAW / "candidate_party_speech_context.csv"),
        landscape,
        _read(RAW / "auto_issue_seed" / "candidate_issue_profile.csv"),
        _read(v22.OUTPUT_DIR / "mega_issue_intensity.csv"),
        active.nested.engine.ELECTION_DATES,
    )
    pressure_path = _write(pressure, "third_candidate_pressure.csv")
    _write(pressure_audit, "third_candidate_pressure_audit.csv")

    inherited_paths = [
        _write(_read(v22.OUTPUT_DIR / name), name)
        for name in [
            "mega_issue_taxonomy.csv",
            "mega_issue_intensity.csv",
            "economic_slot_alignment.csv",
            "housing_slot_alignment.csv",
            "regional_alignment_with_policy.csv",
        ]
    ]

    outputs = [
        profiles_path,
        third_profile_path,
        landscape_path,
        transfer_path,
        generation_path,
        pressure_path,
        *inherited_paths,
    ]
    for path in outputs:
        frame = _read(path)
        if "election_id" in frame and frame["election_id"].astype(str).str.contains("2025").any():
            raise RuntimeError(f"post-2022 row found in {path}")
        if "target_outcome_used" in frame and frame["target_outcome_used"].astype(str).str.lower().isin(
            {"1", "true", "yes", "y"}
        ).any():
            raise RuntimeError(f"target outcome flag found in {path}")

    manifest = {
        "schema": "automatic_controls_v23",
        "status": status,
        "active_model_changed": active_model_changed,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "candidate_profile_source": "single_automatic_canonical_output",
        "withdrawal_event_source": "factual_registry_only",
        "withdrawal_transfer_policy": "universal_low_medium_high_semiautomatic_scenarios",
        "prediction_transfer_registry": "withdrawal_transfer_registry.csv",
        "legacy_transfer_inputs_active": [],
        "legacy_transfer_inputs_isolated": [
            "data/raw/withdrawn_candidate_transfers.csv",
            "data/raw/withdrawal_event_profiles.csv",
            "presidential_issue_engine/fixed_dataset/coalition_events.csv",
        ],
        "generation_weights_status": "automatic_latest_strictly_prior_official_report",
        "outputs": {
            path.name: {"rows": int(len(_read(path))), "sha256": _sha256(path)}
            for path in outputs
        },
    }
    (OUTPUT_DIR / "lineage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
