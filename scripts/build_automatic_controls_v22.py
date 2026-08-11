"""Build all V22 shadow automation inputs without changing the active model."""

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
    SCHEMA_VERSION,
    build_automatic_generation_weights,
    build_automatic_mega_taxonomy,
    build_automatic_responsibility_alignments,
    build_automatic_withdrawn_landscape,
    build_behavioral_party_transitions,
    build_full_automatic_third_profile,
    build_third_pressure_v22,
)
from presidential_issue_engine.regional_policy_commitment import (  # noqa: E402
    compile_policy_alignment,
)
from scripts import build_automatic_preliminary_candidate_profile_v21 as preliminary  # noqa: E402
from scripts import build_election_derived_third_candidate_profile_v15 as third_v15  # noqa: E402
from scripts import build_speech_derived_mega_intensity_v5 as mega_v5  # noqa: E402
from scripts import build_unified_exact_lineage_v21 as lineage_v21  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_controls_v22"
RAW = ROOT / "data" / "raw"
FIXED = ROOT / "presidential_issue_engine" / "fixed_dataset"


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


def build(
    *,
    status: str = "shadow_inputs_only",
    active_model_changed: bool = False,
) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mega_v5.build()
    third_v15.main()
    preliminary.main()
    lineage_v21.main()

    diagnostics = _read(
        mega_v5.OUTPUT_DIR / "mega_issue_intensity_diagnostics.csv"
    )
    taxonomy, intensity, taxonomy_audit = build_automatic_mega_taxonomy(diagnostics)
    taxonomy_path = _write(taxonomy, "mega_issue_taxonomy.csv")
    intensity_path = _write(intensity, "mega_issue_intensity.csv")
    _write(taxonomy_audit, "mega_issue_taxonomy_audit.csv")

    candidate_issue_profile = _read(RAW / "auto_issue_seed" / "candidate_issue_profile.csv")
    candidate_context = _read(RAW / "candidate_party_speech_context.csv")
    economic, housing, responsibility_audit = build_automatic_responsibility_alignments(
        candidate_issue_profile,
        candidate_context,
        active.nested.engine.ELECTION_DATES,
    )
    economic_path = _write(economic, "economic_slot_alignment.csv")
    housing_path = _write(housing, "housing_slot_alignment.csv")
    _write(responsibility_audit, "responsibility_alignment_audit.csv")

    generation, generation_audit = build_automatic_generation_weights(
        _read(RAW / "official_sources" / "nec_age_turnout_composition_history.csv"),
        active.nested.engine.ELECTION_DATES,
    )
    generation_path = _write(generation, "election_generation_weights.csv")
    _write(generation_audit, "generation_weights_audit.csv")

    active_third = _read(
        ROOT
        / "outputs"
        / "automatic_third_character_v20b"
        / "third_candidate_profile.csv"
    )
    automatic_third, third_audit = build_full_automatic_third_profile(
        active_third,
        _read(third_v15.OUTPUT_DIR / "third_candidate_profile.csv"),
        _read(preliminary.OUTPUT_DIR / "automatic_preliminary_profile.csv"),
        _read(
            ROOT
            / "outputs"
            / "preliminary_slot_assignment_v21"
            / "candidate_slot_assignments_v2.csv"
        ),
    )
    third_profile_path = _write(automatic_third, "third_candidate_profile.csv")
    _write(third_audit, "third_candidate_profile_audit.csv")

    automatic_landscape, landscape_audit = build_automatic_withdrawn_landscape(
        _read(RAW / "candidate_political_landscape.csv"), automatic_third
    )
    landscape_path = _write(automatic_landscape, "candidate_political_landscape.csv")
    _write(landscape_audit, "withdrawn_landscape_audit.csv")

    active_context_pressure, active_context_pressure_audit = build_third_pressure_v22(
        active_third,
        candidate_context,
        _read(RAW / "candidate_political_landscape.csv"),
        candidate_issue_profile,
        intensity,
        active.nested.engine.ELECTION_DATES,
    )
    active_context_pressure_path = _write(
        active_context_pressure, "third_candidate_pressure_active_context.csv"
    )
    _write(
        active_context_pressure_audit,
        "third_candidate_pressure_active_context_audit.csv",
    )

    pressure, pressure_audit = build_third_pressure_v22(
        automatic_third,
        candidate_context,
        automatic_landscape,
        candidate_issue_profile,
        intensity,
        active.nested.engine.ELECTION_DATES,
    )
    pressure_path = _write(pressure, "third_candidate_pressure.csv")
    _write(pressure_audit, "third_candidate_pressure_audit.csv")

    policy_alignment, policy_audit = compile_policy_alignment(
        _read(RAW / "regional_policy_commitments.csv"),
        candidate_issue_profile,
        _read(RAW / "issue_epoch_importance.csv"),
        active.nested.engine.ELECTION_DATES,
    )
    base_alignment = _read(
        ROOT
        / "outputs"
        / "automatic_regional_party_alignment_v11"
        / "automatic_alignment.csv"
    )
    combined_alignment = pd.concat(
        [base_alignment, policy_alignment], ignore_index=True
    ).sort_values(["election_id", "region_scope", "candidate_name"])
    policy_path = _write(combined_alignment, "regional_alignment_with_policy.csv")
    _write(policy_alignment, "regional_policy_alignment.csv")
    _write(policy_audit, "regional_policy_alignment_audit.csv")

    behavioral_transitions, transition_audit = build_behavioral_party_transitions(
        _read(RAW / "party_lineage_transitions.csv"),
        _read(lineage_v21.OUTPUT_DIR / "exact_lineage_events.csv"),
    )
    transitions_path = _write(
        behavioral_transitions, "party_lineage_transitions_behavioral.csv"
    )
    _write(transition_audit, "party_lineage_retention_audit.csv")

    outputs = [
        taxonomy_path,
        intensity_path,
        economic_path,
        housing_path,
        generation_path,
        third_profile_path,
        landscape_path,
        active_context_pressure_path,
        pressure_path,
        policy_path,
        transitions_path,
    ]
    for path in outputs:
        frame = _read(path)
        if "election_id" in frame and frame["election_id"].astype(str).str.contains("2025").any():
            raise RuntimeError(f"post-2022 row found in {path}")
    manifest = {
        "schema": SCHEMA_VERSION,
        "status": status,
        "active_model_changed": active_model_changed,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "withdrawal_transfer_rate_mode": "retained_semiautomatic_scenario_input",
        "manual_strength_removed_from_policy_registry": True,
        "active_outputs": [
            "mega_issue_taxonomy.csv",
            "mega_issue_intensity.csv",
            "economic_slot_alignment.csv",
            "housing_slot_alignment.csv",
            "third_candidate_profile.csv",
            "candidate_political_landscape.csv",
            "third_candidate_pressure.csv",
            "regional_alignment_with_policy.csv",
        ] if active_model_changed else [],
        "shadow_outputs": [
            "election_generation_weights.csv",
            "party_lineage_transitions_behavioral.csv",
        ],
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
