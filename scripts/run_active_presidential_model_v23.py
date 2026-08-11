"""Run active V23 with unified candidate, withdrawal, and generation controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import build_automatic_controls_v23 as automatic_v23  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402
from scripts import run_active_presidential_model_v22 as active_v22  # noqa: E402


CONFIG = ROOT / "data" / "config" / "active_presidential_model_v23.json"
OUTPUT_DIR = ROOT / "outputs" / "active_presidential_nested_v23"
ASSIGNMENT_DIR = ROOT / "outputs" / "preliminary_slot_assignment_v23"
AUTOMATIC_DIR = ROOT / "outputs" / "automatic_controls_v23"
REGISTRY = AUTOMATIC_DIR / "withdrawal_transfer_registry.csv"
GENERATION = AUTOMATIC_DIR / "election_generation_weights.csv"

LEGACY_TRANSFER_INPUTS = {
    "data/raw/withdrawn_candidate_transfers.csv",
    "data/raw/withdrawal_event_profiles.csv",
    "presidential_issue_engine/fixed_dataset/coalition_events.csv",
}

V23_BUILD_SOURCES = (
    ROOT / "data" / "raw" / "candidate_identity_aliases.csv",
    ROOT / "data" / "raw" / "withdrawal_events.csv",
    ROOT / "data" / "raw" / "official_sources" / "assembly_candidate_attention_history.csv",
    ROOT / "data" / "raw" / "official_sources" / "nec_age_turnout_composition_history.csv",
    ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv",
    AUTOMATIC_DIR / "candidate_political_profiles.csv",
    AUTOMATIC_DIR / "candidate_political_landscape.csv",
    AUTOMATIC_DIR / "third_candidate_profile.csv",
    AUTOMATIC_DIR / "third_candidate_pressure.csv",
    REGISTRY,
    GENERATION,
)


class _AutomaticBuilderProxy:
    @staticmethod
    def build(*, status: str, active_model_changed: bool) -> dict[str, object]:
        del status
        return automatic_v23.build(
            status="active_v23_unified_withdrawal_generation",
            active_model_changed=active_model_changed,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finalize_input_manifest() -> None:
    manifest_path = OUTPUT_DIR / "input_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    normalized = manifest["path"].astype(str).str.replace("\\", "/", regex=False)
    legacy_read = sorted(set(normalized) & LEGACY_TRANSFER_INPUTS)
    if legacy_read:
        raise RuntimeError(f"legacy transfer inputs were read by active V23: {legacy_read}")

    extra = pd.DataFrame(
        [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (CONFIG, *V23_BUILD_SOURCES)
        ]
    )
    manifest = (
        pd.concat([manifest, extra], ignore_index=True)
        .drop_duplicates("path", keep="last")
        .sort_values("path")
        .reset_index(drop=True)
    )
    active._atomic_csv(manifest, manifest_path)


def main() -> None:
    engines = {active.nested.engine, active.assignment_builder.engine}
    attributes: list[tuple[object, str, object]] = [
        (active_v22, "CONFIG", CONFIG),
        (active_v22, "OUTPUT_DIR", OUTPUT_DIR),
        (active_v22, "ASSIGNMENT_DIR", ASSIGNMENT_DIR),
        (active_v22, "AUTOMATIC_DIR", AUTOMATIC_DIR),
        (active_v22, "ALIGNMENT", AUTOMATIC_DIR / "regional_alignment_with_policy.csv"),
        (active_v22, "THIRD_PROFILE", AUTOMATIC_DIR / "third_candidate_profile.csv"),
        (active_v22, "THIRD_PRESSURE", AUTOMATIC_DIR / "third_candidate_pressure.csv"),
        (active_v22, "THIRD_LANDSCAPE", AUTOMATIC_DIR / "candidate_political_landscape.csv"),
        (active_v22, "MEGA_INTENSITY", AUTOMATIC_DIR / "mega_issue_intensity.csv"),
        (active_v22, "MEGA_TAXONOMY", AUTOMATIC_DIR / "mega_issue_taxonomy.csv"),
        (active_v22, "ECONOMIC_ALIGNMENT", AUTOMATIC_DIR / "economic_slot_alignment.csv"),
        (active_v22, "HOUSING_ALIGNMENT", AUTOMATIC_DIR / "housing_slot_alignment.csv"),
        (active_v22, "automatic_v22", _AutomaticBuilderProxy),
    ]
    for engine in engines:
        attributes.extend(
            [
                (engine, "WITHDRAWAL_TRANSFER_REGISTRY", str(REGISTRY)),
                (engine, "ELECTION_GENERATION_WEIGHTS", str(GENERATION)),
            ]
        )

    with patching.patched(attributes):
        active_v22.main()

    _finalize_input_manifest()
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    promotion = {
        "schema": "active_presidential_model_promotion_v23",
        "status": "active",
        "predecessor": "active_v22",
        "experiment": "automatic_controls_v23_ablation_v3",
        "promotion_candidate": "v23_unified_profile_transfer_generation",
        "strict_nested": True,
        "single_candidate_profile": True,
        "single_withdrawal_transfer_registry": True,
        "legacy_transfer_inputs_active": [],
        "generation_weights_status": "active_latest_strictly_prior_official_report",
        "withdrawal_transfer_rate_mode": "common_low_medium_high_scenarios",
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used_by_new_layers": [],
        "selection_is_development_outcome_aware": True,
        "untouched_historical_holdout": False,
        "config_sha256": _sha256(CONFIG),
        "registry_sha256": _sha256(REGISTRY),
        "generation_weights_sha256": _sha256(GENERATION),
        "predictions_sha256": _sha256(OUTPUT_DIR / "nested_predictions.csv"),
        "metrics": summary["metrics"],
        "rollback_checkpoint": "backups/model_checkpoints/20260802_pre_v23_unified_withdrawal_generation",
    }
    active._atomic_json(promotion, OUTPUT_DIR / "promotion_manifest.json")
    print(json.dumps(promotion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
