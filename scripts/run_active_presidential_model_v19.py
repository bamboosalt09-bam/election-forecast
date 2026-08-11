"""Run active V19 with automatic viability and party-lineage corroboration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import automatic_contest_response  # noqa: E402
from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine.regional_party_channels import (  # noqa: E402
    build_lineage_corroborated_identity_events,
)
from scripts import build_election_derived_third_candidate_profile_v14 as profile_v14  # noqa: E402
from scripts import build_election_derived_third_candidate_profile_v14b as profile_v14b  # noqa: E402
from scripts import build_lineage_corroborated_identity_v19b as identity_v19b  # noqa: E402
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


CONFIG = ROOT / "data" / "config" / "active_presidential_model_v19.json"
OUTPUT_DIR = ROOT / "outputs" / "active_presidential_nested_v19"
ASSIGNMENT_DIR = ROOT / "outputs" / "preliminary_slot_assignment_v19"
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)
ALIGNMENT = (
    ROOT
    / "outputs"
    / "automatic_regional_party_alignment_v11"
    / "manual_plus_automatic_alignment.csv"
)
THIRD_PROFILE = (
    ROOT
    / "outputs"
    / "election_derived_third_candidate_profile_v14b"
    / "third_candidate_profile.csv"
)
ASSEMBLY_HISTORY = (
    ROOT / "data" / "raw" / "official_sources" / "nec_assembly_district_history.csv"
)
CORROBORATION_GAIN = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    profile_v14.main()
    profile_v14b.main()
    identity_v19b.main()
    assembly = pd.read_csv(ASSEMBLY_HISTORY, encoding="utf-8-sig")

    def identity_events(history: pd.DataFrame) -> pd.DataFrame:
        return build_lineage_corroborated_identity_events(
            history, assembly, corroboration_gain=CORROBORATION_GAIN
        )

    original_apply = contest_regime.apply_contest_regime_response
    audit_holder: dict[str, pd.DataFrame] = {}

    def automatic_apply(
        frame,
        regimes,
        *,
        prediction_column,
        slot_column="source_slot",
        output_column=None,
        expansion_gain=0.50,
        log_shift_cap=0.40,
        critical_elasticity=0.75,
        swing_elasticity=1.25,
        swing_log_shift_cap=0.50,
    ):
        del expansion_gain, log_shift_cap, swing_log_shift_cap
        result, audit = automatic_contest_response.apply_prior_selected_contest_response(
            frame,
            regimes,
            prediction_column=prediction_column,
            apply_response=original_apply,
            election_order=active.nested.ELECTIONS,
            slot_column=slot_column,
            output_column=output_column,
            critical_elasticity=critical_elasticity,
            swing_elasticity=swing_elasticity,
        )
        audit_holder["audit"] = audit
        return result

    with patching.patched(
        [
            (active.contest_regime, "apply_contest_regime_response", automatic_apply),
            (active.chungcheong_identity, "build_identity_events", identity_events),
        ]
    ):
        run_dir = clean._run_variant(
            "active_v19",
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=ALIGNMENT,
            third_profile_path=THIRD_PROFILE,
            config_path=CONFIG,
            run_dir_override=OUTPUT_DIR,
            assignment_dir_override=ASSIGNMENT_DIR,
            regenerate_issue_seeds_enabled=False,
            output_root=ROOT / "outputs",
        )
    audit_holder["audit"].to_csv(
        run_dir / "automatic_response_gain_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    promotion = {
        "schema": "active_presidential_model_promotion_v19",
        "status": "active",
        "predecessor": "active_v18",
        "experiment_lineage": [
            "automatic_contest_response_v10",
            "footprint_candidate_base_v9",
            "automatic_regional_party_alignment_v11",
            "election_derived_third_candidate_profile_v14b:viability_only",
            "lineage_corroborated_identity_v19b:reliability_only",
        ],
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used_by_new_layers": [],
        "automatic_third_viability": True,
        "party_level_assembly_history_restored": True,
        "regional_identity_magnitude_preserved": True,
        "lineage_corroboration_gain": CORROBORATION_GAIN,
        "manual_third_character_traits_retained": True,
        "manual_third_pressure_retained": True,
        "config_sha256": _sha256(CONFIG),
        "third_profile_sha256": _sha256(THIRD_PROFILE),
        "identity_events_sha256": _sha256(
            identity_v19b.OUTPUT_DIR / "identity_events.csv"
        ),
        "predictions_sha256": _sha256(run_dir / "nested_predictions.csv"),
        "metrics": summary["metrics"],
        "rollback_checkpoint": "backups/model_checkpoints/20260801_active_v18",
    }
    (run_dir / "promotion_manifest.json").write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(promotion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

