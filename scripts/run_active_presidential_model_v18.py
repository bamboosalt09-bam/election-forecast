"""Run the isolated v18 V10 successor with automatic third viability."""

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
from presidential_issue_engine.automatic_regional_party_alignment import (  # noqa: E402
    build_full_history_identity_events,
)
from scripts import build_election_derived_third_candidate_profile_v14 as profile_v14  # noqa: E402
from scripts import build_election_derived_third_candidate_profile_v14b as profile_v14b  # noqa: E402
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


CONFIG = ROOT / "data" / "config" / "active_presidential_model_v18.json"
OUTPUT_DIR = ROOT / "outputs" / "active_presidential_nested_v18"
ASSIGNMENT_DIR = ROOT / "outputs" / "preliminary_slot_assignment_v18"
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    # Rebuild deterministic derived inputs so an old CSV cannot silently drive
    # a current forecast.
    profile_v14.main()
    profile_v14b.main()

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
            (
                active.chungcheong_identity,
                "build_identity_events",
                build_full_history_identity_events,
            ),
        ]
    ):
        run_dir = clean._run_variant(
            "active_v18",
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
        "schema": "active_presidential_model_promotion_v18",
        "status": "active",
        "predecessor": "active_v17",
        "experiment_lineage": [
            "automatic_contest_response_v10",
            "footprint_candidate_base_v9",
            "automatic_regional_party_alignment_v11:supplemental_full_history",
            "election_derived_third_candidate_profile_v14b:viability_only",
        ],
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used_by_new_layers": [],
        "automatic_third_viability": True,
        "manual_third_character_traits_retained": True,
        "manual_third_pressure_retained": True,
        "performance_improvement_claim": False,
        "automation_equivalence_gate": {
            "regional_mae_degradation_cap_pp": 0.01,
            "national_mae_must_not_regress": True,
            "maximum_election_regression_cap_pp": 0.05,
        },
        "config_sha256": _sha256(CONFIG),
        "third_profile_sha256": _sha256(THIRD_PROFILE),
        "predictions_sha256": _sha256(run_dir / "nested_predictions.csv"),
        "metrics": summary["metrics"],
        "rollback_checkpoint": "backups/model_checkpoints/20260801_pre_v18",
    }
    (run_dir / "promotion_manifest.json").write_text(
        json.dumps(promotion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(promotion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
