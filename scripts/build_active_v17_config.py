"""Build the versioned v10-successor configuration without editing v16."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "config" / "active_presidential_model_v16.json"
DESTINATION = ROOT / "data" / "config" / "active_presidential_model_v17.json"


def main() -> None:
    policy = json.loads(SOURCE.read_text(encoding="utf-8"))
    policy["policy_version"] = (
        "active_strict_nested_v17_v10_successor_automatic_regional_party"
    )
    policy["derived_inputs"] = {
        "candidate_regional_base": (
            "outputs/footprint_candidate_base_v9/candidate_regional_base.csv"
        ),
        "chungcheong_identity_alignment": (
            "outputs/automatic_regional_party_alignment_v11/"
            "manual_plus_automatic_alignment.csv"
        ),
        "regional_party_identity_events": (
            "strictly_prior_full_election_history_generated_at_runtime"
        ),
        "third_candidate_profile": "data/raw/third_candidate_profile.csv",
        "third_candidate_pressure": "data/raw/third_candidate_pressure.csv",
    }
    identity = policy["structural_layers"]["chungcheong_regional_identity"]
    identity["reservoir_source"] = "strictly_prior_full_election_history"
    identity["routing_evidence"] = [
        "footprint_candidate_regional_base",
        "dated_pre_election_alignment",
        "automatic_regional_party_candidate_fit",
    ]
    identity["automatic_alignment_schema"] = (
        "automatic_regional_party_alignment_v11"
    )
    postprocess = policy["postprocess"]
    postprocess["contest_regime_gain_selection"] = "earlier_outer_folds_only"
    postprocess["rejection_beneficiary_routing"] = True
    promotion = policy["promotion"]
    promotion["status"] = "active_v17_after_strict_nested_v11_ablation"
    promotion["source_experiment"] = (
        "outputs/automatic_regional_party_alignment_v11_ablation/"
        "supplemental_full_history"
    )
    promotion["manual_third_profile_retained_after_v12_regression"] = True
    promotion["manual_third_pressure_retained_after_v12_regression"] = True
    DESTINATION.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
