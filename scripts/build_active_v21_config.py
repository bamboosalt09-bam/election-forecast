"""Build V21 with one exact-lineage ledger for every regional channel."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "config" / "active_presidential_model_v20.json"
DESTINATION = ROOT / "data" / "config" / "active_presidential_model_v21.json"


def main() -> None:
    policy = json.loads(SOURCE.read_text(encoding="utf-8"))
    policy["policy_version"] = (
        "active_strict_nested_v21_unified_exact_party_genealogy"
    )
    structural = policy["structural_layers"]
    structural["unified_exact_lineage_identity"] = {
        "enabled": True,
        "ledger": "outputs/unified_exact_lineage_v21/exact_lineage_events.csv",
        "party_transition_registry": "data/raw/party_lineage_transitions.csv",
        "party_name_policy": "preserve exact observed name before lineage mapping",
        "region_policy": "same estimator and routing formula for all regions",
        "ridge_boundary": "project the same exact ledger to analytic blocs only at feature attachment",
        "candidate_ballot_reliability": "estimated from strictly prior same-date direct-party agreement",
        "lineage_resolution": "prior exact spatial profile with unresolved state retained",
        "genealogy_routing": "dated predecessor-successor graph without vote fitting",
        "manual_alignment_rows": False,
        "direct_score_scope": "non_major_only",
        "gain": 0.5,
        "regional_shift_cap": 0.08,
        "half_life_years": 12.0,
        "prior_strength": 1.5,
        "outcome_fields_used": [],
    }
    structural["chungcheong_regional_identity"]["superseded_by"] = (
        "unified_exact_lineage_identity"
    )
    structural["general_regional_identity"]["superseded_by"] = (
        "unified_exact_lineage_identity"
    )
    policy["derived_inputs"]["chungcheong_identity_alignment"] = (
        "outputs/automatic_regional_party_alignment_v11/automatic_alignment.csv"
    )
    policy["derived_inputs"]["regional_party_identity_events"] = (
        "outputs/unified_exact_lineage_v21/exact_lineage_events.csv"
    )
    policy["derived_inputs"]["party_lineage_transitions"] = (
        "data/raw/party_lineage_transitions.csv"
    )
    policy["promotion"] = {
        **policy["promotion"],
        "status": "active_v21_unified_exact_party_genealogy",
        "source_experiment": "outputs/unified_exact_lineage_v21_ablation",
        "methodology_priority": (
            "consistent exact genealogy over small retrospective MAE advantage"
        ),
        "manual_candidate_alignment_rows_used": False,
        "same_formula_all_regions": True,
        "exact_party_name_preserved": True,
        "assembly_constituency_exact_party_restored": True,
        "party_transition_weights_fitted_to_presidential_outcomes": False,
        "post_2022_outcomes_used": False,
        "observed_tradeoff_vs_v20": {
            "regional_change_pp": 0.1779019992878048,
            "national_change_pp": 0.28433582741005603,
            "maximum_election_regression_pp": 0.4480489895621975,
            "winner_accuracy_change": 0.0,
            "performance_improvement_claim": False,
        },
    }
    DESTINATION.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
