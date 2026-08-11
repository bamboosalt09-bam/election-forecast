"""Build the v18 V10-successor config with automatic third viability."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "config" / "active_presidential_model_v17.json"
DESTINATION = ROOT / "data" / "config" / "active_presidential_model_v18.json"


def main() -> None:
    policy = json.loads(SOURCE.read_text(encoding="utf-8"))
    policy["policy_version"] = (
        "active_strict_nested_v18_v10_successor_automatic_third_viability"
    )
    policy["derived_inputs"]["third_candidate_profile"] = (
        "outputs/election_derived_third_candidate_profile_v14b/"
        "third_candidate_profile.csv"
    )
    promotion = policy["promotion"]
    promotion["status"] = "active_v18_automatic_third_viability_equivalence"
    promotion["source_experiment"] = (
        "outputs/election_derived_third_candidate_v14b_ablation"
    )
    promotion["manual_third_profile_retained_after_v12_regression"] = False
    promotion["automatic_third_viability_active"] = True
    promotion["manual_third_character_traits_retained"] = True
    promotion["manual_third_pressure_retained_after_v16_regression"] = True
    promotion["automation_equivalence_gate"] = {
        "regional_mae_degradation_cap_pp": 0.01,
        "national_mae_must_not_regress": True,
        "maximum_election_regression_cap_pp": 0.05,
        "observed_regional_change_pp": 0.0032801960696021,
        "observed_national_change_pp": -0.0020731130585659,
        "observed_maximum_election_regression_pp": 0.015377074889593167,
        "performance_improvement_claim": False,
    }
    DESTINATION.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
