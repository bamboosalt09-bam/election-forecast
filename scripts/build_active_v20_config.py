"""Build active V20 from V18 plus equivalent automatic character fields."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "config" / "active_presidential_model_v18.json"
DESTINATION = ROOT / "data" / "config" / "active_presidential_model_v20.json"


def main() -> None:
    policy = json.loads(SOURCE.read_text(encoding="utf-8"))
    policy["policy_version"] = (
        "active_strict_nested_v20_v10_successor_automatic_third_character"
    )
    policy["derived_inputs"]["third_candidate_profile"] = (
        "outputs/automatic_third_character_v20b/third_candidate_profile.csv"
    )
    promotion = policy["promotion"]
    promotion["status"] = "active_v20_automatic_third_character_equivalence"
    promotion["source_experiment"] = (
        "outputs/automatic_third_character_v20b_ablation"
    )
    promotion["automatic_third_viability_active"] = True
    promotion["automatic_third_character_fields"] = [
        "anti_major_party_appeal",
        "regional_base_overlap",
    ]
    promotion["manual_third_character_fields_retained"] = ["centrist_appeal"]
    promotion["manual_third_pressure_retained_after_v16_regression"] = True
    promotion["automation_equivalence_gate"] = {
        "regional_degradation_cap_pp": 0.01,
        "national_degradation_cap_pp": 0.01,
        "maximum_election_regression_cap_pp": 0.05,
        "observed_regional_change_pp": 0.0007028340994281734,
        "observed_national_change_pp": 0.0020111739859050015,
        "observed_maximum_election_regression_pp": 0.0021739546386339015,
        "performance_improvement_claim": False,
    }
    DESTINATION.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()

