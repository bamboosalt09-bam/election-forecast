"""Build the active V19 policy from the verified V18 policy."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "config" / "active_presidential_model_v18.json"
DESTINATION = ROOT / "data" / "config" / "active_presidential_model_v19.json"


def main() -> None:
    policy = json.loads(SOURCE.read_text(encoding="utf-8"))
    policy["policy_version"] = (
        "active_strict_nested_v19_v10_successor_lineage_corroboration"
    )
    policy["derived_inputs"]["regional_party_identity_events"] = (
        "outputs/lineage_corroborated_identity_v19b/identity_events.csv"
    )
    promotion = policy["promotion"]
    promotion["status"] = "active_v19_lineage_corroboration"
    promotion["source_experiment"] = (
        "outputs/lineage_corroborated_identity_v19b_ablation"
    )
    promotion["party_level_assembly_history_restored"] = True
    promotion["regional_identity_magnitude_preserved"] = True
    promotion["party_lineage_used_as_reliability_only"] = True
    promotion["lineage_corroboration_gain"] = 0.25
    promotion["generic_third_rows_downweighted"] = False
    promotion["strict_nested_v19_metrics"] = {
        "regional_equal_election_macro_mae_pp": 3.216070977441497,
        "national_equal_election_macro_mae_pp": 1.4727318791700408,
        "winner_accuracy": 0.8,
        "maximum_election_regression_pp": 0.015652920275747384,
    }
    DESTINATION.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()

