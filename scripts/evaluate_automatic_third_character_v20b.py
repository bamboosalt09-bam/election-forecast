"""Strict nested confirmation of the two passing automatic character fields."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.automatic_regional_party_alignment import (  # noqa: E402
    build_full_history_identity_events,
)
from scripts import evaluate_two_channel_regional_party_identity_v19 as evaluation  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_third_character_v20b_ablation"
PROFILE = ROOT / "outputs" / "automatic_third_character_v20b" / "third_candidate_profile.csv"
REFERENCE = json.loads(
    (ROOT / "outputs" / "active_presidential_nested_v18" / "summary.json").read_text(
        encoding="utf-8"
    )
)["metrics"]


def main() -> None:
    evaluation.main(
        output_dir=OUTPUT_DIR,
        event_builder=build_full_history_identity_events,
        experiment_name="automatic_third_character_v20b",
        variant_label="automatic_third_character_v20b",
        third_profile_path=PROFILE,
        decision_metadata={
            "single_changed_layer": "third_candidate_profile.character_subset",
            "party_level_assembly_history_restored": False,
            "direct_party_and_organization_channels_separated": False,
            "generic_third_rows_reliability_reduced": False,
            "automatic_fields": [
                "anti_major_party_appeal",
                "regional_base_overlap",
            ],
            "manual_fields_retained": ["centrist_appeal"],
        },
    )
    path = OUTPUT_DIR / "decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    regional_change = float(decision["regional_mae_pp"]) - float(
        REFERENCE["regional_equal_election_macro_mae_pp"]
    )
    national_change = float(decision["national_mae_pp"]) - float(
        REFERENCE["national_equal_election_macro_mae_pp"]
    )
    equivalent = bool(
        regional_change <= 0.01
        and national_change <= 0.01
        and float(decision["maximum_election_regression_pp"]) <= 0.05
    )
    decision["regional_change_pp"] = regional_change
    decision["national_change_pp"] = national_change
    decision["automation_equivalence_pass"] = equivalent
    decision["promotion_decision"] = (
        "promote_automation_equivalent" if equivalent else "experiment_only"
    )
    path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

