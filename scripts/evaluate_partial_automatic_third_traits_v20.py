"""Strict nested one-field ablations for third-candidate character traits."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.automatic_regional_party_alignment import (  # noqa: E402
    build_full_history_identity_events,
)
from scripts import evaluate_two_channel_regional_party_identity_v19 as evaluation  # noqa: E402
from scripts.build_partial_automatic_third_traits_v20 import FIELDS  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "partial_automatic_third_traits_v20_ablation"
PROFILE_DIR = ROOT / "outputs" / "partial_automatic_third_traits_v20"
REFERENCE_METRICS = json.loads(
    (ROOT / "outputs" / "active_presidential_nested_v18" / "summary.json").read_text(
        encoding="utf-8"
    )
)["metrics"]


def main() -> None:
    rows: list[dict[str, object]] = []
    for field in FIELDS:
        variant = f"automatic_{field}_v20"
        destination = OUTPUT_DIR / field
        evaluation.main(
            output_dir=destination,
            event_builder=build_full_history_identity_events,
            experiment_name=variant,
            variant_label=variant,
            third_profile_path=PROFILE_DIR / f"third_candidate_profile_{field}.csv",
            decision_metadata={
                "single_changed_layer": f"third_candidate_profile.{field}",
                "party_level_assembly_history_restored": False,
                "direct_party_and_organization_channels_separated": False,
                "generic_third_rows_reliability_reduced": False,
                "automatic_character_field": field,
            },
        )
        decision = json.loads((destination / "decision.json").read_text(encoding="utf-8"))
        regional_change = (
            float(decision["regional_mae_pp"])
            - float(REFERENCE_METRICS["regional_equal_election_macro_mae_pp"])
        )
        national_change = (
            float(decision["national_mae_pp"])
            - float(REFERENCE_METRICS["national_equal_election_macro_mae_pp"])
        )
        equivalent = bool(
            regional_change <= 0.01
            and national_change <= 0.01
            and float(decision["maximum_election_regression_pp"]) <= 0.05
        )
        rows.append(
            {
                "field": field,
                "regional_mae_pp": decision["regional_mae_pp"],
                "regional_change_pp": regional_change,
                "national_mae_pp": decision["national_mae_pp"],
                "national_change_pp": national_change,
                "maximum_election_regression_pp": decision[
                    "maximum_election_regression_pp"
                ],
                "automation_equivalence_pass": equivalent,
            }
        )
    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    decision = {
        "experiment": "partial_automatic_third_traits_v20",
        "strict_nested": True,
        "automation_equivalence_gate": {
            "regional_degradation_cap_pp": 0.01,
            "national_degradation_cap_pp": 0.01,
            "maximum_election_regression_cap_pp": 0.05,
        },
        "passing_fields": summary.loc[
            summary["automation_equivalence_pass"], "field"
        ].tolist(),
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

