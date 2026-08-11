"""Run pressure-only and automatic-viability-plus-pressure strict ablations."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_election_derived_third_candidate_v14 import main  # noqa: E402


AUTO_PRESSURE = (
    ROOT
    / "outputs"
    / "election_derived_third_pressure_v16"
    / "third_candidate_pressure.csv"
)


if __name__ == "__main__":
    main(
        output_dir=ROOT / "outputs" / "election_derived_third_pressure_v16a_ablation",
        profile_path=ROOT / "data" / "raw" / "third_candidate_profile.csv",
        pressure_path=AUTO_PRESSURE,
        variant_label="manual_profile_automatic_pressure",
        experiment_name="election_derived_third_pressure_v16a",
    )
    main(
        output_dir=ROOT / "outputs" / "election_derived_third_pressure_v16b_ablation",
        profile_path=(
            ROOT
            / "outputs"
            / "election_derived_third_candidate_profile_v14b"
            / "third_candidate_profile.csv"
        ),
        pressure_path=AUTO_PRESSURE,
        variant_label="automatic_viability_and_pressure",
        experiment_name="election_derived_third_pressure_v16b",
    )
