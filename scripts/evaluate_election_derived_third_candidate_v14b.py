"""Run the strict nested automatic-viability-only ablation."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_election_derived_third_candidate_v14 import main  # noqa: E402


if __name__ == "__main__":
    main(
        output_dir=(
            ROOT / "outputs" / "election_derived_third_candidate_v14b_ablation"
        ),
        profile_path=(
            ROOT
            / "outputs"
            / "election_derived_third_candidate_profile_v14b"
            / "third_candidate_profile.csv"
        ),
        variant_label="automatic_viability_manual_character_traits",
        experiment_name="election_derived_third_candidate_v14b",
    )
