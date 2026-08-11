"""Strict nested v17 ablation of separated-evidence third-candidate stature."""

from __future__ import annotations

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
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "election_derived_third_candidate_v14_ablation"
V17_RUN = ROOT / "outputs" / "active_presidential_nested_v17"
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)
ALIGNMENT = (
    ROOT
    / "outputs"
    / "automatic_regional_party_alignment_v11"
    / "manual_plus_automatic_alignment.csv"
)
PROFILE = (
    ROOT
    / "outputs"
    / "election_derived_third_candidate_profile_v14"
    / "third_candidate_profile.csv"
)
PRESSURE = ROOT / "data" / "raw" / "third_candidate_pressure.csv"


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def main(
    *,
    output_dir: Path = OUTPUT_DIR,
    profile_path: Path = PROFILE,
    pressure_path: Path = PRESSURE,
    variant_label: str = "separated_election_profile_manual_pressure",
    experiment_name: str = "election_derived_third_candidate_v14",
) -> None:
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
        run = clean._run_variant(
            variant_label,
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=ALIGNMENT,
            third_profile_path=profile_path,
            third_pressure_path=pressure_path,
            output_root=output_dir,
        )
    audit_holder["audit"].to_csv(
        output_dir / "automatic_response_gain_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    labels = [
        ("v17_reference", V17_RUN),
        (variant_label, run),
    ]
    summary = pd.DataFrame(
        [{"variant_label": label, **_metrics(path)} for label, path in labels]
    )
    frames = []
    for label, path in labels:
        frame = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
        frame["variant_label"] = label
        frames.append(frame)
    by_election = pd.concat(frames, ignore_index=True)
    reference = by_election.loc[
        by_election["variant_label"].eq("v17_reference"),
        ["election_id", "regional_weighted_mae_pp"],
    ].set_index("election_id")["regional_weighted_mae_pp"]
    by_election["regional_change_vs_v17_pp"] = (
        by_election["regional_weighted_mae_pp"]
        - by_election["election_id"].map(reference)
    )
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        output_dir / "by_election.csv", index=False, encoding="utf-8-sig"
    )

    candidate = summary.loc[
        summary["variant_label"].eq(variant_label)
    ].iloc[0]
    candidate_changes = by_election.loc[
        by_election["variant_label"].eq(variant_label)
    ]
    reference_metrics = summary.loc[
        summary["variant_label"].eq("v17_reference")
    ].iloc[0]
    regional_improved = bool(
        candidate["regional_equal_election_macro_mae_pp"]
        < reference_metrics["regional_equal_election_macro_mae_pp"]
    )
    national_improved = bool(
        candidate["national_equal_election_macro_mae_pp"]
        < reference_metrics["national_equal_election_macro_mae_pp"]
    )
    maximum_regression = float(
        candidate_changes["regional_change_vs_v17_pp"].max()
    )
    promote = regional_improved and national_improved and maximum_regression <= 0.10
    decision = {
        "experiment": experiment_name,
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "direct_party_and_candidate_ballots_separated": True,
        "active_model_changed": False,
        "regional_mae_pp": float(
            candidate["regional_equal_election_macro_mae_pp"]
        ),
        "national_mae_pp": float(
            candidate["national_equal_election_macro_mae_pp"]
        ),
        "maximum_election_regression_pp": maximum_regression,
        "promotion_decision": "promote_candidate" if promote else "experiment_only",
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print()
    print(by_election.to_string(index=False))
    print()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
