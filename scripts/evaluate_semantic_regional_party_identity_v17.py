"""Strict nested ablation of semantically separated election-type weights."""

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
    SEMANTIC_HISTORY_TYPE_WEIGHTS,
    build_full_history_identity_events,
)
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "semantic_regional_party_identity_v17_ablation"
V17_RUN = ROOT / "outputs" / "active_presidential_nested_v17"
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)
ALIGNMENT = (
    ROOT
    / "outputs"
    / "semantic_regional_party_alignment_v17"
    / "manual_plus_automatic_alignment.csv"
)
CONFIG = ROOT / "data" / "config" / "active_presidential_model_v17.json"


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def semantic_events(history: pd.DataFrame) -> pd.DataFrame:
    return build_full_history_identity_events(
        history, type_weights=SEMANTIC_HISTORY_TYPE_WEIGHTS
    )


def main() -> None:
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
            (active.chungcheong_identity, "build_identity_events", semantic_events),
        ]
    ):
        run = clean._run_variant(
            "semantic_regional_party_identity",
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            chungcheong_alignment_path=ALIGNMENT,
            config_path=CONFIG,
            run_dir_override=OUTPUT_DIR / "active_run",
            assignment_dir_override=OUTPUT_DIR / "slot_assignment",
            regenerate_issue_seeds_enabled=False,
            output_root=OUTPUT_DIR,
        )
    audit_holder["audit"].to_csv(
        OUTPUT_DIR / "automatic_response_gain_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = pd.DataFrame(
        [
            {"variant_label": "v17_reference", **_metrics(V17_RUN)},
            {"variant_label": "semantic_regional_party_identity", **_metrics(run)},
        ]
    )
    reference_by = pd.read_csv(V17_RUN / "by_election.csv", encoding="utf-8-sig")
    reference_by["variant_label"] = "v17_reference"
    candidate_by = pd.read_csv(run / "by_election.csv", encoding="utf-8-sig")
    candidate_by["variant_label"] = "semantic_regional_party_identity"
    by_election = pd.concat([reference_by, candidate_by], ignore_index=True)
    reference = reference_by.set_index("election_id")["regional_weighted_mae_pp"]
    by_election["regional_change_vs_v17_pp"] = (
        by_election["regional_weighted_mae_pp"]
        - by_election["election_id"].map(reference)
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    candidate_metrics = summary.loc[
        summary["variant_label"].eq("semantic_regional_party_identity")
    ].iloc[0]
    reference_metrics = summary.loc[
        summary["variant_label"].eq("v17_reference")
    ].iloc[0]
    changes = by_election.loc[
        by_election["variant_label"].eq("semantic_regional_party_identity")
    ]
    promote = bool(
        candidate_metrics["regional_equal_election_macro_mae_pp"]
        < reference_metrics["regional_equal_election_macro_mae_pp"]
        and candidate_metrics["national_equal_election_macro_mae_pp"]
        < reference_metrics["national_equal_election_macro_mae_pp"]
        and float(changes["regional_change_vs_v17_pp"].max()) <= 0.10
    )
    decision = {
        "experiment": "semantic_regional_party_identity_v17",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "candidate_ballots_downweighted": True,
        "active_model_changed": False,
        "regional_mae_pp": float(
            candidate_metrics["regional_equal_election_macro_mae_pp"]
        ),
        "national_mae_pp": float(
            candidate_metrics["national_equal_election_macro_mae_pp"]
        ),
        "maximum_election_regression_pp": float(
            changes["regional_change_vs_v17_pp"].max()
        ),
        "promotion_decision": "promote_candidate" if promote else "experiment_only",
    }
    (OUTPUT_DIR / "decision.json").write_text(
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
