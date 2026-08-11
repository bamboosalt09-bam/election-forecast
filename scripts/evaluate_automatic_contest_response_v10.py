"""Strict nested test of prior-only automatic contest-response gains."""

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
from scripts import evaluate_district_candidate_base_clean_v8 as clean  # noqa: E402
from scripts import evaluate_speech_derived_issue_context as patching  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_contest_response_v10_ablation"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v16"
FOOTPRINT_BASE = (
    ROOT / "outputs" / "footprint_candidate_base_v9" / "candidate_regional_base.csv"
)


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        [(active.contest_regime, "apply_contest_regime_response", automatic_apply)]
    ):
        run_dir = clean._run_variant(
            "footprint_prior_selected_routed",
            None,
            rejection_routing=True,
            candidate_base_path=FOOTPRINT_BASE,
            output_root=OUTPUT_DIR,
        )
    audit = audit_holder["audit"]
    audit.to_csv(
        OUTPUT_DIR / "automatic_response_gain_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [
            {"variant_label": "active_v16", **_metrics(ACTIVE_DIR)},
            {"variant_label": "footprint_prior_selected_routed", **_metrics(run_dir)},
        ]
    )
    frames = []
    for label, path in [("active_v16", ACTIVE_DIR), ("footprint_prior_selected_routed", run_dir)]:
        frame = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
        frame["variant_label"] = label
        frames.append(frame)
    by_election = pd.concat(frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    decision = {
        "experiment": "automatic_contest_response_v10",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "target_excluded_from_each_gain_selection": bool(
            audit["target_excluded_from_selection"].all()
        ),
        "candidate_base_is_fully_automatic": True,
        "contest_response_gain_is_prior_only": True,
        "active_model_changed": False,
        "promotion_decision": "hold_as_candidate",
        "promotion_reason": (
            "Aggregate, 2017, and 2022 regional errors improve, but 2007 "
            "regional MAE regresses because pre-2007 official election history "
            "does not identify Lee Hoi-chang's Chungcheong affinity."
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print()
    print(audit.to_string(index=False))
    print()
    print(
        by_election.sort_values(["election_id", "variant_label"])[
            [
                "variant_label",
                "election_id",
                "regional_weighted_mae_pp",
                "national_candidate_mae_pp",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
