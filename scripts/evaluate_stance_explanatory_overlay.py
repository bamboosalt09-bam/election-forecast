"""Compare the active engine with and without the bounded v14 issue overlay."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "presidential_issue_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import issue_vote_engine as engine  # noqa: E402
import robustness_check as robustness  # noqa: E402


OVERLAY = (
    ROOT
    / "outputs"
    / "assembly_stance"
    / "stance_nli_ambiguity_v14"
    / "explanatory_overlay"
    / "stance_issue_overlay.csv"
)
BASELINE_OVERLAY = (
    OVERLAY.parent / "baseline_overlay_before_v14.csv"
)
OUTPUT = OVERLAY.parent / "forecast_validation.json"


def _evaluate(
    gain: float,
    direct_overlay: Path | None,
    electorate_overlay: Path,
) -> dict[str, object]:
    engine.THROUGH_2022_REDERIVED_LAYER_CONFIG["overlay_gain"] = gain
    os.environ["POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH"] = (
        str(direct_overlay) if direct_overlay is not None else "off"
    )
    engine.ASSEMBLY_ISSUE_CHARACTER_OVERLAY = str(electorate_overlay)
    all_rows = engine.assemble()
    frame = robustness.competition_frame(all_rows)
    warmup = robustness.rolling_warmup_frame(all_rows)
    rows = robustness.rolling_origin_error_frame(
        frame,
        engine.PREDICTORS,
        warmup=warmup,
        warmup_order=engine.ROLLING_WARMUP_ORDER,
    )
    meta = engine.scored_contest_rows(all_rows)[
        ["election_id", "region_id", "slot", "candidate_name", "votes"]
    ].copy()
    meta["contest_votes"] = meta.groupby(["election_id", "region_id"])["votes"].transform(
        "sum"
    )
    rows = rows.merge(
        meta,
        on=["election_id", "region_id", "slot", "candidate_name"],
        how="left",
    )
    by_election: list[dict[str, object]] = []
    for election_id, group in rows.groupby("election_id", sort=True):
        by_election.append(
            {
                "election_id": election_id,
                "row_mae_pp": float(group["abs_err_pp"].mean()),
                "contest_vote_weighted_mae_pp": float(
                    np.average(group["abs_err_pp"], weights=group["contest_votes"])
                ),
            }
        )
    scored = engine.scored_contest_rows(frame)
    _, ridge_r2, _, _, _, _ = engine.ridge_fit(
        scored[engine.PREDICTORS].to_numpy(float),
        engine.normalized_vote_share_target(scored),
        alpha=engine.RIDGE_ALPHA,
        sample_weight=engine.election_epoch_sample_weight(scored),
    )
    return {
        "rolling_row_mae_pp": float(rows["abs_err_pp"].mean()),
        "rolling_contest_vote_weighted_macro_mae_pp": float(
            pd.DataFrame(by_election)["contest_vote_weighted_mae_pp"].mean()
        ),
        "loeo_row_mae_pp": float(engine.loeo_cv(frame, engine.PREDICTORS)),
        "ridge_r2": float(ridge_r2),
        "by_election": by_election,
    }


def main() -> None:
    original_config = dict(engine.THROUGH_2022_REDERIVED_LAYER_CONFIG)
    original_electorate_overlay = engine.ASSEMBLY_ISSUE_CHARACTER_OVERLAY
    previous_overlay = os.environ.get("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH")
    try:
        baseline = _evaluate(0.0, None, BASELINE_OVERLAY)
        candidate = _evaluate(0.04, OVERLAY, OVERLAY)
    finally:
        engine.THROUGH_2022_REDERIVED_LAYER_CONFIG = original_config
        engine.ASSEMBLY_ISSUE_CHARACTER_OVERLAY = original_electorate_overlay
        if previous_overlay is None:
            os.environ.pop("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH", None)
        else:
            os.environ["POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH"] = previous_overlay
    result = {
        "status": "fixed_explanatory_overlay_diagnostic",
        "selection_status": "theory_driven_reactivation_not_nested_performance_selection",
        "source_model": "stance_nli_ambiguity_v14",
        "character_gain": 0.04,
        "link_gain": 0.01,
        "direct_candidate_vote_adjustment": False,
        "baseline_v3_inactive_direct_overlay": baseline,
        "candidate_v14_bounded_overlay": candidate,
        "delta_candidate_minus_baseline": {
            key: float(candidate[key]) - float(baseline[key])
            for key in (
                "rolling_row_mae_pp",
                "rolling_contest_vote_weighted_macro_mae_pp",
                "loeo_row_mae_pp",
                "ridge_r2",
            )
        },
        "interpretation": (
            "The layer is retained for issue-character explanation. Its vote-share effect "
            "is deliberately negligible and is not evidence of predictive improvement."
        ),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
