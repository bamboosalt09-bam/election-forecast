"""Evaluate the bounded incumbent/shock response against the active snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import incumbent_shock_adjustment as shock  # noqa: E402
from scripts import evaluate_preliminary_slot_shadow_nested as nested  # noqa: E402


ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested"
OUTPUT_DIR = ROOT / "outputs" / "incumbent_shock_response"
BASELINE_SNAPSHOT = OUTPUT_DIR / "input_active_v3_predictions.csv"
PROFILE = ROOT / "data" / "raw" / "auto_issue_seed" / "candidate_issue_profile.csv"
INTENSITY = ROOT / "data" / "raw" / "mega_issue_intensity.csv"

VARIANTS = (
    ("active_baseline", 0.0, 0.0),
    ("government_burden_only", 1.0, 0.0),
    ("rupture_extra_only", 0.0, 0.40),
    ("bounded_combined", 1.0, 0.40),
)


def _winner_accuracy(national: pd.DataFrame) -> float:
    hits: list[bool] = []
    for _, group in national.groupby("election_id"):
        hits.append(
            group.loc[group["pred_pct"].idxmax(), "candidate_key"]
            == group.loc[group["actual_pct"].idxmax(), "candidate_key"]
        )
    return float(np.mean(hits))


def run() -> dict[str, object]:
    if not BASELINE_SNAPSHOT.exists():
        active_summary = json.loads(
            (ACTIVE_DIR / "summary.json").read_text(encoding="utf-8")
        )
        if active_summary.get("incumbent_shock_response") is True:
            raise RuntimeError(
                "pre-promotion v3 snapshot is missing; refusing to apply the response twice"
            )
        active = pd.read_csv(
            ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        active.to_csv(BASELINE_SNAPSHOT, index=False, encoding="utf-8-sig")
    else:
        active = pd.read_csv(BASELINE_SNAPSHOT, encoding="utf-8-sig")
    profile = pd.read_csv(PROFILE, encoding="utf-8-sig")
    intensity = pd.read_csv(INTENSITY, encoding="utf-8-sig")
    burden = shock.compile_government_burden_scores(profile, nested.engine.ELECTION_DATES)

    summaries: list[dict[str, object]] = []
    elections: list[pd.DataFrame] = []
    nationals: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    for name, burden_gain, rupture_gain in VARIANTS:
        evaluated = shock.apply_incumbent_shock_response(
            active,
            burden,
            intensity,
            nested.engine.ELECTION_DATES,
            prediction_column="layer_pred",
            output_column="experiment_pred",
            government_burden_gain=burden_gain,
            rupture_extra_gain=rupture_gain,
        )
        summary, by_election, national = nested._metrics(
            evaluated, "experiment_pred", name
        )
        summary["winner_accuracy"] = _winner_accuracy(national)
        summary["government_burden_gain"] = burden_gain
        summary["rupture_extra_gain"] = rupture_gain
        summaries.append(summary)
        elections.append(by_election)
        nationals.append(national)
        evaluated["variant"] = name
        predictions.append(evaluated)

    summary_frame = pd.DataFrame(summaries)
    election_frame = pd.concat(elections, ignore_index=True)
    national_frame = pd.concat(nationals, ignore_index=True)
    baseline = summary_frame.loc[summary_frame["variant"].eq("active_baseline")].iloc[0]
    combined = summary_frame.loc[summary_frame["variant"].eq("bounded_combined")].iloc[0]
    baseline_by = election_frame.loc[
        election_frame["variant"].eq("active_baseline")
    ].set_index("election_id")
    combined_by = election_frame.loc[
        election_frame["variant"].eq("bounded_combined")
    ].set_index("election_id")
    per_election_change = (
        combined_by["national_candidate_mae_pp"]
        - baseline_by["national_candidate_mae_pp"]
    )
    checks = {
        "national_improvement_at_least_0_20pp": float(
            baseline["national_equal_election_macro_mae_pp"]
            - combined["national_equal_election_macro_mae_pp"]
        )
        >= 0.20,
        "regional_improvement_at_least_0_15pp": float(
            baseline["regional_equal_election_macro_mae_pp"]
            - combined["regional_equal_election_macro_mae_pp"]
        )
        >= 0.15,
        "no_election_worsens_over_0_05pp": float(per_election_change.max()) <= 0.05,
        "winner_accuracy_not_worse": float(combined["winner_accuracy"])
        >= float(baseline["winner_accuracy"]),
        "pres_2007_improves": float(per_election_change.get("pres_2007", 0.0)) < 0.0,
        "pres_2017_improves": float(per_election_change.get("pres_2017", 0.0)) < 0.0,
    }
    payload = {
        "status": "promote" if all(checks.values()) else "retain_experimental",
        "selection_note": (
            "Fixed one-shot coefficients; outcomes are used only for this declared "
            "through-2022 development promotion test."
        ),
        "checks": checks,
        "baseline": baseline.to_dict(),
        "combined": combined.to_dict(),
        "per_election_national_change_pp": per_election_change.to_dict(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    election_frame.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    national_frame.to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    burden.to_csv(OUTPUT_DIR / "government_burden_scores.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
