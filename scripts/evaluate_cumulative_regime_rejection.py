"""Fixed promotion test for broad, coherent government-rejection evidence."""

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

from presidential_issue_engine import contest_regime  # noqa: E402
from presidential_issue_engine import incumbent_shock_adjustment  # noqa: E402
from scripts import evaluate_preliminary_slot_shadow_nested as nested  # noqa: E402


ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested"
OUTPUT_DIR = ROOT / "outputs" / "cumulative_regime_rejection_experiment"
BASELINE_SNAPSHOT = OUTPUT_DIR / "input_active_v5_predictions.csv"
PROFILE = ROOT / "data" / "raw" / "auto_issue_seed" / "candidate_issue_profile.csv"


def _load_baseline() -> pd.DataFrame:
    if BASELINE_SNAPSHOT.exists():
        return pd.read_csv(BASELINE_SNAPSHOT, encoding="utf-8-sig")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary.get("cumulative_regime_rejection") is True:
        raise RuntimeError("pre-promotion v5 snapshot is missing")
    frame = pd.read_csv(ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(BASELINE_SNAPSHOT, index=False, encoding="utf-8-sig")
    return frame


def _remove_existing_contest_response(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover the pre-v5 contest predictions from the recorded bounded shift."""

    required = {
        "election_id",
        "region_id",
        "layer_pred",
        "contest_regime_log_shift",
        "regime_core_floor",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"v5 snapshot cannot be inverted; missing {sorted(missing)}")
    out = frame.copy().reset_index(drop=True)
    out["precontest_pred"] = pd.to_numeric(
        out["layer_pred"], errors="coerce"
    ).fillna(0.0)
    for _, group in out.groupby(["election_id", "region_id"], sort=False):
        shifts = pd.to_numeric(
            group["contest_regime_log_shift"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        active = np.flatnonzero(np.abs(shifts) > 1e-15)
        if len(active) != 2:
            continue
        indices = group.index.to_numpy()[active]
        final_pair = out.loc[indices, "precontest_pred"].to_numpy(float)
        floors = pd.to_numeric(
            out.loc[indices, "regime_core_floor"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        final_flexible = np.clip(final_pair - floors, 1e-12, None)
        original_flexible = final_flexible * np.exp(-shifts[active])
        original_flexible *= float(final_flexible.sum()) / max(
            float(original_flexible.sum()), 1e-12
        )
        out.loc[indices, "precontest_pred"] = floors + original_flexible
    return out


def run() -> dict[str, object]:
    baseline = _load_baseline()
    precontest = _remove_existing_contest_response(baseline)
    profile = pd.read_csv(PROFILE, encoding="utf-8-sig")
    burden = incumbent_shock_adjustment.compile_government_burden_scores(
        profile, nested.engine.ELECTION_DATES
    ).rename(columns={"slot": "source_slot"})
    value_columns = [
        column
        for column in burden.columns
        if column not in {"election_id", "source_slot"}
    ]
    prepared = precontest.drop(
        columns=[column for column in value_columns if column in precontest.columns]
    ).merge(burden, on=["election_id", "source_slot"], how="left")
    for column in value_columns:
        prepared[column] = pd.to_numeric(
            prepared[column], errors="coerce"
        ).fillna(0.0)

    regimes = contest_regime.derive_contest_regimes(
        prepared, prediction_column="precontest_pred"
    )
    evaluated = contest_regime.apply_contest_regime_response(
        prepared,
        regimes,
        prediction_column="precontest_pred",
        output_column="experiment_pred",
        expansion_gain=0.50,
        log_shift_cap=0.40,
    )
    base_summary, base_by, base_national = nested._metrics(
        baseline, "layer_pred", "active_v5_baseline"
    )
    new_summary, new_by, new_national = nested._metrics(
        evaluated, "experiment_pred", "cumulative_regime_rejection"
    )
    base_index = base_by.set_index("election_id")
    new_index = new_by.set_index("election_id")
    national_change = (
        new_index["national_candidate_mae_pp"]
        - base_index["national_candidate_mae_pp"]
    )
    changed = (evaluated["experiment_pred"] - baseline["layer_pred"]).abs()
    protected = evaluated["election_id"].isin(
        ["pres_2002", "pres_2012", "pres_2022"]
    )
    third = evaluated["source_slot"].eq("C")
    checks = {
        "national_improvement_at_least_0_20pp": float(
            base_summary["national_equal_election_macro_mae_pp"]
            - new_summary["national_equal_election_macro_mae_pp"]
        )
        >= 0.20,
        "regional_improvement_at_least_0_15pp": float(
            base_summary["regional_equal_election_macro_mae_pp"]
            - new_summary["regional_equal_election_macro_mae_pp"]
        )
        >= 0.15,
        "protected_elections_unchanged": float(changed.loc[protected].max())
        <= 1e-12,
        "third_candidates_unchanged": float(changed.loc[third].max()) <= 1e-12,
        "winner_accuracy_not_worse": float(new_summary["winner_accuracy"])
        >= float(base_summary["winner_accuracy"]),
        "pres_2007_improves": float(national_change.get("pres_2007", 0.0)) < 0.0,
        "pres_2017_improves": float(national_change.get("pres_2017", 0.0)) < 0.0,
    }
    payload = {
        "status": "promote" if all(checks.values()) else "retain_experimental",
        "selection_note": (
            "One fixed cumulative-rejection formula; no presidential-outcome grid. "
            "The recorded v5 contest shift is inverted before applying v6 once. "
            "Promotion remains a through-2022 development comparison."
        ),
        "checks": checks,
        "baseline": base_summary,
        "candidate": new_summary,
        "per_election_national_change_pp": national_change.to_dict(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    burden.to_csv(
        OUTPUT_DIR / "government_burden_scores.csv", index=False, encoding="utf-8-sig"
    )
    regimes.to_csv(OUTPUT_DIR / "contest_regimes.csv", index=False, encoding="utf-8-sig")
    evaluated.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat([base_by, new_by], ignore_index=True).to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([base_national, new_national], ignore_index=True).to_csv(
        OUTPUT_DIR / "national_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
