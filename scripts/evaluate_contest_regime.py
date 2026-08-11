"""Fixed-coefficient promotion test for the contest-regime gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine import contest_regime  # noqa: E402
from scripts import evaluate_preliminary_slot_shadow_nested as nested  # noqa: E402


ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested"
OUTPUT_DIR = ROOT / "outputs" / "contest_regime_experiment"
BASELINE_SNAPSHOT = OUTPUT_DIR / "input_active_v4_predictions.csv"
VARIANT = "fixed_conservative_core_regime"


def _load_baseline() -> pd.DataFrame:
    if BASELINE_SNAPSHOT.exists():
        return pd.read_csv(BASELINE_SNAPSHOT, encoding="utf-8-sig")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary.get("contest_regime_response") is True:
        raise RuntimeError(
            "pre-promotion v4 snapshot is missing; refusing to apply the regime twice"
        )
    baseline = pd.read_csv(
        ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(BASELINE_SNAPSHOT, index=False, encoding="utf-8-sig")
    return baseline


def run() -> dict[str, object]:
    baseline = _load_baseline()
    regimes = contest_regime.derive_contest_regimes(
        baseline, prediction_column="layer_pred"
    )
    evaluated = contest_regime.apply_contest_regime_response(
        baseline,
        regimes,
        prediction_column="layer_pred",
        output_column="experiment_pred",
        expansion_gain=0.50,
        log_shift_cap=0.30,
    )
    base_summary, base_by, base_national = nested._metrics(
        baseline, "layer_pred", "active_v4_baseline"
    )
    new_summary, new_by, new_national = nested._metrics(
        evaluated, "experiment_pred", VARIANT
    )
    base_index = base_by.set_index("election_id")
    new_index = new_by.set_index("election_id")
    national_change = (
        new_index["national_candidate_mae_pp"]
        - base_index["national_candidate_mae_pp"]
    )
    prediction_change = (
        evaluated["experiment_pred"] - evaluated["layer_pred"]
    ).abs()
    close_mask = evaluated["election_id"].isin(["pres_2012", "pres_2022"])
    early_mask = evaluated["election_id"].eq("pres_2002")
    checks = {
        "national_improvement_at_least_0_30pp": float(
            base_summary["national_equal_election_macro_mae_pp"]
            - new_summary["national_equal_election_macro_mae_pp"]
        )
        >= 0.30,
        "regional_improvement_at_least_0_20pp": float(
            base_summary["regional_equal_election_macro_mae_pp"]
            - new_summary["regional_equal_election_macro_mae_pp"]
        )
        >= 0.20,
        "no_election_worsens_over_0_05pp": float(national_change.max()) <= 0.05,
        "winner_accuracy_not_worse": float(new_summary["winner_accuracy"])
        >= float(base_summary["winner_accuracy"]),
        "close_elections_protected": float(prediction_change.loc[close_mask].max())
        <= 0.0005,
        "low_reliability_2002_protected": float(
            prediction_change.loc[early_mask].max()
        )
        <= 1e-12,
        "pres_2007_improves": float(national_change.get("pres_2007", 0.0)) < 0.0,
        "pres_2017_improves": float(national_change.get("pres_2017", 0.0)) < 0.0,
    }
    payload = {
        "status": "promote" if all(checks.values()) else "retain_experimental",
        "selection_note": (
            "One fixed conservative-core specification; no coefficient grid was "
            "selected from presidential outcomes. The promotion decision itself "
            "remains a through-2022 development comparison."
        ),
        "checks": checks,
        "baseline": base_summary,
        "candidate": new_summary,
        "per_election_national_change_pp": national_change.to_dict(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
