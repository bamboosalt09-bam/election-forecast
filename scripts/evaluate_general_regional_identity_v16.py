"""Evaluate a PIT-safe non-Chungcheong regional-distinctiveness layer."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "presidential_issue_engine", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from presidential_issue_engine.chungcheong_identity import CHUNGCHEONG  # noqa: E402
from scripts import run_active_presidential_model as active  # noqa: E402


BASELINE_DIR = ROOT / "outputs" / "active_presidential_nested_v15"
OUTPUT_DIR = ROOT / "outputs" / "regional_identity_v16_camp_donor_experiment"
GAINS = (0.10, 0.25, 0.50)
SHIFT_CAP = 0.04


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _by_election(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for election_id, group in frame.groupby("election_id", sort=True):
        rows.append(
            {
                "variant": label,
                "election_id": election_id,
                "regional_weighted_mae_pp": float(
                    np.average(
                        np.abs(group["layer_pred"] - group["actual"]) * 100.0,
                        weights=group["contest_votes"],
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _by_region(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region_id, group in frame.groupby("region_id", sort=True):
        by_election = []
        for _, election in group.groupby("election_id", sort=False):
            by_election.append(
                float(
                    np.average(
                        np.abs(election["layer_pred"] - election["actual"]) * 100.0,
                        weights=election["contest_votes"],
                    )
                )
            )
        rows.append(
            {
                "variant": label,
                "region_id": region_id,
                "equal_election_weighted_mae_pp": float(np.mean(by_election)),
            }
        )
    return pd.DataFrame(rows)


def _summary(directory: Path) -> dict[str, float]:
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return {
        "regional_macro_mae_pp": float(metrics["regional_equal_election_macro_mae_pp"]),
        "national_macro_mae_pp": float(metrics["national_equal_election_macro_mae_pp"]),
        "winner_accuracy": float(metrics["winner_accuracy"]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_DIR / "nested_predictions.csv", encoding="utf-8-sig")
    baseline_values = _summary(BASELINE_DIR)
    all_elections = [_by_election(baseline, "active_v15")]
    all_regions = [_by_region(baseline, "active_v15")]
    sensitivity: list[dict[str, object]] = []

    for gain in GAINS:
        label = f"gain_{gain:.2f}"
        run_dir = OUTPUT_DIR / label
        if not (run_dir / "summary.json").exists():
            active.run(
                output_dir=run_dir,
                general_regional_identity_gain=gain,
                general_regional_identity_shift_cap=SHIFT_CAP,
            )
        experiment = pd.read_csv(run_dir / "nested_predictions.csv", encoding="utf-8-sig")
        result = _summary(run_dir)
        election_table = pd.concat(
            [_by_election(baseline, "active_v15"), _by_election(experiment, label)],
            ignore_index=True,
        ).pivot(index="election_id", columns="variant", values="regional_weighted_mae_pp")
        election_change = election_table[label] - election_table["active_v15"]
        key = ["election_id", "region_id", "slot"]
        merged = baseline[key + ["layer_pred"]].merge(
            experiment[key + ["layer_pred"]],
            on=key,
            suffixes=("_base", "_experiment"),
            validate="one_to_one",
        )
        chung_delta = np.abs(
            merged.loc[
                merged["region_id"].isin(CHUNGCHEONG),
                "layer_pred_experiment",
            ]
            - merged.loc[
                merged["region_id"].isin(CHUNGCHEONG),
                "layer_pred_base",
            ]
        )
        chung_change = float(chung_delta.max()) if not chung_delta.empty else 0.0
        changes = {name: result[name] - baseline_values[name] for name in result}
        sensitivity.append(
            {
                "variant": label,
                "gain": gain,
                "shift_cap": SHIFT_CAP,
                **result,
                **{f"change_{name}": value for name, value in changes.items()},
                "worst_election_regional_mae_change_pp": float(election_change.max()),
                "best_election_regional_mae_change_pp": float(election_change.min()),
                "maximum_chungcheong_prediction_change": chung_change,
            }
        )
        all_elections.append(_by_election(experiment, label))
        all_regions.append(_by_region(experiment, label))

    sensitivity_frame = pd.DataFrame(sensitivity)
    eligible = sensitivity_frame.loc[
        sensitivity_frame["change_regional_macro_mae_pp"].lt(0.0)
        & sensitivity_frame["change_national_macro_mae_pp"].le(0.05)
        & sensitivity_frame["change_winner_accuracy"].ge(0.0)
        & sensitivity_frame["worst_election_regional_mae_change_pp"].le(0.15)
        & sensitivity_frame["maximum_chungcheong_prediction_change"].le(1e-12)
    ].copy()
    selected = None
    if not eligible.empty:
        selected = eligible.sort_values(
            ["gain", "change_regional_macro_mae_pp"], ascending=[True, True]
        ).iloc[0].to_dict()
    decision = {
        "experiment": "general_regional_identity_v16",
        "strict_point_in_time": True,
        "target_outcomes_used_by_layer": False,
        "chungcheong_excluded": True,
        "candidate_routing_source": "dated_candidate_regional_base",
        "regional_profile_source": "strictly_prior_direct_party_and_downweighted_presidential_ballots",
        "baseline": baseline_values,
        "promotion_gate": {
            "regional_macro_mae_change_pp": "< 0",
            "national_macro_mae_change_pp": "<= 0.05",
            "winner_accuracy_change": ">= 0",
            "worst_election_regional_mae_change_pp": "<= 0.15",
            "maximum_chungcheong_prediction_change": "<= 1e-12",
        },
        "promoted": selected is not None,
        "selected": selected,
    }
    sensitivity_frame.to_csv(
        OUTPUT_DIR / "sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_elections, ignore_index=True).to_csv(
        OUTPUT_DIR / "comparison_by_election.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_regions, ignore_index=True).to_csv(
        OUTPUT_DIR / "comparison_by_region.csv", index=False, encoding="utf-8-sig"
    )
    _atomic_json(decision, OUTPUT_DIR / "decision.json")
    print(json.dumps({"decision": decision, "sensitivity": sensitivity}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
