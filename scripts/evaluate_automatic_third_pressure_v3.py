"""Compare manual, absent, and automatic third-candidate pressure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import build_speech_derived_candidate_context_v3 as context_builder  # noqa: E402
from scripts import evaluate_speech_derived_candidate_context_v2 as v2_evaluator  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_third_pressure_v3_ablation"
V2_BASELINE = ROOT / "outputs" / "speech_derived_candidate_context_v2" / "active_run"


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _empty_pressure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "election_id",
            "slot",
            "source_slot",
            "transfer_pressure",
            "available_date",
            "confidence",
            "notes",
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _by_election(run_dir: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "by_election.csv", encoding="utf-8-sig")
    frame.insert(0, "pressure_variant", variant)
    return frame


def _build_decision(
    summary: pd.DataFrame,
    by_election: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    manual = summary.loc[
        summary["pressure_variant"].eq("manual_pressure_v2_baseline")
    ].iloc[0]
    automatic = summary.loc[
        summary["pressure_variant"].eq("automatic_pressure")
    ].iloc[0]
    no_pressure = summary.loc[summary["pressure_variant"].eq("no_pressure")].iloc[0]
    metric_columns = ["regional_weighted_mae_pp", "national_candidate_mae_pp"]
    manual_election = by_election.loc[
        by_election["pressure_variant"].eq("manual_pressure_v2_baseline"),
        ["election_id", *metric_columns],
    ].rename(columns={column: f"manual_{column}" for column in metric_columns})
    automatic_election = by_election.loc[
        by_election["pressure_variant"].eq("automatic_pressure"),
        ["election_id", *metric_columns],
    ].rename(columns={column: f"automatic_{column}" for column in metric_columns})
    comparison = manual_election.merge(
        automatic_election, on="election_id", how="inner", validate="one_to_one"
    )
    for column in metric_columns:
        comparison[f"automatic_minus_manual_{column}"] = (
            comparison[f"automatic_{column}"] - comparison[f"manual_{column}"]
        )
    regressed = comparison.loc[
        comparison[
            "automatic_minus_manual_national_candidate_mae_pp"
        ].gt(0.0),
        "election_id",
    ].astype(str).tolist()
    improved = comparison.loc[
        comparison[
            "automatic_minus_manual_national_candidate_mae_pp"
        ].lt(0.0),
        "election_id",
    ].astype(str).tolist()
    decision = {
        "experiment": "automatic_third_pressure_v3",
        "scope": "strict nested pres_2002 through pres_2022",
        "active_model_changed": False,
        "post_2022_outcomes_used": False,
        "manual_third_candidate_pressure_read_by_automatic_variants": False,
        "new_layer_outcome_fields_used": [],
        "manual_metrics": {
            "regional_mae_pp": float(manual["regional_equal_election_macro_mae_pp"]),
            "national_mae_pp": float(manual["national_equal_election_macro_mae_pp"]),
        },
        "no_pressure_metrics": {
            "regional_mae_pp": float(
                no_pressure["regional_equal_election_macro_mae_pp"]
            ),
            "national_mae_pp": float(
                no_pressure["national_equal_election_macro_mae_pp"]
            ),
        },
        "automatic_metrics": {
            "regional_mae_pp": float(
                automatic["regional_equal_election_macro_mae_pp"]
            ),
            "national_mae_pp": float(
                automatic["national_equal_election_macro_mae_pp"]
            ),
        },
        "automatic_improved_elections": improved,
        "automatic_regressed_elections": regressed,
        "diagnosis": (
            "Assembly political axes do not separate 2017 source lanes enough; "
            "retuning the split against the 2017 outcome would be outcome-aware"
        ),
        "promotion_status": "not_promoted_concentrated_regression",
    }
    return decision, comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_dir = ROOT / "outputs" / "speech_derived_candidate_context_v3"
    context = context_builder.build_context(context_dir)
    empty_path = context_dir / "auto_candidate_role" / "empty_third_pressure.csv"
    _empty_pressure(empty_path)

    rows = [
        {
            "pressure_variant": "manual_pressure_v2_baseline",
            **_metrics(V2_BASELINE),
        }
    ]
    election_frames = [_by_election(V2_BASELINE, "manual_pressure_v2_baseline")]
    for variant, pressure_path in [
        ("no_pressure", empty_path),
        ("automatic_pressure", Path(context["third_pressure"])),
    ]:
        variant_dir = OUTPUT_DIR / variant
        payload = v2_evaluator._run(
            context,
            output_dir=variant_dir,
            role_aware=True,
            rejection_routing=True,
            third_pressure_path=pressure_path,
        )
        rows.append({"pressure_variant": variant, **payload["metrics"]})
        election_frames.append(_by_election(variant_dir / "active_run", variant))

    summary = pd.DataFrame(rows)
    by_election = pd.concat(election_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(
        OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig"
    )
    decision, comparison = _build_decision(summary, by_election)
    comparison.to_csv(
        OUTPUT_DIR / "comparison_by_election.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
