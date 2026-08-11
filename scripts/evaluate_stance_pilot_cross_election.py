"""Apply the fixed 2022 stance-shadow specification across scored elections.

The experiment uses one 3,000-row stratified batch per election and the current
canonical rolling predictions. It is non-PIT and its rule labels are not human
validated, so it must not update active model metrics or coefficients.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from presidential_issue_engine import issue_vote_engine as engine  # noqa: E402
from scripts.evaluate_stance_pilot_3000_sensitivity import CONFIGS, build_features  # noqa: E402


ELECTIONS = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
SCALE = 0.60
DEFAULT_SAMPLE_SIZE = 3000
BASELINE_INPUT = (
    ROOT
    / "presidential_issue_engine"
    / "report"
    / "tables"
    / "issue_vote_engine_rolling_predictions.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "assembly_stance" / "cross_election_scale_060"


def _weighted_average(group: pd.DataFrame, column: str) -> float:
    weights = pd.to_numeric(group["contest_votes"], errors="coerce").fillna(0.0).to_numpy(float)
    values = pd.to_numeric(group[column], errors="coerce").fillna(0.0).to_numpy(float)
    return float(np.average(values, weights=weights)) if weights.sum() > 0 else float(values.mean())


def evaluate_config(
    baseline: pd.DataFrame,
    config: dict[str, object],
    *,
    sample_size: int,
    scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []

    for election_id in ELECTIONS:
        pilot = ROOT / "outputs" / "assembly_stance" / f"pilot_{election_id}_{sample_size}" / "review_batch.csv"
        features = build_features(config, pilot_input=pilot)
        feature_columns = [
            "election_id", "slot", "candidate_name", "evidence_count", "attention_count",
            "person_evidence_count", "party_evidence_count", "person_attention_count",
            "party_attention_count", "person_signal", "party_signal", "stance_shadow_signal",
            "covered_slot_count", "coverage_gate_passed",
            "speaker_context_mapped_count", "same_bloc_count", "other_bloc_count",
            "context_neutral_count", "context_issue_overlap_count",
            "global_context_neutral_count", "global_context_speaker_count",
            "global_context_committee_count", "global_context_period_count",
            "global_context_bloc_count", "global_context_analysis_count",
            "global_context_impact_count", "global_context_issue_overlap_count",
            "global_context_structure_strength", "global_context_content_strength",
            "global_context_strength", "global_context_relative_strength",
        ]
        election_features = features.loc[features["election_id"].eq(election_id)].reindex(
            columns=feature_columns,
            fill_value=0.0,
        ).copy()
        election_features.insert(0, "config", str(config["name"]))
        coverage_frames.append(election_features)

        rows = baseline.loc[baseline["election_id"].eq(election_id)].copy()
        rows = rows.merge(
            election_features[["election_id", "slot", "stance_shadow_signal"]],
            on=["election_id", "slot"],
            how="left",
        )
        rows["stance_shadow_signal"] = pd.to_numeric(
            rows["stance_shadow_signal"], errors="coerce"
        ).fillna(0.0)
        rows["shadow_pred"] = engine.normalize_vote_share_predictions(
            rows,
            rows["pred"].to_numpy(float) + scale * rows["stance_shadow_signal"].to_numpy(float),
        )
        rows["baseline_abs_err_pp"] = np.abs(rows["pred"] - rows["actual"]) * 100.0
        rows["shadow_abs_err_pp"] = np.abs(rows["shadow_pred"] - rows["actual"]) * 100.0
        rows.insert(0, "config", str(config["name"]))
        row_frames.append(rows)

    row_results = pd.concat(row_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    election_summary_rows: list[dict[str, float | str | int]] = []
    candidate_rows: list[dict[str, float | str]] = []
    for election_id, group in row_results.groupby("election_id", sort=False):
        national_entries = []
        for (slot, candidate_name), candidate in group.groupby(["slot", "candidate_name"], sort=False):
            baseline_pred = _weighted_average(candidate, "pred")
            shadow_pred = _weighted_average(candidate, "shadow_pred")
            actual = _weighted_average(candidate, "actual")
            national_entries.append((baseline_pred, shadow_pred, actual))
            candidate_rows.append({
                "config": str(config["name"]),
                "election_id": str(election_id),
                "slot": str(slot),
                "candidate_name": str(candidate_name),
                "baseline_pred_pct": baseline_pred * 100.0,
                "shadow_pred_pct": shadow_pred * 100.0,
                "actual_pct": actual * 100.0,
                "baseline_err_pp": (baseline_pred - actual) * 100.0,
                "shadow_err_pp": (shadow_pred - actual) * 100.0,
            })
        national_array = np.asarray(national_entries, dtype=float)
        baseline_national_mae = float(np.abs(national_array[:, 0] - national_array[:, 2]).mean() * 100.0)
        shadow_national_mae = float(np.abs(national_array[:, 1] - national_array[:, 2]).mean() * 100.0)
        election_summary_rows.append({
            "config": str(config["name"]),
            "election_id": str(election_id),
            "n_rows": int(len(group)),
            "baseline_row_mae_pp": float(group["baseline_abs_err_pp"].mean()),
            "shadow_row_mae_pp": float(group["shadow_abs_err_pp"].mean()),
            "row_mae_change_pp": float(group["shadow_abs_err_pp"].mean() - group["baseline_abs_err_pp"].mean()),
            "baseline_national_mae_pp": baseline_national_mae,
            "shadow_national_mae_pp": shadow_national_mae,
            "national_mae_change_pp": shadow_national_mae - baseline_national_mae,
            "mean_abs_prediction_shift_pp": float(np.abs(group["shadow_pred"] - group["pred"]).mean() * 100.0),
        })
    return row_results, pd.DataFrame(election_summary_rows), pd.DataFrame(candidate_rows), coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-regex", help="Run only configuration names matching this regex.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            OUTPUT_DIR
            if args.sample_size == DEFAULT_SAMPLE_SIZE and args.scale == SCALE
            else ROOT
            / "outputs"
            / "assembly_stance"
            / f"cross_election_{args.sample_size}_scale_{int(round(args.scale * 100)):03d}"
        )
    baseline = pd.read_csv(BASELINE_INPUT)
    baseline = baseline.loc[baseline["election_id"].isin(ELECTIONS)].copy()
    if len(baseline) != 215:
        raise RuntimeError(f"expected 215 canonical rolling rows, found {len(baseline)}")

    all_rows: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_candidates: list[pd.DataFrame] = []
    all_coverage: list[pd.DataFrame] = []
    selected_configs = list(CONFIGS)
    if args.config_regex:
        pattern = re.compile(args.config_regex)
        selected_configs = [config for config in selected_configs if pattern.search(str(config["name"]))]
        if not selected_configs:
            raise ValueError(f"no configurations matched: {args.config_regex}")
    for config in selected_configs:
        rows, summary, candidates, coverage = evaluate_config(
            baseline,
            config,
            sample_size=args.sample_size,
            scale=args.scale,
        )
        all_rows.append(rows)
        all_summaries.append(summary)
        all_candidates.append(candidates)
        all_coverage.append(coverage)

    row_results = pd.concat(all_rows, ignore_index=True)
    summaries = pd.concat(all_summaries, ignore_index=True)
    candidates = pd.concat(all_candidates, ignore_index=True)
    coverage = pd.concat(all_coverage, ignore_index=True)
    direction = candidates.merge(
        coverage[
            [
                "config", "election_id", "slot", "stance_shadow_signal", "evidence_count",
                "speaker_context_mapped_count", "same_bloc_count", "other_bloc_count",
            ]
        ],
        on=["config", "election_id", "slot"],
        how="left",
    )
    direction["needed_correction_direction"] = np.sign(
        direction["actual_pct"] - direction["baseline_pred_pct"]
    )
    direction["stance_signal_direction"] = np.sign(direction["stance_shadow_signal"])
    direction["direction_evaluable"] = direction["stance_signal_direction"].ne(0).astype(int)
    direction["direction_aligned"] = (
        direction["needed_correction_direction"].eq(direction["stance_signal_direction"])
        & direction["stance_signal_direction"].ne(0)
    ).astype(int)
    direction_summary_rows: list[dict[str, float | str | int]] = []
    for config_name, group in direction.groupby("config", sort=False):
        evaluable = group.loc[group["direction_evaluable"].eq(1)].copy()
        residual = (evaluable["actual_pct"] - evaluable["baseline_pred_pct"]).to_numpy(float)
        signal = evaluable["stance_shadow_signal"].to_numpy(float)
        pearson = float(np.corrcoef(signal, residual)[0, 1]) if len(evaluable) >= 2 else float("nan")
        signal_rank = pd.Series(signal).rank().to_numpy(float)
        residual_rank = pd.Series(residual).rank().to_numpy(float)
        spearman = float(np.corrcoef(signal_rank, residual_rank)[0, 1]) if len(evaluable) >= 2 else float("nan")
        direction_summary_rows.append({
            "config": str(config_name),
            "evaluable_slots": int(len(evaluable)),
            "aligned_slots": int(evaluable["direction_aligned"].sum()),
            "direction_alignment_rate": float(evaluable["direction_aligned"].mean()) if len(evaluable) else float("nan"),
            "residual_signal_pearson": pearson,
            "residual_signal_spearman": spearman,
        })
    direction_summary = pd.DataFrame(direction_summary_rows)
    aggregate_rows: list[dict[str, float | str | int]] = []
    for config_name, group in row_results.groupby("config", sort=False):
        holdout = group.loc[~group["election_id"].eq("pres_2022")]
        for scope, scoped in (("all_2002_2022", group), ("other_2002_2017", holdout)):
            aggregate_rows.append({
                "config": str(config_name),
                "scope": scope,
                "n_rows": int(len(scoped)),
                "baseline_row_mae_pp": float(scoped["baseline_abs_err_pp"].mean()),
                "shadow_row_mae_pp": float(scoped["shadow_abs_err_pp"].mean()),
                "row_mae_change_pp": float(scoped["shadow_abs_err_pp"].mean() - scoped["baseline_abs_err_pp"].mean()),
                "mean_abs_prediction_shift_pp": float(np.abs(scoped["shadow_pred"] - scoped["pred"]).mean() * 100.0),
            })
    aggregate = pd.DataFrame(aggregate_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    row_results.to_csv(output_dir / "row_predictions.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(output_dir / "election_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output_dir / "candidate_national.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "feature_coverage.csv", index=False, encoding="utf-8-sig")
    direction.to_csv(output_dir / "direction_diagnostics.csv", index=False, encoding="utf-8-sig")
    direction_summary.to_csv(output_dir / "direction_alignment_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False, encoding="utf-8-sig")

    tuned_2022_name = "person_party_attention_070"
    robust_name = "person_party_speaker_confirmed_gate2"
    tuned_2022 = summaries.loc[summaries["config"].eq(tuned_2022_name)].copy()
    robust = summaries.loc[summaries["config"].eq(robust_name)].copy()
    robust_aggregate = aggregate.loc[aggregate["config"].eq(robust_name)].copy()
    neutral_boundary_names = [
        "person_party_speaker_confirmed_conf3_gate2",
        "person_party_speaker_confirmed_conf3_context020_gate2",
        "person_party_speaker_confirmed_conf3_context035_gate2",
        "person_party_speaker_confirmed_conf3_context050_gate2",
        "person_party_speaker_confirmed_conf3_context075_gate2",
        "person_party_speaker_confirmed_conf3_context100_gate2",
        "person_party_speaker_confirmed_conf3_context125_gate2",
        "person_party_speaker_confirmed_conf3_context150_gate2",
    ]
    neutral_boundary = summaries.loc[summaries["config"].isin(neutral_boundary_names)].copy()
    neutral_boundary_aggregate = aggregate.loc[aggregate["config"].isin(neutral_boundary_names)].copy()
    neutral_information = summaries.loc[
        summaries["config"].str.contains("globalrel|issueglobal|_global", regex=True)
    ].copy()
    neutral_information_aggregate = aggregate.loc[
        aggregate["config"].str.contains("globalrel|issueglobal|_global", regex=True)
    ].copy()
    report = [
        f"# Cross-Election {args.sample_size:,}-Sentence Stance Shadow",
        "",
        f"Sample size is {args.sample_size} sentences per election.",
        f"Scale is fixed at {args.scale:.2f} for every election; no election-specific retuning is performed.",
        "This is a stratified, rule-labelled, non-PIT diagnostic and is not an active-model metric.",
        "",
        "## Cross-Election Candidate",
        "",
        "Candidate + party directional signal, party weight 0.65, minimum two covered slots, no neutral-attention boost.",
        "Speaker bloc is used as confirmation: agreement strengthens up to 15%, conflict halves the signal.",
        "",
        robust.to_csv(index=False),
        "",
        "## Cross-Election Aggregate",
        "",
        robust_aggregate.to_csv(index=False),
        "",
        "## Neutral Context Boundary Diagnostic",
        "",
        "Neutral rows never create or reverse direction. They only strengthen an existing directional signal when target and issue both match.",
        "Confidence power 3 differentiates stronger rule evidence; context gains 0.00-1.50 locate the diagnostic saturation boundary.",
        "These settings were compared post hoc and are not confirmed out-of-sample performance.",
        "",
        neutral_boundary_aggregate.to_csv(index=False),
        "",
        neutral_boundary.to_csv(index=False),
        "",
        "## Neutral Information Extraction Diagnostic",
        "",
        "Untargeted neutral rows contribute issue volume, speaker and committee breadth, persistence, cross-bloc spread, and analysis/impact cues.",
        "They can only reweight an already directional candidate or party issue; they cannot create or reverse vote direction.",
        "",
        neutral_information_aggregate.to_csv(index=False),
        "",
        neutral_information.to_csv(index=False),
        "",
        "## 2022-Tuned High-Attention Diagnostic",
        "",
        tuned_2022.to_csv(index=False),
        "",
        "## All Configurations",
        "",
        summaries.to_csv(index=False),
        "",
        "## Direction Alignment",
        "",
        direction_summary.to_csv(index=False),
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "sample_size": args.sample_size,
        "scale": args.scale,
        "output_dir": str(output_dir),
        "cross_election_candidate": robust_name,
        "elections": robust[["election_id", "baseline_row_mae_pp", "shadow_row_mae_pp", "row_mae_change_pp"]].to_dict("records"),
        "aggregate": robust_aggregate.to_dict("records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
