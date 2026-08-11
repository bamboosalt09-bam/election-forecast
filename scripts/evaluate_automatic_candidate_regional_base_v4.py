"""Compare manual, absent, and automatic candidate regional bases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import build_speech_derived_candidate_context_v4 as context_builder  # noqa: E402
from scripts import evaluate_speech_derived_candidate_context_v2 as v2_evaluator  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "automatic_candidate_regional_base_v4_ablation"
V2_BASELINE = ROOT / "outputs" / "speech_derived_candidate_context_v2" / "active_run"


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _empty_base(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "election_id",
            "slot",
            "candidate_name",
            "region_id",
            "regional_affinity",
            "organization_depth",
            "available_date",
            "confidence",
            "source_type",
            "notes",
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def _by_election(run_dir: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "by_election.csv", encoding="utf-8-sig")
    frame.insert(0, "regional_base_variant", variant)
    return frame


def _build_decision(
    summary: pd.DataFrame,
    by_election: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    variants = summary.set_index("regional_base_variant")
    manual = variants.loc["manual_regional_base_v2_baseline"]
    absent = variants.loc["no_candidate_regional_base"]
    automatic = variants.loc["automatic_nonmajor_regional_base"]
    metrics = ["regional_weighted_mae_pp", "national_candidate_mae_pp"]
    comparison = by_election.loc[
        by_election["regional_base_variant"].isin(
            ["manual_regional_base_v2_baseline", "automatic_nonmajor_regional_base"]
        ),
        ["regional_base_variant", "election_id", *metrics],
    ].pivot(index="election_id", columns="regional_base_variant", values=metrics)
    comparison.columns = [
        f"{variant}_{metric}" for metric, variant in comparison.columns
    ]
    comparison = comparison.reset_index()
    for metric in metrics:
        comparison[f"automatic_minus_manual_{metric}"] = (
            comparison[f"automatic_nonmajor_regional_base_{metric}"]
            - comparison[f"manual_regional_base_v2_baseline_{metric}"]
        )
    regression_column = "automatic_minus_manual_regional_weighted_mae_pp"
    regressed = comparison.loc[
        comparison[regression_column].gt(0.0), "election_id"
    ].astype(str).tolist()
    improved = comparison.loc[
        comparison[regression_column].lt(0.0), "election_id"
    ].astype(str).tolist()
    decision = {
        "experiment": "automatic_candidate_regional_base_v4",
        "scope": "strict nested pres_2002 through pres_2022",
        "active_model_changed": False,
        "post_2022_outcomes_used": False,
        "manual_candidate_regional_base_read_by_automatic_variants": False,
        "new_layer_outcome_fields_used": [],
        "manual_metrics": {
            "regional_mae_pp": float(manual["regional_equal_election_macro_mae_pp"]),
            "national_mae_pp": float(manual["national_equal_election_macro_mae_pp"]),
        },
        "no_base_metrics": {
            "regional_mae_pp": float(absent["regional_equal_election_macro_mae_pp"]),
            "national_mae_pp": float(absent["national_equal_election_macro_mae_pp"]),
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
        "promotion_status": "not_promoted_missing_personal_regional_history",
        "automatic_component_status": (
            "validated_for_nonmajor_party_organization_only"
        ),
        "diagnosis": (
            "The prior direct-party ballot component identifies non-major-party "
            "organization in 2002 and 2017 without outcomes, but it cannot replace "
            "personal political bases such as Roh Moo-hyun in PK, Lee Hoi-chang in "
            "Chungcheong, or Lee Jae-myung in Gyeonggi. Replacing the mixed manual "
            "file therefore improves 2017 while causing concentrated regional "
            "regression in 2002, 2007, and 2022."
        ),
        "next_step": (
            "Keep active v16 unchanged. Split party-organization evidence from "
            "dated candidate office and constituency history, then derive the "
            "personal component from a factual no-strength input before another "
            "strict nested ablation."
        ),
    }
    return decision, comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_dir = ROOT / "outputs" / "speech_derived_candidate_context_v4"
    context = context_builder.build_context(context_dir)
    empty_path = context_dir / "auto_candidate_role" / "empty_candidate_regional_base.csv"
    _empty_base(empty_path)

    rows = [
        {
            "regional_base_variant": "manual_regional_base_v2_baseline",
            **_metrics(V2_BASELINE),
        }
    ]
    election_frames = [_by_election(V2_BASELINE, "manual_regional_base_v2_baseline")]
    for variant, regional_base_path in [
        ("no_candidate_regional_base", empty_path),
        (
            "automatic_nonmajor_regional_base",
            Path(context["candidate_regional_base"]),
        ),
    ]:
        variant_dir = OUTPUT_DIR / variant
        payload = v2_evaluator._run(
            context,
            output_dir=variant_dir,
            role_aware=True,
            rejection_routing=True,
            candidate_regional_base_path=regional_base_path,
        )
        rows.append({"regional_base_variant": variant, **payload["metrics"]})
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
