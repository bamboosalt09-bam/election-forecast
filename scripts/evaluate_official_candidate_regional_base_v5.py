"""Strict nested ablation for official candidate-history regional evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts", ROOT / "presidential_issue_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import build_speech_derived_candidate_context_v4 as context_builder  # noqa: E402
from scripts import evaluate_speech_derived_candidate_context_v2 as evaluator  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "official_candidate_regional_base_v5_ablation"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v16"
OFFICIAL_BASE = (
    ROOT
    / "data"
    / "raw"
    / "official_sources"
    / "automatic_candidate_regional_base_official.csv"
)


def _metrics(path: Path) -> dict[str, object]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["metrics"]


def _strongest_evidence_union(paths: list[Path], destination: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in ["regional_affinity", "organization_depth", "confidence"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)
    combined["evidence_product"] = (
        combined["regional_affinity"]
        * combined["organization_depth"]
        * combined["confidence"]
    )
    keys = ["election_id", "slot", "candidate_name", "region_id"]
    combined = (
        combined.sort_values([*keys, "evidence_product"], ascending=[True] * len(keys) + [False])
        .drop_duplicates(keys, keep="first")
        .drop(columns="evidence_product")
        .reset_index(drop=True)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False, encoding="utf-8-sig")
    return combined


def _by_election(path: Path, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "by_election.csv", encoding="utf-8-sig")
    frame["variant"] = variant
    return frame[["variant", *[column for column in frame.columns if column != "variant"]]]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_dir = ROOT / "outputs" / "speech_derived_candidate_context_v4"
    context = context_builder.build_context(context_dir)
    combined_path = OUTPUT_DIR / "inputs" / "official_plus_prior_party.csv"
    combined = _strongest_evidence_union(
        [OFFICIAL_BASE, Path(context["candidate_regional_base"])],
        combined_path,
    )

    rows = [{**_metrics(ACTIVE_DIR), "variant": "active_manual_v16"}]
    election_frames = [_by_election(ACTIVE_DIR, "active_manual_v16")]
    for variant, base_path in [
        ("official_history_only", OFFICIAL_BASE),
        ("official_plus_prior_party", combined_path),
    ]:
        variant_dir = OUTPUT_DIR / variant
        payload = evaluator._run(
            context,
            output_dir=variant_dir,
            role_aware=True,
            rejection_routing=False,
            candidate_regional_base_path=base_path,
        )
        rows.append({**payload["metrics"], "variant": variant})
        election_frames.append(_by_election(variant_dir / "active_run", variant))

    summary = pd.DataFrame(rows)
    by_election = pd.concat(election_frames, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    by_election.to_csv(OUTPUT_DIR / "by_election.csv", index=False, encoding="utf-8-sig")
    decision = {
        "experiment": "official_candidate_regional_base_v5",
        "strict_nested": True,
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used_by_new_layer": [],
        "official_input_rows": int(len(pd.read_csv(OFFICIAL_BASE, encoding="utf-8-sig"))),
        "combined_input_rows": int(len(combined)),
        "active_model_changed": False,
        "promotion_rule": (
            "Do not promote unless aggregate and election-level regional metrics "
            "improve without concentrated regression or provenance failure."
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
