"""Inventory manual, derived, and fixed-weight inputs used by the candidate v2 run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "manual_weight_lineage_audit_v2"
RUN_MANIFEST = (
    ROOT
    / "outputs"
    / "speech_derived_candidate_context_v2"
    / "active_run"
    / "input_manifest.csv"
)
CONFIG = ROOT / "data" / "config" / "active_presidential_model_v16.json"


INPUTS = [
    ("data/raw/candidate_issue_profile.csv", "candidate issue direction/strength", "manual target rows", "replaced in speech candidate v2", "automatic Assembly issue profile"),
    ("data/raw/mega_issue_attribution.csv", "mega-issue target attribution", "manual target rows", "replaced in speech candidate v2", "automatic explicit target attribution"),
    ("data/raw/third_candidate_profile.csv", "third-candidate stature", "manual candidate rows", "replaced in speech candidate v2", "automatic speech stature and political landscape"),
    ("data/raw/third_candidate_pressure.csv", "third-candidate source-lane pressure", "manual election-slot rows", "still read; automatic v3 rejected", "requires a general candidate-affinity by source-camp-rupture interaction; do not retune from 2017 outcome"),
    ("data/raw/candidate_regional_base.csv", "candidate-specific regional base", "manual dated candidate-region facts and strengths", "still read; automatic v4 party-organization component validated but full replacement rejected", "separate the validated prior-party organization signal from a factual no-strength office/constituency history input"),
    ("data/raw/chungcheong_identity_alignment.csv", "Chungcheong recipient routing", "dated curated event facts", "still read", "replace numeric affinity with event evidence strength"),
    ("data/raw/mega_issue_intensity.csv", "election-wide shock intensity", "manual election scalar", "still read; automatic speech v5 and event-class v5b not promoted", "retain the continuous activation fix; validate the automatic compiler outside the five presidential outcomes before replacement"),
    ("data/raw/mega_issue_taxonomy.csv", "shock type and severity", "manual event taxonomy", "still read", "derive severity fields from issue-character evidence while retaining factual event identity"),
    ("data/raw/election_generation_weights.csv", "electorate age composition", "rough manual election shares", "still read", "replace with dated demographic and turnout history"),
    ("data/raw/withdrawal_event_profiles.csv", "withdrawal and endorsement transfer", "factual events plus manual behavioral rates", "still read", "retain event facts; estimate compliance from prior comparable events"),
    ("data/raw/withdrawn_candidate_transfers.csv", "withdrawn-candidate target split", "manual behavioral rates", "still read", "derive target affinity and compliance from political vectors and event timing"),
    ("data/raw/candidate_generation_profile.csv", "candidate generation affinity", "Assembly-derived", "still read", "already automatic; audit upstream sensitivity map later"),
    ("data/raw/issue_epoch_importance.csv", "issue epoch importance", "Assembly-derived", "still read", "already automatic"),
    ("data/raw/issue_temporal_conversion.csv", "time-varying issue conversion", "Assembly-derived", "still read", "already automatic"),
    ("data/raw/candidate_political_landscape.csv", "candidate political vector", "mostly Assembly-derived", "still read", "remove remaining curated withdrawn-candidate rows"),
]


def _flatten_numeric(value: object, prefix: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append(
            {
                "parameter": prefix,
                "value": value,
                "scope": "universal fixed parameter",
                "status": "not automatically estimated",
                "selection_warning": "historically developed on through-2022 sample",
            }
        )
    return rows


def main() -> None:
    used = set()
    if RUN_MANIFEST.exists():
        used = set(pd.read_csv(RUN_MANIFEST)["path"].astype(str))
    rows = []
    for path, purpose, provenance, status, next_step in INPUTS:
        candidate = ROOT / path
        rows.append(
            {
                "path": path,
                "purpose": purpose,
                "provenance": provenance,
                "status": status,
                "read_by_candidate_v2_run": path in used,
                "exists": candidate.exists(),
                "next_step": next_step,
            }
        )
    inputs = pd.DataFrame(rows)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parameters = pd.DataFrame(_flatten_numeric(config))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs.to_csv(OUTPUT_DIR / "input_lineage.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(
        OUTPUT_DIR / "fixed_numeric_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "candidate_v2_input_rows": int(len(inputs)),
        "candidate_v2_inputs_read": int(inputs["read_by_candidate_v2_run"].sum()),
        "manual_or_curated_inputs_still_read": int(
            inputs.loc[
                inputs["read_by_candidate_v2_run"]
                & ~inputs["provenance"].str.contains("derived", case=False),
            ].shape[0]
        ),
        "fixed_numeric_parameter_count": int(len(parameters)),
        "automatic_migration_policy": (
            "replace one lineage at a time and retain only when strict nested "
            "all-election diagnostics do not show concentrated regression"
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
