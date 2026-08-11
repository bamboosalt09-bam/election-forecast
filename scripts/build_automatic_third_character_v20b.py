"""Build automatic anti-major and regional-overlap traits, retaining centrist appeal."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "automatic_third_character_v20b"
BASE = (
    ROOT
    / "outputs"
    / "election_derived_third_candidate_profile_v14b"
    / "third_candidate_profile.csv"
)
AUTOMATIC = (
    ROOT
    / "outputs"
    / "election_derived_third_candidate_profile_v15"
    / "third_candidate_profile.csv"
)
AUTOMATIC_FIELDS = ("anti_major_party_appeal", "regional_base_overlap")


def main() -> None:
    base = pd.read_csv(BASE, encoding="utf-8-sig")
    automatic = pd.read_csv(AUTOMATIC, encoding="utf-8-sig")
    keys = ["election_id", "slot"]
    replacement = automatic[keys + ["candidate_name", *AUTOMATIC_FIELDS]].rename(
        columns={
            "candidate_name": "automatic_candidate_name",
            **{field: f"automatic_{field}" for field in AUTOMATIC_FIELDS},
        }
    )
    out = base.merge(replacement, on=keys, how="left", validate="one_to_one")
    matched = out["automatic_candidate_name"].notna()
    audit_rows = []
    for field in AUTOMATIC_FIELDS:
        automatic_field = f"automatic_{field}"
        for row in out.loc[matched, keys + ["candidate_name", field, automatic_field]].itertuples(index=False):
            audit_rows.append(
                {
                    "election_id": row.election_id,
                    "slot": row.slot,
                    "candidate_name": row.candidate_name,
                    "field": field,
                    "manual_value": getattr(row, field),
                    "automatic_value": getattr(row, automatic_field),
                    "target_outcome_used": False,
                }
            )
        out.loc[matched, field] = pd.to_numeric(
            out.loc[matched, automatic_field], errors="raise"
        )
    out = out.drop(
        columns=["automatic_candidate_name", *[f"automatic_{field}" for field in AUTOMATIC_FIELDS]]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        OUTPUT_DIR / "third_candidate_profile.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(audit_rows).to_csv(
        OUTPUT_DIR / "replacement_audit.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema": "automatic_third_character_v20b",
        "automatic_fields": list(AUTOMATIC_FIELDS),
        "manual_fields_retained": ["centrist_appeal"],
        "matched_profile_rows": int(matched.sum()),
        "post_2022_outcomes_used": False,
        "target_outcome_fields_used": [],
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

