"""Build one-field-at-a-time automatic third-character profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "partial_automatic_third_traits_v20"
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
FIELDS = ("centrist_appeal", "anti_major_party_appeal", "regional_base_overlap")


def _build(base: pd.DataFrame, automatic: pd.DataFrame, field: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["election_id", "slot"]
    replacement = automatic[keys + ["candidate_name", field]].rename(
        columns={"candidate_name": "automatic_candidate_name", field: "automatic_value"}
    )
    out = base.merge(replacement, on=keys, how="left", validate="one_to_one")
    matched = out["automatic_value"].notna()
    audit = out.loc[
        matched,
        keys + ["candidate_name", "automatic_candidate_name", field, "automatic_value"],
    ].copy()
    audit = audit.rename(columns={field: "manual_value"})
    audit["replaced_field"] = field
    audit["target_outcome_used"] = False
    out.loc[matched, field] = pd.to_numeric(
        out.loc[matched, "automatic_value"], errors="raise"
    )
    out = out.drop(columns=["automatic_candidate_name", "automatic_value"])
    return out, audit


def main() -> None:
    base = pd.read_csv(BASE, encoding="utf-8-sig")
    automatic = pd.read_csv(AUTOMATIC, encoding="utf-8-sig")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audits = []
    for field in FIELDS:
        profile, audit = _build(base, automatic, field)
        profile.to_csv(
            OUTPUT_DIR / f"third_candidate_profile_{field}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        audits.append(audit)
    combined_audit = pd.concat(audits, ignore_index=True)
    combined_audit.to_csv(
        OUTPUT_DIR / "replacement_audit.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "schema": "partial_automatic_third_traits_v20",
        "base_profile": str(BASE.relative_to(ROOT)),
        "automatic_source": str(AUTOMATIC.relative_to(ROOT)),
        "fields": list(FIELDS),
        "matched_rows_per_field": int(len(automatic)),
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

