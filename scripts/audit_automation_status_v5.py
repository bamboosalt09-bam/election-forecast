"""Inventory active and shadow automation after promotion of active V20."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "automation_status_v5"
CONFIG = ROOT / "data" / "config" / "active_presidential_model_v20.json"


INPUT_STATUS = [
    ("candidate_issue_profile", "automatic_active", "automatic issue interpretation v2"),
    ("mega_issue_attribution", "automatic_active", "automatic issue interpretation v2"),
    ("third_candidate_viability", "mixed_manual_active", "2002/2007/2017 automatic; 2012 final minor and 2022 withdrawn rows retained"),
    ("third_candidate_centrist_appeal", "manual_active", "automatic singleton failed 2007 regional gate"),
    ("third_candidate_anti_major_appeal", "mixed_manual_active", "2002/2007/2017 automatic; 2012/2022 retained"),
    ("third_candidate_regional_overlap", "mixed_manual_active", "2002/2007/2017 automatic; 2012/2022 retained"),
    ("third_candidate_pressure", "manual_active", "automatic pressure failed 2017 asymmetry"),
    ("candidate_regional_base", "automatic_active", "footprint-controlled official history v9"),
    ("chungcheong_identity", "mixed_manual_active", "full-history reservoir and 2007/2017 routing automatic; dated 2002/2012 facts curated"),
    ("party_lineage_channels", "automatic_shadow", "NEC party-level preference/organization compiler; V19 not promoted"),
    ("mega_issue_intensity", "manual_active", "automatic transcript intensity not yet promoted"),
    ("mega_issue_taxonomy", "manual_active", "needs conservative dated event classifier"),
    ("election_generation_weights", "manual_active", "needs KOSIS population and NEC age turnout"),
    ("withdrawal_event_profiles", "mixed_manual_active", "event fact is factual; compliance rate remains manual"),
    ("withdrawn_candidate_transfers", "manual_active", "needs prior-event affinity and timing model"),
    ("candidate_generation_profile", "automatic_active", "Assembly-derived"),
    ("issue_epoch_importance", "automatic_active", "Assembly-derived"),
    ("issue_temporal_conversion", "automatic_active", "Assembly-derived"),
    ("candidate_political_landscape", "mostly_automatic_active", "remaining withdrawal rows are curated"),
]

AUTOMATIC_ACTIVE_PARAMETERS = {
    "postprocess.contest_regime_expansion_gain",
    "postprocess.contest_regime_log_shift_cap",
    "postprocess.contest_regime_swing_log_shift_cap",
}
SAFETY_PARAMETER_SUFFIXES = (
    "gain_cap",
    "shift_cap",
    "score_cap",
    "minimum_intensity",
    "minimum_prior_scored_elections",
    "vif_threshold",
    "other_lineage_core",
    "third_absolute_cap",
    "third_to_second_cap",
)


def _flatten_numeric(value: object, prefix: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_numeric(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if prefix in AUTOMATIC_ACTIVE_PARAMETERS:
            status = "prior_only_automatic_active"
        elif prefix.endswith(SAFETY_PARAMETER_SUFFIXES):
            status = "safety_bound_document_and_keep"
        else:
            status = "behavioral_parameter_pending_automation"
        rows.append({"parameter": prefix, "value": value, "automation_status": status})
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = pd.DataFrame(
        INPUT_STATUS, columns=["control", "automation_status", "replacement_or_next_source"]
    )
    parameters = pd.DataFrame(
        _flatten_numeric(json.loads(CONFIG.read_text(encoding="utf-8")))
    )
    inputs.to_csv(OUTPUT_DIR / "input_status.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(
        OUTPUT_DIR / "parameter_status.csv", index=False, encoding="utf-8-sig"
    )
    counts = inputs["automation_status"].value_counts().to_dict()
    summary = {
        "active_model": "v20",
        "input_controls": int(len(inputs)),
        "input_status_counts": {str(key): int(value) for key, value in counts.items()},
        "fixed_numeric_parameters": int(len(parameters)),
        "prior_only_automatic_active_parameters": int(parameters["automation_status"].eq("prior_only_automatic_active").sum()),
        "safety_bounds": int(parameters["automation_status"].eq("safety_bound_document_and_keep").sum()),
        "behavioral_parameters_pending": int(parameters["automation_status"].eq("behavioral_parameter_pending_automation").sum()),
        "post_2022_outcomes_used": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
