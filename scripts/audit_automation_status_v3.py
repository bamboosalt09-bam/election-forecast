"""Inventory active manual controls and validated automatic replacements."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "automation_status_v3"
CONFIG = ROOT / "data" / "config" / "active_presidential_model_v16.json"


INPUT_STATUS = [
    ("candidate_issue_profile", "automatic_active", "automatic_issue_interpretation_v2"),
    ("mega_issue_attribution", "automatic_active", "automatic_issue_interpretation_v2"),
    ("third_candidate_profile", "automatic_candidate", "speech and political-vector compiler"),
    ("third_candidate_pressure", "manual_active", "automatic same-lane pressure v3 previously rejected"),
    ("candidate_regional_base", "automatic_candidate", "footprint-controlled official history v9"),
    ("chungcheong_identity_alignment", "manual_active", "needs dated biographical and regional-affinity evidence"),
    ("mega_issue_intensity", "manual_active", "automatic transcript intensity v5 not promoted"),
    ("mega_issue_taxonomy", "manual_active", "needs conservative event classifier"),
    ("election_generation_weights", "manual_active", "needs KOSIS age population and NEC turnout"),
    ("withdrawal_event_profiles", "mixed_manual_active", "event fact is factual; compliance rate is manual"),
    ("withdrawn_candidate_transfers", "manual_active", "needs prior-event affinity and timing model"),
    ("candidate_generation_profile", "automatic_active", "Assembly-derived"),
    ("issue_epoch_importance", "automatic_active", "Assembly-derived"),
    ("issue_temporal_conversion", "automatic_active", "Assembly-derived"),
    ("candidate_political_landscape", "mostly_automatic_active", "remaining withdrawn-candidate rows are curated"),
]

AUTOMATIC_PARAMETER_CANDIDATES = {
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
        if prefix in AUTOMATIC_PARAMETER_CANDIDATES:
            status = "prior_only_automatic_candidate"
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
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parameters = pd.DataFrame(_flatten_numeric(config))
    inputs.to_csv(OUTPUT_DIR / "input_status.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(
        OUTPUT_DIR / "parameter_status.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "input_controls": int(len(inputs)),
        "manual_or_mixed_active_inputs": int(
            inputs["automation_status"].str.contains("manual_active").sum()
        ),
        "automatic_active_inputs": int(
            inputs["automation_status"].str.contains("automatic_active").sum()
        ),
        "automatic_candidate_inputs": int(
            inputs["automation_status"].eq("automatic_candidate").sum()
        ),
        "fixed_numeric_parameters": int(len(parameters)),
        "prior_only_automatic_parameter_candidates": int(
            parameters["automation_status"].eq("prior_only_automatic_candidate").sum()
        ),
        "safety_bounds": int(
            parameters["automation_status"].eq("safety_bound_document_and_keep").sum()
        ),
        "behavioral_parameters_pending": int(
            parameters["automation_status"].eq(
                "behavioral_parameter_pending_automation"
            ).sum()
        ),
        "active_model_changed": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
