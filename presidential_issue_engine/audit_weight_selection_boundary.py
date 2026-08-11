"""Audit that model selection and evaluation stop at the 2022 election."""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from presidential_issue_engine import issue_vote_engine as engine  # noqa: E402
from presidential_issue_engine import robustness_check  # noqa: E402


EXPECTED_SCORED = ("pres_2002", "pres_2007", "pres_2012", "pres_2017", "pres_2022")
EXPECTED_ROLLING_WARMUP = ("pres_1997",)
EXPECTED_SELECTION_STEPS = (
    "ridge_alpha",
    "residual",
    "neutral_context",
    "overlay",
    "conversion",
    "district_terrain",
    "regionalism",
    "third",
    "within_bloc_transfer",
)
FORBIDDEN_ELECTION = "pres_" + "2025"
DATA_ROOTS = (ROOT / "data", ROOT / "presidential_issue_engine" / "fixed_dataset")
REPORT_ROOTS = (
    ROOT / "presidential_issue_engine" / "report" / "tables",
    ROOT / "presidential_issue_engine" / "report" / "through2022_rederived",
)
REDERIVATION_SUMMARY = (
    ROOT / "presidential_issue_engine" / "report" / "through2022_rederived" / "summary.json"
)
ELECTORATE_LAYER_SUMMARY = ROOT / "outputs" / "electorate_layer_experiment" / "summary.json"
ELECTORATE_NESTED_LEARNING_SUMMARY = (
    ROOT / "outputs" / "electorate_nested_learning" / "summary.json"
)
ELECTORATE_FIXED_EXPERIMENT = (
    ROOT / "data" / "config" / "electorate_layers_fixed_experiment.json"
)
ACTIVE_POLICY = ROOT / "data" / "config" / "active_presidential_model.json"
DIRECT_MEGA_SCORES = ROOT / "outputs" / "active_presidential_nested" / "direct_mega_issue_scores.csv"
GOVERNMENT_BURDEN_SCORES = (
    ROOT / "outputs" / "active_presidential_nested" / "government_burden_scores.csv"
)
CONTEST_REGIMES = ROOT / "outputs" / "active_presidential_nested" / "contest_regimes.csv"
AUTO_SEED_OUTPUTS = (
    ROOT / "data" / "raw" / "auto_issue_seed" / "candidate_issue_profile.csv",
    ROOT / "data" / "raw" / "auto_issue_seed" / "mega_issue_axis.csv",
    ROOT / "data" / "raw" / "auto_issue_seed" / "mega_issue_attribution.csv",
)
SPARSE_AUTO_SEED_OUTPUTS = {"mega_issue_attribution.csv"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_engine_scope() -> None:
    require(tuple(engine.ORDER) == EXPECTED_SCORED, f"Unexpected engine.ORDER: {engine.ORDER}")
    require(
        tuple(engine.WEIGHT_SELECTION_ELECTIONS) == EXPECTED_SCORED,
        f"Unexpected weight-selection elections: {engine.WEIGHT_SELECTION_ELECTIONS}",
    )
    require(
        tuple(engine.ROLLING_WARMUP_ORDER) == EXPECTED_ROLLING_WARMUP,
        f"Unexpected rolling warmup: {engine.ROLLING_WARMUP_ORDER}",
    )
    require(
        tuple(robustness_check.COMPETITION_ELECTIONS) == EXPECTED_SCORED,
        f"Unexpected robustness scope: {robustness_check.COMPETITION_ELECTIONS}",
    )
    require(engine.THROUGH_2022_REDERIVED_CONFIG_PATH.exists(), "Missing rederived layer config")
    payload = json.loads(engine.THROUGH_2022_REDERIVED_CONFIG_PATH.read_text(encoding="utf-8"))
    require(
        payload.get("provenance") == "rederived only from rolling elections through 2022",
        "Invalid rederived layer-config provenance",
    )
    require(
        payload.get("config") == engine.THROUGH_2022_REDERIVED_LAYER_CONFIG,
        "Loaded layer config differs from the provenance file",
    )
    require(
        payload.get("registered_layers") == engine.THROUGH_2022_LAYER_REGISTRY,
        "Loaded layer registry differs from the provenance file",
    )
    registry = engine.THROUGH_2022_LAYER_REGISTRY
    require(ACTIVE_POLICY.exists(), "Missing active presidential policy")
    active_policy = json.loads(ACTIVE_POLICY.read_text(encoding="utf-8"))
    direct_mega = active_policy.get("postprocess", {})
    require(
        direct_mega.get("direct_mega_issue_shift") is True,
        "Active direct mega-issue shift is disabled",
    )
    require(
        float(direct_mega.get("direct_mega_logit_gain", -1.0)) == 0.40,
        "Active direct mega-issue gain drifted",
    )
    require(
        float(direct_mega.get("direct_mega_log_shift_cap", -1.0)) == 0.20,
        "Active direct mega-issue log-shift cap drifted",
    )
    require(
        float(direct_mega.get("direct_mega_minimum_intensity", -1.0)) == 1.0,
        "Active direct mega-issue intensity gate drifted",
    )
    require(
        direct_mega.get("incumbent_shock_response") is True,
        "Active incumbent-shock response is disabled",
    )
    require(
        float(direct_mega.get("government_burden_gain", -1.0)) == 1.0
        and float(direct_mega.get("rupture_extra_gain", -1.0)) == 0.40
        and float(direct_mega.get("incumbent_shock_log_shift_cap", -1.0)) == 0.15,
        "Active incumbent-shock response parameters drifted",
    )
    require(
        direct_mega.get("contest_regime_response") is True,
        "Active contest-regime response is disabled",
    )
    require(
        direct_mega.get("cumulative_regime_rejection") is True,
        "Active cumulative regime rejection is disabled",
    )
    require(
        float(direct_mega.get("contest_regime_expansion_gain", -1.0)) == 0.50
        and float(direct_mega.get("contest_regime_log_shift_cap", -1.0)) == 0.40
        and int(direct_mega.get("cumulative_rejection_breadth_reference", -1)) == 4
        and float(direct_mega.get("cumulative_rejection_party_erosion_width", -1.0)) == 0.08
        and float(direct_mega.get("cumulative_rejection_conversion_buffer", -1.0)) == 0.15
        and float(direct_mega.get("cumulative_rejection_rupture_score_reference", -1.0)) == 0.25,
        "Active contest-regime response parameters drifted",
    )
    require(DIRECT_MEGA_SCORES.exists(), "Missing active direct mega-issue score output")
    with DIRECT_MEGA_SCORES.open("r", encoding="utf-8-sig", newline="") as handle:
        direct_rows = list(csv.DictReader(handle))
    require(bool(direct_rows), "Active direct mega-issue score output is empty")
    require(
        {
            row.get("election_id", "").strip() for row in direct_rows
        }.issubset(set(EXPECTED_SCORED)),
        "Direct mega-issue score contains an out-of-scope election",
    )
    require(
        all(float(row["mega_issue_intensity"]) > 1.0 for row in direct_rows),
        "Direct mega-issue score bypassed the intensity gate",
    )
    require(
        all(abs(float(row["direct_mega_score"])) <= 0.50 for row in direct_rows),
        "Direct mega-issue score exceeds the active cap",
    )
    require(GOVERNMENT_BURDEN_SCORES.exists(), "Missing government-burden score output")
    with GOVERNMENT_BURDEN_SCORES.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        government_rows = list(csv.DictReader(handle))
    require(bool(government_rows), "Government-burden score output is empty")
    require(
        {row.get("election_id", "").strip() for row in government_rows}.issubset(
            set(EXPECTED_SCORED)
        ),
        "Government-burden score contains an out-of-scope election",
    )
    require(CONTEST_REGIMES.exists(), "Missing active contest-regime output")
    with CONTEST_REGIMES.open("r", encoding="utf-8-sig", newline="") as handle:
        regime_rows = list(csv.DictReader(handle))
    require(len(regime_rows) == len(EXPECTED_SCORED), "Unexpected contest-regime row count")
    require(
        {row.get("election_id", "").strip() for row in regime_rows}
        == set(EXPECTED_SCORED),
        "Contest-regime output scope differs from scored elections",
    )
    require(
        set(registry)
        == {
            "issue_character_overlay",
            "third_candidate_competitiveness_gate",
            "third_candidate_character_multiplier",
            "candidate_vote_conversion_context",
            "district_terrain",
            "within_bloc_regional_transfer",
            "manual_issue_seed",
            "automatic_issue_seed",
        },
        "Unexpected registered layer set",
    )
    manual_seed = registry["manual_issue_seed"]
    require(manual_seed.get("enabled") is False, "Manual issue seed must be disabled")
    require(
        tuple(manual_seed.get("elections", [])) == EXPECTED_SCORED,
        "Manual issue seed scope differs from the scored elections",
    )
    require(REDERIVATION_SUMMARY.exists(), "Missing rederivation summary")
    summary_text = REDERIVATION_SUMMARY.read_text(encoding="utf-8")
    require(FORBIDDEN_ELECTION not in summary_text, "Rederivation summary contains forbidden data")
    summary = json.loads(summary_text)
    require(
        tuple(summary["policy"]["allowed_elections"]) == EXPECTED_SCORED,
        "Rederivation used an unexpected election",
    )
    require(
        tuple(summary["policy"]["selection_steps"]) == EXPECTED_SELECTION_STEPS,
        "Rederivation used an unexpected selection protocol",
    )
    selected_config = summary["final_deployment_selection"]["config"]
    active_config = engine.THROUGH_2022_REDERIVED_LAYER_CONFIG
    differing_keys = {
        key
        for key in set(selected_config) | set(active_config)
        if selected_config.get(key) != active_config.get(key)
    }
    require(
        differing_keys <= {"overlay_gain"},
        f"Unexpected override of nested-selected config: {sorted(differing_keys)}",
    )
    if differing_keys:
        overlay_registry = registry["issue_character_overlay"]
        require(
            float(selected_config.get("overlay_gain", 0.0)) == 0.0,
            "Nested selection did not preserve the zero overlay baseline",
        )
        require(
            float(active_config.get("overlay_gain", 0.0)) == 0.04,
            "Explanatory overlay override must remain at the declared 0.04 cap",
        )
        require(
            overlay_registry.get("enabled") is True
            and overlay_registry.get("source_model") == "stance_nli_ambiguity_v14"
            and overlay_registry.get("direct_candidate_vote_adjustment") is False,
            "Explanatory overlay override lacks bounded provenance",
        )
        require(
            "not selected as a predictive MAE improvement"
            in str(overlay_registry.get("activation_policy", "")),
            "Explanatory overlay is being misrepresented as nested selection",
        )
    require(
        engine.RIDGE_ALPHA == float(engine.THROUGH_2022_REDERIVED_LAYER_CONFIG["ridge_alpha"]),
        "Active Ridge alpha differs from the provenance config",
    )
    require(
        engine.neutral_issue_context_scale()
        == float(engine.THROUGH_2022_REDERIVED_LAYER_CONFIG["neutral_context_scale"]),
        "Active neutral-context scale differs from the provenance config",
    )
    require(engine.ELECTORATE_LAYER_CONFIG_PATH.exists(), "Missing electorate-layer config")
    electorate_payload = json.loads(
        engine.ELECTORATE_LAYER_CONFIG_PATH.read_text(encoding="utf-8")
    )
    require(electorate_payload.get("enabled") is True, "Electorate layer must be enabled")
    require(engine.ELECTORATE_LAYER_ENABLED is True, "Loaded electorate layer is disabled")
    require(
        electorate_payload.get("config")
        == {
            "terrain_anchor_gain": engine.ELECTORATE_LAYER_CONFIG.terrain_anchor_gain,
            "camp_core_anchor_gain": engine.ELECTORATE_LAYER_CONFIG.camp_core_anchor_gain,
            "camp_regional_lean_gain": engine.ELECTORATE_LAYER_CONFIG.camp_regional_lean_gain,
            "camp_composition_gain": engine.ELECTORATE_LAYER_CONFIG.camp_composition_gain,
            "preference_gain": engine.ELECTORATE_LAYER_CONFIG.preference_gain,
            "layer_separation": engine.ELECTORATE_LAYER_CONFIG.layer_separation,
            "layer_response_profile": engine.ELECTORATE_LAYER_CONFIG.layer_response_profile,
            "mass_profile": engine.ELECTORATE_LAYER_CONFIG.mass_profile,
            "turnout_gain": engine.ELECTORATE_LAYER_CONFIG.turnout_gain,
            "nonvoter_gain": engine.ELECTORATE_LAYER_CONFIG.nonvoter_gain,
        },
        "Loaded electorate-layer config differs from provenance file",
    )
    constraints = electorate_payload.get("constraints", {})
    require(
        constraints.get("post_2022_presidential_outcomes_used") is False,
        "Electorate layer declares post-2022 outcome use",
    )
    require(
        constraints.get("turnout_channel_active") is False
        and constraints.get("nonvoter_channel_active") is False,
        "Turnout layers cannot be active without official prior turnout history",
    )
    require(ELECTORATE_LAYER_SUMMARY.exists(), "Missing electorate-layer selection summary")
    electorate_summary_text = ELECTORATE_LAYER_SUMMARY.read_text(encoding="utf-8")
    require(
        FORBIDDEN_ELECTION not in electorate_summary_text,
        "Electorate-layer selection summary contains forbidden data",
    )
    electorate_summary = json.loads(electorate_summary_text)
    require(
        tuple(electorate_summary["scope"]["scored_elections"]) == EXPECTED_SCORED,
        "Electorate-layer selection used an unexpected election",
    )
    require(
        ELECTORATE_NESTED_LEARNING_SUMMARY.exists(),
        "Missing capped nested electorate-learning summary",
    )
    nested_learning_text = ELECTORATE_NESTED_LEARNING_SUMMARY.read_text(encoding="utf-8")
    require(
        FORBIDDEN_ELECTION not in nested_learning_text,
        "Nested electorate-learning summary contains forbidden data",
    )
    nested_learning = json.loads(nested_learning_text)
    require(
        nested_learning["scope"].get("run_mode") == "capped_candidate",
        "Active electorate summary was overwritten by a non-capped run",
    )
    require(
        tuple(nested_learning["scope"]["scored_elections"]) == EXPECTED_SCORED,
        "Nested electorate learner used an unexpected election",
    )
    nested_deployment = nested_learning["future_deployment_through2022"]
    require(
        electorate_payload.get("config") == nested_deployment,
        "Active electorate config differs from the strict nested selection result",
    )
    require(
        nested_learning["adopt_into_active_engine"] is True,
        "Electorate active-adoption flag differs from strict nested selection",
    )
    require(
        all(nested_learning["adoption_gates"].values()),
        "Active electorate learner has a failed adoption gate",
    )
    require(ELECTORATE_FIXED_EXPERIMENT.exists(), "Missing fixed electorate experiment config")
    fixed_experiment = json.loads(ELECTORATE_FIXED_EXPERIMENT.read_text(encoding="utf-8"))
    require(
        fixed_experiment.get("constraints", {}).get("not_active") is True
        and fixed_experiment.get("evaluation", {}).get("strict_nested_selection_result") is False,
        "Fixed electorate experiment is not clearly separated from the active nested config",
    )
    automatic_seed = registry["automatic_issue_seed"]
    require(automatic_seed.get("enabled") is True, "Automatic issue seed is disabled")
    require(
        tuple(automatic_seed.get("elections", [])) == EXPECTED_SCORED,
        "Automatic issue seed scope differs from the scored elections",
    )
    require(
        automatic_seed.get("outcome_fields_used") == [],
        "Automatic issue seed declares outcome-field use",
    )
    for path in AUTO_SEED_OUTPUTS:
        require(path.exists(), f"Missing registered automatic seed output: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            election_ids = {
                row.get("election_id", "").strip()
                for row in rows
                if row.get("election_id", "").strip()
            }
        if path.name in SPARSE_AUTO_SEED_OUTPUTS:
            require(
                election_ids.issubset(set(EXPECTED_SCORED)),
                f"Sparse automatic seed output contains an unexpected election: {path}",
            )
        else:
            require(
                election_ids == set(EXPECTED_SCORED),
                f"Automatic seed output has unexpected election coverage: {path}",
            )
        for row in rows:
            election_id = row.get("election_id", "").strip()
            available = row.get("available_date", "").strip()
            require(bool(available), f"Automatic seed row lacks available_date: {path}")
            cutoff = date.fromisoformat(engine.ELECTION_DATES[election_id]) - timedelta(days=1)
            require(
                date.fromisoformat(available) <= cutoff,
                f"Automatic seed row exceeds election cutoff: {path}",
            )


def audit_active_csv_rows() -> int:
    checked = 0
    for base in DATA_ROOTS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.reader(handle):
                    checked += 1
                    require(FORBIDDEN_ELECTION not in row, f"Forbidden election row: {path}")
    return checked


def audit_report_scope() -> int:
    checked = 0
    allowed_context_ids = {*engine.WARMUP_ORDER, *EXPECTED_SCORED, "Overall"}
    scored_report_prefixes = ("competition_rolling_", "issue_vote_engine_rolling_")
    for report_root in REPORT_ROOTS:
        if not report_root.exists():
            continue
        for path in sorted(report_root.rglob("*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    checked += 1
                    election_id = (row.get("election_id") or "").strip()
                    if election_id:
                        require(
                            election_id in allowed_context_ids,
                            f"Report includes out-of-scope election: {path}",
                        )
                        if path.name.startswith(scored_report_prefixes):
                            require(
                                election_id in {*EXPECTED_SCORED, "Overall"},
                                f"Scored report includes non-scored election: {path}",
                            )
                    require(
                        FORBIDDEN_ELECTION not in " ".join(row.values()),
                        f"Report references forbidden comparison: {path}",
                    )
    return checked


def main() -> None:
    audit_engine_scope()
    data_rows = audit_active_csv_rows()
    report_rows = audit_report_scope()
    print("[through-2022 weight-selection audit: PASS]")
    print(f"scored_elections={','.join(EXPECTED_SCORED)}")
    print(f"rolling_warmup={','.join(EXPECTED_ROLLING_WARMUP)}")
    print(f"active_csv_rows_checked={data_rows}")
    print(f"report_rows_checked={report_rows}")


if __name__ == "__main__":
    main()
