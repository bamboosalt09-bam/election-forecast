from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from election_forecast.features.issue_matcher import IssueContextRule
from scripts import run_prospective_forecast as prospective

# Building the 2025 target context reads bulk Assembly material that is kept
# outside Git, so these tests exercise nothing on a fresh public checkout.
BULK_TARGET_SOURCES = (
    prospective.ROOT
    / "data/raw/official_sources/assembly_pres_2025_minutes/assembly_stance_rows_2025_h1.csv",
    prospective.ROOT
    / "archives/experiments/manual_seed_lineage_v17_rejected_20260728/artifacts/assembly_speaker_issue_matches_15_22.csv",
)


def _require_bulk_target_sources() -> None:
    missing = [path for path in BULK_TARGET_SOURCES if not path.exists()]
    if missing:
        pytest.skip(
            "bulk 2025 Assembly sources are not part of the public repository: "
            + ", ".join(path.name for path in missing)
        )


def test_v25_target_mega_controls_are_pit_automatic_and_preserve_history(
    tmp_path,
) -> None:
    _require_bulk_target_sources()
    paths, diagnostics = prospective._automatic_target_mega_controls(tmp_path)
    intensity = pd.read_csv(paths["mega_issue_intensity"], encoding="utf-8-sig")
    taxonomy = pd.read_csv(paths["mega_issue_taxonomy"], encoding="utf-8-sig")
    audit = pd.read_csv(paths["mega_issue_taxonomy_audit"], encoding="utf-8-sig")
    historical_intensity = pd.read_csv(
        prospective.AUTOMATIC_DIR / "mega_issue_intensity.csv",
        encoding="utf-8-sig",
    )
    historical_taxonomy = pd.read_csv(
        prospective.AUTOMATIC_DIR / "mega_issue_taxonomy.csv",
        encoding="utf-8-sig",
    )

    target_intensity = intensity.loc[
        intensity["election_id"].astype(str).eq(prospective.TARGET_ELECTION)
    ]
    target_taxonomy = taxonomy.loc[
        taxonomy["election_id"].astype(str).eq(prospective.TARGET_ELECTION)
    ]
    assert len(target_intensity) == len(target_taxonomy) == 1
    assert target_intensity.iloc[0]["mega_issue_intensity"] == pytest.approx(2.0)
    assert target_taxonomy.iloc[0]["shock_type"] == "institutional_crisis"
    assert pd.Timestamp(target_intensity.iloc[0]["available_date"]) <= pd.Timestamp(
        prospective.FORECAST_CUTOFF
    )
    assert diagnostics["semantic_gate"] is True
    # The gate fires, but with pres_2025 crisis vocabulary registered the
    # frequency path reaches the same class on its own, so the gate decides
    # nothing and must not report an adjustment it did not make.
    assert diagnostics["semantic_gate_adjustment_applied"] is False
    # pres_2025 crisis vocabulary is registered in mega_issue_terms.csv, so the
    # frequency-only classification reaches the crisis class on term evidence
    # alone and the official-proceeding gate no longer decides it.
    assert diagnostics["frequency_shock_type"] == diagnostics["shock_type"]
    assert diagnostics["frequency_shock_type"] == "institutional_crisis"
    assert diagnostics["frequency_mega_issue_intensity"] == pytest.approx(
        diagnostics["mega_issue_intensity"]
    )
    assert diagnostics["historical_compatible_match_rows"] > 0
    assert diagnostics["target_outcomes_used"] is False
    assert audit.iloc[0]["semantic_institutional_gate"]
    assert not audit.iloc[0]["semantic_gate_adjustment_applied"]
    assert audit.iloc[0]["frequency_automatic_shock_type"] == (
        audit.iloc[0]["selected_shock_type"]
    )
    assert audit.iloc[0]["selected_shock_type"] == "institutional_crisis"
    assert audit.iloc[0]["selected_mega_issue_intensity"] == pytest.approx(2.0)
    assert audit.iloc[0]["target_match_granularity"] == (
        "reconstructed_historical_speech_row"
    )

    pd.testing.assert_frame_equal(
        intensity.loc[
            ~intensity["election_id"].astype(str).eq(prospective.TARGET_ELECTION)
        ].reset_index(drop=True),
        historical_intensity.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        taxonomy.loc[
            ~taxonomy["election_id"].astype(str).eq(prospective.TARGET_ELECTION)
        ].reset_index(drop=True),
        historical_taxonomy.reset_index(drop=True),
        check_dtype=False,
    )


def test_target_mega_match_reconstructs_cross_sentence_context(
    monkeypatch,
) -> None:
    target = pd.DataFrame(
        {
            "source_id": ["meeting", "meeting", "meeting"],
            "source_row_id": ["row", "row", "row"],
            "sentence_index": [1, 1, 2],
            "period": ["2025-05-01"] * 3,
            "speaker": ["speaker"] * 3,
            "text_excerpt": [
                "housing pressure",
                "housing pressure",
                "martial law responsibility",
            ],
        }
    )
    monkeypatch.setattr(
        prospective.assembly_match_builder,
        "build_keyword_inputs",
        lambda: (
            {prospective.TARGET_ELECTION: {"housing": ["housing"]}},
            {prospective.TARGET_ELECTION: {}},
            {prospective.TARGET_ELECTION: {}},
            {
                prospective.TARGET_ELECTION: [
                    IssueContextRule(
                        source_issue="housing",
                        context_terms=("martial law",),
                        target_issue="regime_change",
                        target_weight=0.5,
                    )
                ]
            },
        ),
    )

    matches, diagnostics = prospective._historical_compatible_target_matches(target)

    weights = matches.set_index("issue_name")["issue_weight"].to_dict()
    assert weights == {"housing": 0.35, "regime_change": 0.175}
    assert diagnostics == {
        "sentence_issue_rows": 3,
        "unique_sentence_rows": 2,
        "reconstructed_speech_rows": 1,
        "historical_compatible_match_rows": 2,
    }


def test_v25_target_mega_controls_reject_outcome_columns(monkeypatch, tmp_path) -> None:
    source = tmp_path / "forbidden.csv"
    pd.DataFrame({"vote_share": [0.5]}).to_csv(source, index=False)
    monkeypatch.setattr(prospective, "OFFICIAL_2025_MINUTES", source)

    with pytest.raises(RuntimeError, match="outcome columns"):
        prospective._automatic_target_mega_controls(tmp_path)


def test_target_candidate_profiles_separate_government_from_direct_strength(
    tmp_path,
) -> None:
    _require_bulk_target_sources()
    cutoff = pd.Timestamp(prospective.FORECAST_CUTOFF)
    registry = prospective._validate_registry(
        pd.read_csv(prospective.REGISTRY, encoding="utf-8-sig"), cutoff
    )
    _, candidate_link, _ = prospective.load_forecast_only_assembly_inputs(
        prospective.TARGET_ELECTION,
        prospective.CONTEXT_DIR,
    )
    selected = prospective._select_model_candidates(
        registry,
        candidate_link,
        prospective.active.nested.engine,
    )
    _, paths, diagnostics = prospective._build_target_candidate_context(
        tmp_path,
        registry,
        selected,
        candidate_link,
        version="v25",
    )
    burden = pd.read_csv(paths["candidate_issue_profile"], encoding="utf-8-sig")
    direct = pd.read_csv(
        paths["candidate_direct_issue_profile"], encoding="utf-8-sig"
    )
    burden_sources = burden["target_source_types"].fillna("").astype(str)
    direct_sources = direct["target_source_types"].fillna("").astype(str)

    assert burden_sources.str.contains(r"(?:^|\|)government(?:\||$)", regex=True).any()
    assert not direct_sources.str.contains(
        r"(?:^|\|)government(?:\||$)", regex=True
    ).any()
    assert diagnostics["government_burden_profile_government_rows"] > 0
    assert diagnostics["candidate_direct_profile_government_rows"] == 0


def test_target_base_preserves_merged_candidate_names(monkeypatch) -> None:
    target = pd.DataFrame(
        {
            "election_id": ["pres_2025"],
            "region_id": ["sido_11"],
            "slot": ["A"],
            "candidate_name": ["candidate_a"],
        }
    )
    historical = pd.DataFrame(
        columns=[
            "election_id",
            "region_id",
            "slot",
            "candidate_name_x",
            "candidate_name_y",
            "contest_votes",
            "actual",
        ]
    )
    monkeypatch.setattr(
        prospective,
        "_prior_region_volume",
        lambda version="v23": pd.Series({"sido_11": 100.0}),
    )

    result = prospective._target_base(target, historical)

    assert result.loc[0, "candidate_name_x"] == "candidate_a"
    assert result.loc[0, "candidate_name_y"] == "candidate_a"
    assert np.isnan(result.loc[0, "actual"])


def test_committed_prospective_output_is_forecast_only() -> None:
    output = prospective.ROOT / "outputs/prospective_pres_2025_v23"
    predictions = pd.read_csv(
        output / "prospective_predictions.csv", encoding="utf-8-sig"
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    inputs = set(
        pd.read_csv(output / "input_manifest.csv", encoding="utf-8-sig")["path"]
        .astype(str)
        .tolist()
    )

    assert predictions.columns.tolist() == list(prospective.OUTPUT_COLUMNS)
    assert len(predictions) == 51
    assert predictions["candidate_name"].astype(str).ne("0.0").all()
    assert np.allclose(
        predictions.groupby(["election_id", "region_id"])["predicted_share"].sum(),
        1.0,
    )
    assert manifest["forecast_cutoff"] == "2025-06-02"
    assert manifest["training_latest_election"] == "pres_2022"
    assert manifest["outcome_columns_used"] == []
    assert manifest["performance_metrics_computed"] is False
    assert manifest["pres_2025_outcome_present"] is False
    assert "candidate_context_lineage_manifest_sha256" in manifest
    assert "assembly22_roster_manifest_sha256" in manifest
    assert {
        "data/raw/official_sources/assembly_pres_2025_context/candidate_context_v2/candidate_political_landscape.csv",
        "data/raw/official_sources/assembly_pres_2025_context/candidate_context_v2/auto_candidate_role/third_candidate_profile.csv",
        "data/raw/official_sources/assembly_pres_2025_context/candidate_context_v2/auto_issue_seed/candidate_issue_profile.csv",
        "data/raw/official_sources/assembly_pres_2025_context/candidate_context_v2/auto_issue_seed/mega_issue_axis.csv",
        "data/raw/official_sources/assembly_pres_2025_context/candidate_context_v2/auto_issue_seed/mega_issue_attribution.csv",
    }.issubset(inputs)
    assert not any("automatic_third_candidate_pressure.csv" in path for path in inputs)
    assert not any("model_mega_issue_" in path for path in inputs)


def test_v24_prospective_output_runs_every_promoted_extension() -> None:
    output = prospective.ROOT / "outputs/prospective_pres_2025_v24"
    predictions = pd.read_csv(
        output / "prospective_predictions.csv", encoding="utf-8-sig"
    )
    stages = pd.read_csv(
        output / "prediction_stage_audit.csv",
        encoding="utf-8-sig",
        low_memory=False,
    )
    features = pd.read_csv(
        output / "target_feature_audit.csv",
        encoding="utf-8-sig",
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    inputs = set(
        pd.read_csv(output / "input_manifest.csv", encoding="utf-8-sig")["path"]
        .astype(str)
        .tolist()
    )

    assert len(predictions) == 51
    assert len(stages) == 51
    assert len(features) == 51
    assert np.allclose(
        predictions.groupby(["election_id", "region_id"])["predicted_share"].sum(),
        1.0,
    )
    assert manifest["version"] == "v24"
    assert manifest["forecast_cutoff"] == "2025-06-02"
    assert manifest["performance_metrics_computed"] is False
    assert manifest["model_selection_performed"] is False
    assert manifest["model_parameters_changed"] is False
    assert manifest["frozen_historical_reproduction"]["required"] is True
    assert manifest["frozen_historical_reproduction"]["passed"] is True
    assert manifest["frozen_historical_reproduction"]["rows"] == 232
    assert (
        manifest["frozen_historical_reproduction"][
            "maximum_absolute_difference"
        ]
        <= 1e-12
    )
    assert manifest["target_feature_audit_rows"] == 51
    assert set(prospective.active.nested.BASE_PREDICTORS).issubset(features.columns)
    assert features[list(prospective.active.nested.BASE_PREDICTORS)].notna().all().all()
    assert features["candidate_weight"].notna().all()
    assert features["assigned_slot"].notna().all()
    assert (
        manifest["government_context_link"]["government_evidence_destination"]
        == "issue_character_burden_only"
    )
    assert manifest["government_context_link"]["candidate_attention_source_types"] == [
        "person",
        "party",
    ]
    assert manifest["v24_postprocess_order"] == list(
        prospective.V24_POSTPROCESS_ORDER
    )
    assert manifest["v24_postprocess_audit_rows"] == {
        "strong_incumbent_veto": 0,
        "third_candidate_lineage_ceiling": 17,
        "weak_same_lane_refusal": 17,
    }
    government_fields = stages[
        [
            "government_evidence_count",
            "government_evidence_weight",
            "government_rejection_strength",
        ]
    ].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    assert government_fields.to_numpy(float).any()
    assert manifest["government_context_link"]["prior_election"] == "pres_2022"
    assert manifest["government_context_link"]["current_assembly"] == 22
    assert manifest["government_context_link"]["target_outcomes_used"] is False
    assert manifest["government_context_link"]["directional_aggregate_rows"] > 0
    assert {
        "v24_pre_extension_pred",
        "v24_post_strong_veto_pred",
        "v24_post_lineage_ceiling_pred",
        "v24_post_weak_lane_refusal_pred",
    }.issubset(stages.columns)
    assert {
        "presidential_issue_engine/fixed_dataset/v24/third_candidate_lineage.csv",
        "presidential_issue_engine/strong_incumbent_veto.py",
        "presidential_issue_engine/third_candidate_lineage_constraint.py",
        "presidential_issue_engine/weak_same_lane_refusal.py",
        "data/raw/official_sources/assembly_pres_2025_context/explicit_target_context_weekly.csv",
        "data/raw/official_sources/assembly_pres_2025_context/candidate_target_context_weekly.csv",
    }.issubset(inputs)


def test_v24_uses_the_promoted_wrapper_and_versioned_history() -> None:
    assert prospective._config_path("v24") == prospective.active_v24.CONFIG_PATH
    assert prospective._historical_results_path("v24") == (
        prospective.active_v24.V24_DATA / "presidential_results_standardized.csv"
    )
    assert prospective.V24_POSTPROCESS_ORDER == (
        "strong_incumbent_veto",
        "third_candidate_lineage_ceiling",
        "weak_same_lane_refusal",
    )


def test_v25_prospective_output_reproduces_bounded_historical_runtime() -> None:
    output = prospective.ROOT / "outputs/prospective_pres_2025_v25"
    predictions = pd.read_csv(
        output / "prospective_predictions.csv", encoding="utf-8-sig"
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    weak_audit = pd.read_csv(
        output / "weak_same_lane_refusal_audit.csv", encoding="utf-8-sig"
    )

    assert len(predictions) == 51
    assert np.allclose(
        predictions.groupby(["election_id", "region_id"])["predicted_share"].sum(),
        1.0,
    )
    assert manifest["version"] == "v25"
    assert manifest["runtime_policy_matches_declared_config"] is True
    assert manifest["outcome_columns_used"] == []
    assert manifest["performance_metrics_computed"] is False
    assert manifest["pres_2025_outcome_present"] is False
    assert manifest["model_selection_performed"] is False
    assert manifest["model_parameters_changed"] is False
    assert (
        manifest["government_context_link"][
            "candidate_direct_profile_government_rows"
        ]
        == 0
    )
    assert (
        manifest["government_context_link"][
            "government_burden_profile_government_rows"
        ]
        > 0
    )
    assert manifest["frozen_historical_reproduction"]["passed"] is True
    assert manifest["frozen_historical_reproduction"]["rows"] == 232
    assert (
        manifest["frozen_historical_reproduction"]["maximum_absolute_difference"]
        <= 1e-12
    )
    assert set(weak_audit["recipient_weight_mode"].astype(str)) == {
        "prediction_tilted"
    }
    assert manifest["mega_issue_controls"]["semantic_gate"] is True
    # the gate fires but changes nothing once the frequency path reaches the
    # same class on registered vocabulary alone
    assert manifest["mega_issue_controls"]["semantic_gate_adjustment_applied"] is False
    assert (
        manifest["mega_issue_controls"]["shock_type"]
        == "institutional_crisis"
    )
    assert manifest["mega_issue_controls"]["mega_issue_intensity"] == pytest.approx(2.0)
    # The registered pres_2025 crisis vocabulary lets the frequency-only path
    # reach the same class and intensity as the gated path, so the gate is a
    # no-op for this target rather than the source of the classification.
    controls = manifest["mega_issue_controls"]
    assert controls["frequency_shock_type"] == controls["shock_type"]
    assert controls["frequency_mega_issue_intensity"] == pytest.approx(
        controls["mega_issue_intensity"]
    )
    assert manifest["mega_issue_controls"]["target_outcomes_used"] is False
    assert manifest["v24_postprocess_audit_rows"] == {
        "strong_incumbent_veto": 17,
        "third_candidate_lineage_ceiling": 17,
        "weak_same_lane_refusal": 17,
    }


def test_v25_prospective_runtime_keeps_rejected_third_candidate_rebind_off() -> None:
    assert "third_candidate_inputs" not in prospective.active_v25.RUNTIME_REPAIRS
    assert prospective._config_path("v25") == prospective.active_v25.CONFIG_PATH
    assert prospective._historical_results_path("v25") == (
        prospective.active_v24.V24_DATA / "presidential_results_standardized.csv"
    )
    assert prospective._runtime_policy_path(
        "v25", prospective.active_v25.CONFIG_PATH
    ) == prospective.active_v25.CONFIG_PATH


def test_stage_audit_removes_outcome_shaped_columns() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2025"],
            "pred": [0.4],
            "layer_pred": [0.5],
            "actual": [np.nan],
            "votes": [0.0],
            "row_mae": [np.nan],
        }
    )

    result = prospective._safe_stage_audit(frame)

    assert "base_stage_prediction" in result.columns
    assert "layer_pred" in result.columns
    assert "actual" not in result.columns
    assert "votes" not in result.columns
    assert "row_mae" not in result.columns


def test_candidate_strength_prefers_direct_speech_context(monkeypatch, tmp_path) -> None:
    context = tmp_path / "candidate_vote_conversion_context.csv"
    rows = []
    for election_id, name, slot, weight in [
        ("pres_2022", "old", "A", 0.4),
        ("pres_2025", "candidate_a", "A", 0.7),
        ("pres_2025", "candidate_b", "B", 0.6),
        ("pres_2025", "candidate_c", "C", 0.3),
    ]:
        rows.append(
            {
                "election_id": election_id,
                "slot": slot,
                "candidate_name": name,
                "candidate_weight": weight,
                "confidence": 0.5,
                "available_date": "2025-06-02" if election_id == "pres_2025" else "2022-03-08",
            }
        )
    pd.DataFrame(rows).to_csv(context, index=False)
    monkeypatch.setattr(prospective, "CANDIDATE_CONVERSION_HISTORY", context)
    selected = pd.DataFrame(
        {
            "candidate_id": ["id_b", "id_a", "id_c"],
            "candidate_name": ["candidate_b", "candidate_a", "candidate_c"],
            "slot": ["A", "B", "C"],
        }
    )

    combined, diagnostics = prospective._candidate_strength_context(
        selected,
        pd.DataFrame(),
    )

    assert diagnostics["method"] == "direct_speech_derived_candidate_context"
    target = combined.loc[combined["election_id"].eq("pres_2025")]
    assert target.set_index("candidate_name")["slot"].to_dict() == {
        "candidate_b": "A",
        "candidate_a": "B",
        "candidate_c": "C",
    }
    assert diagnostics["target_outcomes_used"] is False
