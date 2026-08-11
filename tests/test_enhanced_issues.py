from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from election_forecast.config import DEFAULT_CONFIG, ModelWeights
from election_forecast.enhanced_issue_audit import audit_enhanced_issue_inputs
from election_forecast.enhanced_issues import compile_enhanced_issue_scores
from election_forecast.issue_score import compute_issue_impact
from election_forecast.load_data import load_raw_data
from election_forecast.schemas import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, validate_optional_columns
from election_forecast.utility import compute_utility_frame, prepare_forecast_inputs


def _copy_required_raw(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    source = Path("data/raw")
    for name in REQUIRED_COLUMNS:
        shutil.copy(source / f"{name}.csv", raw_dir / f"{name}.csv")
    return data_dir


def _inject_sample_candidates(raw: dict[str, pd.DataFrame]) -> None:
    raw["candidates"] = pd.DataFrame(
        [
            {
                "election_id": "dummy_presidential_2026",
                "candidate_id": "sample_cand_a",
                "candidate_name": "Sample Candidate A",
                "party_name": "Sample Party A",
                "official_camp": "conservative",
                "political_weight_score": 0.70,
                "administrative_experience_score": 0.65,
                "favorability_score": 0.52,
                "unfavorability_score": 0.42,
                "risk_score": 0.25,
                "expansion_score": 0.45,
                "available_date": pd.Timestamp("2026-01-01"),
            },
            {
                "election_id": "dummy_presidential_2026",
                "candidate_id": "sample_cand_b",
                "candidate_name": "Sample Candidate B",
                "party_name": "Sample Party B",
                "official_camp": "liberal",
                "political_weight_score": 0.68,
                "administrative_experience_score": 0.70,
                "favorability_score": 0.50,
                "unfavorability_score": 0.40,
                "risk_score": 0.22,
                "expansion_score": 0.48,
                "available_date": pd.Timestamp("2026-01-01"),
            },
            {
                "election_id": "dummy_presidential_2026",
                "candidate_id": "sample_cand_c",
                "candidate_name": "Sample Candidate C",
                "party_name": "Sample Party C",
                "official_camp": "centrist",
                "political_weight_score": 0.35,
                "administrative_experience_score": 0.40,
                "favorability_score": 0.30,
                "unfavorability_score": 0.28,
                "risk_score": 0.18,
                "expansion_score": 0.55,
                "available_date": pd.Timestamp("2026-01-01"),
            },
        ]
    )


def test_load_raw_data_adds_empty_optional_frames_when_files_are_absent(tmp_path: Path) -> None:
    data_dir = _copy_required_raw(tmp_path)

    data = load_raw_data(data_dir)

    for name, columns in OPTIONAL_COLUMNS.items():
        assert name in data
        assert data[name].empty
        assert list(data[name].columns) == columns


def test_load_raw_data_parses_optional_dates_and_numbers() -> None:
    data = load_raw_data("data")

    assert pd.api.types.is_datetime64_any_dtype(data["candidate_issue_profile"]["available_date"])
    assert pd.api.types.is_datetime64_any_dtype(data["mega_issue_axis"]["available_date"])
    assert pd.api.types.is_datetime64_any_dtype(data["mega_issue_attribution"]["available_date"])
    assert pd.to_numeric(data["issue_scope_weights"]["national_weight"], errors="coerce").notna().all()


def test_optional_validation_rejects_out_of_range_values() -> None:
    bad = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "candidate_id": "cand_a",
                "slot": "A",
                "candidate_name": "Candidate A",
                "issue_name": "housing",
                "association_strength": 1.5,
                "direction": 1,
                "available_date": "2026-01-01",
                "source_type": "manual",
                "confidence": 0.5,
                "notes": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="association_strength"):
        validate_optional_columns("candidate_issue_profile", bad)


def test_audit_current_seed_has_no_findings_for_historical_target() -> None:
    raw = load_raw_data("data")

    findings = audit_enhanced_issue_inputs(raw, "2017-05-08", target_election_id="pres_2017")

    assert findings.empty


def test_prepare_forecast_inputs_filters_target_election_candidates() -> None:
    raw = load_raw_data("data")
    _inject_sample_candidates(raw)
    other = raw["candidates"].copy()
    other["election_id"] = "other_election"
    other["candidate_id"] = other["candidate_id"].astype(str) + "_other"
    raw["candidates"] = pd.concat([raw["candidates"], other], ignore_index=True)

    data = prepare_forecast_inputs(raw, "2026-05-01", target_election_id="dummy_presidential_2026")

    assert set(data["candidates"]["candidate_id"]) == {"sample_cand_a", "sample_cand_b", "sample_cand_c"}


def test_prepare_forecast_inputs_keeps_bloc_history_for_target_forecast() -> None:
    raw = load_raw_data("data")

    data = prepare_forecast_inputs(raw, "2026-05-01", target_election_id="dummy_presidential_2026")

    assert not data["bloc_history_results"].empty
    assert "dummy_presidential_2026" not in set(data["bloc_history_results"]["election_id"].astype(str))


def test_candidate_profile_compiles_to_issue_scores() -> None:
    data = {
        "candidates": pd.DataFrame(
            [
                {
                    "candidate_id": "cand_a",
                    "candidate_name": "Candidate A",
                    "party_name": "Party A",
                    "official_camp": "camp_a",
                }
            ]
        ),
        "candidate_issue_profile": pd.DataFrame(
            [
                {
                    "election_id": "pres_x",
                    "candidate_id": "cand_a",
                    "slot": "A",
                    "candidate_name": "Candidate A",
                    "issue_name": "housing",
                    "association_strength": 0.8,
                    "direction": -1,
                    "available_date": "2026-01-01",
                    "source_type": "manual",
                    "confidence": 0.5,
                    "notes": "test",
                }
            ]
        ),
        "mega_issue_axis": pd.DataFrame(),
        "mega_issue_attribution": pd.DataFrame(),
    }

    out = compile_enhanced_issue_scores(data, "2026-02-01")

    assert len(out) == 1
    assert out.loc[0, "candidate_id"] == "cand_a"
    assert out.loc[0, "final_issue_score"] == pytest.approx(-0.4)


def test_mega_attribution_resolves_supported_target_types_and_filters_future_rows() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": "cand_a", "candidate_name": "Candidate A", "party_name": "Party A", "official_camp": "camp_a"},
            {"candidate_id": "cand_b", "candidate_name": "Candidate B", "party_name": "Party B", "official_camp": "camp_b"},
            {"candidate_id": "cand_c", "candidate_name": "Candidate C", "party_name": "Party A", "official_camp": "camp_c"},
        ]
    )
    profiles = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "candidate_id": "cand_a",
                "slot": "A",
                "candidate_name": "Candidate A",
                "issue_name": "regime_change",
                "association_strength": 0.1,
                "direction": 1,
                "available_date": "2026-01-01",
                "source_type": "manual",
                "confidence": 0.5,
                "notes": "",
            },
            {
                "election_id": "pres_x",
                "candidate_id": "cand_b",
                "slot": "B",
                "candidate_name": "Candidate B",
                "issue_name": "regime_change",
                "association_strength": 0.1,
                "direction": -1,
                "available_date": "2026-01-01",
                "source_type": "manual",
                "confidence": 0.5,
                "notes": "",
            },
        ]
    )
    attributions = pd.DataFrame(
        [
            ["pres_x", "event", "issue_candidate", "candidate_id", "cand_a", 1, 0.5, "2026-01-01", 0.5, ""],
            ["pres_x", "event", "issue_slot", "candidate_slot", "B", -1, 0.5, "2026-01-01", 0.5, ""],
            ["pres_x", "event", "issue_party", "party", "Party A", 1, 0.5, "2026-01-01", 0.5, ""],
            ["pres_x", "event", "issue_camp", "camp", "camp_b", 1, 0.5, "2026-01-01", 0.5, ""],
            ["pres_x", "event", "issue_incumbent", "incumbent_camp", "camp_a", -1, 0.5, "2026-01-01", 0.5, ""],
            ["pres_x", "event", "future_issue", "candidate_id", "cand_a", 1, 0.5, "2027-01-01", 0.5, ""],
        ],
        columns=[
            "election_id",
            "mega_event",
            "issue_name",
            "target_type",
            "target",
            "polarity",
            "weight",
            "available_date",
            "confidence",
            "notes",
        ],
    )
    data = {
        "candidates": candidates,
        "candidate_issue_profile": profiles,
        "mega_issue_axis": pd.DataFrame(
            [
                {
                    "election_id": "pres_x",
                    "mega_event": "event",
                    "primary_issue": "regime_change",
                    "secondary_issue": "",
                    "axis_weight": 1.0,
                    "regime_axis_weight": 0.0,
                    "available_date": "2026-01-01",
                    "activation_method": "manual",
                    "notes": "",
                }
            ]
        ),
        "mega_issue_attribution": attributions,
    }

    out = compile_enhanced_issue_scores(data, "2026-06-01")
    by_issue = {
        issue: sorted(rows["candidate_id"].unique())
        for issue, rows in out.groupby("issue_name")
        if issue.startswith("issue_")
    }

    assert by_issue["issue_candidate"] == ["cand_a"]
    assert by_issue["issue_slot"] == ["cand_b"]
    assert by_issue["issue_party"] == ["cand_a", "cand_c"]
    assert by_issue["issue_camp"] == ["cand_b"]
    assert by_issue["issue_incumbent"] == ["cand_a"]
    assert "future_issue" not in set(out["issue_name"])


def test_mega_attribution_clips_final_issue_score_to_unit_range() -> None:
    data = {
        "candidates": pd.DataFrame(
            [
                {
                    "candidate_id": "cand_a",
                    "candidate_name": "Candidate A",
                    "party_name": "Party A",
                    "official_camp": "camp_a",
                }
            ]
        ),
        "candidate_issue_profile": pd.DataFrame(),
        "mega_issue_axis": pd.DataFrame(
            [
                {
                    "election_id": "pres_x",
                    "mega_event": "event",
                    "primary_issue": "regime_change",
                    "secondary_issue": "",
                    "axis_weight": 2.0,
                    "regime_axis_weight": 0.0,
                    "available_date": "2026-01-01",
                    "activation_method": "manual",
                    "notes": "",
                }
            ]
        ),
        "mega_issue_attribution": pd.DataFrame(
            [
                {
                    "election_id": "pres_x",
                    "mega_event": "event",
                    "issue_name": "regime_change",
                    "target_type": "candidate_id",
                    "target": "cand_a",
                    "polarity": 1,
                    "weight": 1.0,
                    "available_date": "2026-01-01",
                    "confidence": 1.0,
                    "notes": "",
                }
            ]
        ),
    }

    out = compile_enhanced_issue_scores(data, "2026-06-01")

    assert out.loc[0, "final_issue_score"] == 1.0


def test_issue_scope_weights_split_national_and_local_components() -> None:
    candidate_scores = pd.DataFrame(
        [
            {
                "candidate_id": "cand_a",
                "issue_name": "housing",
                "issue_score": 0.5,
                "issue_salience": 0.5,
                "candidate_link_score": 1.0,
            }
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {"region_id": "r1", "issue_name": "housing", "sensitivity_score": 0.2},
            {"region_id": "r2", "issue_name": "housing", "sensitivity_score": 0.8},
        ]
    )

    national = compute_issue_impact(
        candidate_scores,
        sensitivity,
        pd.DataFrame([{"issue_name": "housing", "national_weight": 1.0, "local_weight": 0.0}]),
    )
    local = compute_issue_impact(
        candidate_scores,
        sensitivity,
        pd.DataFrame([{"issue_name": "housing", "national_weight": 0.0, "local_weight": 1.0}]),
    )

    assert dict(zip(national["region_id"], national["issue_impact"])) == {"r1": 0.5, "r2": 0.5}
    assert dict(zip(local["region_id"], local["issue_impact"])) == {"r1": 0.1, "r2": 0.4}


def test_forecast_utility_frame_includes_enhanced_issue_diagnostics() -> None:
    raw = load_raw_data("data")
    _inject_sample_candidates(raw)
    extra_profile = pd.DataFrame(
        [
            {
                "election_id": "dummy_presidential_2026",
                "candidate_id": "sample_cand_a",
                "slot": "A",
                "candidate_name": "Sample Candidate A",
                "issue_name": "housing",
                "association_strength": 0.8,
                "direction": 1,
                "available_date": "2026-04-01",
                "source_type": "manual",
                "confidence": 0.9,
                "notes": "test",
            }
        ]
    )
    raw["candidate_issue_profile"] = pd.concat([raw["candidate_issue_profile"], extra_profile], ignore_index=True)
    raw["issue_scope_weights"] = pd.DataFrame(
        [{"issue_name": "housing", "national_weight": 1.0, "local_weight": 0.0, "notes": "test"}]
    )
    raw["region_issue_sensitivity"] = pd.DataFrame(
        [
            {
                "region_id": "sample_sgg_001",
                "issue_name": "housing",
                "sensitivity_score": 0.7,
                "available_date": pd.Timestamp("2026-01-01"),
            },
            {
                "region_id": "sample_sgg_002",
                "issue_name": "housing",
                "sensitivity_score": 0.4,
                "available_date": pd.Timestamp("2026-01-01"),
            },
        ]
    )
    data = prepare_forecast_inputs(raw, "2026-05-01")

    frame, _ = compute_utility_frame(data, "2026-05-01", DEFAULT_CONFIG, ModelWeights())

    assert {"issue_impact", "issue_impact_national", "issue_impact_local"}.issubset(frame.columns)
    cand_a = frame.loc[frame["candidate_id"] == "sample_cand_a"]
    assert cand_a["issue_impact_national"].gt(0).any()
