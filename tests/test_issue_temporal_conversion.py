from __future__ import annotations

import pandas as pd

from presidential_issue_engine import issue_vote_engine


def test_issue_temporal_conversion_applies_available_multiplier(tmp_path, monkeypatch) -> None:
    path = tmp_path / "issue_temporal_conversion.csv"
    pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "issue_name": "corruption_integrity",
                "conversion_multiplier": 1.20,
                "temporal_sensitivity": 1.0,
                "available_date": "2007-01-01",
                "confidence": 0.5,
            }
        ]
    ).to_csv(path, index=False)
    monkeypatch.setattr(issue_vote_engine, "ISSUE_TEMPORAL_CONVERSION", str(path))
    monkeypatch.setenv("POLL_PROJECT_ISSUE_TEMPORAL_CONVERSION_SCALE", "1.0")
    adv = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "issue_name": "corruption_integrity",
                "slot": "A",
                "salience": 10.0,
            }
        ]
    )

    out = issue_vote_engine._apply_issue_temporal_conversion(adv)

    assert out["salience"].iloc[0] == 11.0


def test_issue_temporal_conversion_ignores_future_available_date(tmp_path, monkeypatch) -> None:
    path = tmp_path / "issue_temporal_conversion.csv"
    pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "issue_name": "corruption_integrity",
                "conversion_multiplier": 1.20,
                "temporal_sensitivity": 1.0,
                "available_date": "2008-01-01",
                "confidence": 1.0,
            }
        ]
    ).to_csv(path, index=False)
    monkeypatch.setattr(issue_vote_engine, "ISSUE_TEMPORAL_CONVERSION", str(path))
    monkeypatch.setenv("POLL_PROJECT_ISSUE_TEMPORAL_CONVERSION_SCALE", "1.0")
    adv = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "issue_name": "corruption_integrity",
                "slot": "A",
                "salience": 10.0,
            }
        ]
    )

    out = issue_vote_engine._apply_issue_temporal_conversion(adv)

    assert out["salience"].iloc[0] == 10.0


def test_issue_scope_weights_use_manual_over_assembly_derived(tmp_path, monkeypatch) -> None:
    derived = tmp_path / "derived_scope.csv"
    manual = tmp_path / "manual_scope.csv"
    pd.DataFrame(
        [
            {
                "issue_name": "housing",
                "national_weight": 0.25,
                "local_weight": 0.75,
            },
            {
                "issue_name": "security_nk",
                "national_weight": 0.80,
                "local_weight": 0.20,
            },
        ]
    ).to_csv(derived, index=False)
    pd.DataFrame(
        [
            {
                "issue_name": "housing",
                "national_weight": 1.00,
                "local_weight": 0.00,
            }
        ]
    ).to_csv(manual, index=False)
    monkeypatch.setattr(issue_vote_engine, "ASSEMBLY_DERIVED_ISSUE_SCOPE_WEIGHTS", str(derived))
    monkeypatch.setattr(issue_vote_engine, "ENHANCED_ISSUE_SCOPE_WEIGHTS", str(manual))

    out = issue_vote_engine._issue_scope_weights()

    rows = out.set_index("issue_name")
    assert rows.loc["housing", "national_weight"] == 1.0
    assert rows.loc["security_nk", "national_weight"] == 0.8
