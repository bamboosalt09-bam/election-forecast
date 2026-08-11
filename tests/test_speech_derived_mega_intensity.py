import pandas as pd
import pytest

from presidential_issue_engine.speech_derived_mega_intensity import (
    build_automatic_mega_issue_intensity,
    gate_intensity_by_event_class,
)


DATES = {"pres_low": "2020-01-10", "pres_high": "2024-01-10"}


def _row(
    election_id: str,
    period: str,
    speaker: str,
    issue: str,
    weight: float,
    count: int = 1,
) -> dict[str, object]:
    return {
        "election_id": election_id,
        "period": period,
        "speaker": speaker,
        "issue_name": issue,
        "issue_weight": weight,
        "matched_term_count": count,
    }


def test_broad_dense_regime_axis_receives_higher_intensity() -> None:
    rows = []
    for election, date, regime_weight in [
        ("pres_low", "2020-01-01", 0.2),
        ("pres_high", "2024-01-01", 1.0),
    ]:
        for index in range(8):
            issue = "regime_change" if index < (2 if election == "pres_low" else 6) else "economy_growth"
            rows.append(_row(election, date, f"speaker_{index}", issue, regime_weight))
        rows.append(_row(election, date, "accountability", "corruption_integrity", 1.0))
    output, diagnostics = build_automatic_mega_issue_intensity(pd.DataFrame(rows), DATES)
    scores = output.set_index("election_id")["mega_issue_intensity"]
    assert scores["pres_high"] > scores["pres_low"]
    assert set(diagnostics["source_model"]) == {"speech_derived_mega_intensity_v1"}


def test_future_rows_are_excluded() -> None:
    rows = [
        _row("pres_low", "2020-01-01", "before", "regime_change", 0.5),
        _row("pres_low", "2020-01-11", "future", "regime_change", 10.0),
        _row("pres_low", "2020-01-01", "context", "corruption_integrity", 0.5),
    ]
    output, diagnostics = build_automatic_mega_issue_intensity(pd.DataFrame(rows), DATES)
    assert diagnostics.loc[0, "source_rows"] == 2
    assert output.loc[0, "available_date"] == "2020-01-01"


def test_event_class_gate_ignores_numeric_taxonomy_columns() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "election_id": "pres_low",
                "joint_evidence": 0.8,
                "available_date": "2020-01-01",
            }
        ]
    )
    taxonomy = pd.DataFrame(
        [
            {
                "election_id": "pres_low",
                "shock_type": "institutional_crisis",
                "available_date": "2020-01-02",
                "severity": 0.01,
                "confidence": 0.01,
            }
        ]
    )
    output, audit = gate_intensity_by_event_class(diagnostics, taxonomy, DATES)
    assert output.loc[0, "mega_issue_intensity"] == pytest.approx(1.7)
    assert audit.loc[0, "event_class_level"] == 1.0
