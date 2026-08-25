"""No active feature may be zero across the whole forecast while alive in the panel.

The contract in `prospective_feature_contract` guards one specific gap: columns
the historical frame has and the target lacks. It cannot see a column that is
*present* in the target and merely empty, and two of the five defective
families were exactly that - `wasted_vote_resistance` arrived populated and was
dropped by a reindex, then refilled with zero from a history-only table.

So this checks the symptom directly and over every shared column, not only the
ones a Ridge stage reads: if a column is identically zero across all 51
forecast rows while carrying values for a scored election, it is either a
declared class or a defect. There is no third possibility, and the test says
which.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from presidential_issue_engine import prospective_feature_contract as contract

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
REFERENCE_ELECTION = "pres_2022"


def _pointer() -> dict:
    return json.loads(POINTER.read_text(encoding="utf-8"))


def _frames() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    pointer = _pointer()
    forecast = ROOT / str(pointer["prospective_demonstration"]["artifact"]) / "prediction_stage_audit.csv"
    scored = ROOT / str(pointer["output"]) / "nested_predictions.csv"
    if not forecast.is_file() or not scored.is_file():
        return None
    a = pd.read_csv(forecast, encoding="utf-8-sig", low_memory=False)
    b = pd.read_csv(scored, encoding="utf-8-sig", low_memory=False)
    return a, b.loc[b["election_id"].astype(str).eq(REFERENCE_ELECTION)]


def zeroed_only_in_the_forecast(forecast: pd.DataFrame, scored: pd.DataFrame) -> list[str]:
    """Columns dead in the forecast and alive in the reference election."""

    found = []
    for column in [c for c in forecast.columns if c in scored.columns]:
        a = pd.to_numeric(forecast[column], errors="coerce")
        b = pd.to_numeric(scored[column], errors="coerce")
        if a.notna().sum() == 0 or b.notna().sum() == 0:
            continue
        if (a.fillna(0.0).abs() < 1e-12).all() and (b.fillna(0.0).abs() > 1e-12).any():
            found.append(column)
    return found


def test_every_column_dead_in_the_forecast_has_a_declared_class() -> None:
    frames = _frames()
    if frames is None:
        pytest.skip("the pointer's artifacts are not present")
    zeroed = zeroed_only_in_the_forecast(*frames)
    unclassified = [c for c in zeroed if contract.classify(c)[0] == "UNCLASSIFIED"]
    assert not unclassified, (
        f"{len(unclassified)} column(s) are zero across the whole forecast, alive "
        f"in {REFERENCE_ELECTION}, and belong to no declared class: "
        f"{sorted(unclassified)[:12]}"
    )


def test_no_required_derived_family_is_dead_in_a_contract_built_forecast() -> None:
    """The stronger statement, once a forecast is built under the contract.

    A REQUIRED_DERIVED column that is still zero means its builder did not run
    or did not reach the artifact. While the active demonstration predates the
    contract this is checked against the contract's own output instead.
    """

    built = ROOT / "outputs/prospective_pres_2025_v32/prediction_stage_audit.csv"
    if not built.is_file():
        pytest.skip("no contract-built forecast in this tree")
    frames = _frames()
    if frames is None:
        pytest.skip("the pointer's artifacts are not present")
    _, scored = frames
    forecast = pd.read_csv(built, encoding="utf-8-sig", low_memory=False)

    zeroed = zeroed_only_in_the_forecast(forecast, scored)
    required = [
        column
        for column in zeroed
        if contract.classify(column)[0] == contract.REQUIRED_DERIVED
    ]
    assert not required, (
        "these required-derived columns are still zero in the contract-built "
        f"forecast: {sorted(required)}"
    )


def test_no_input_table_carries_a_2025_outcome() -> None:
    """Condition: the forecast cannot read the result even if it wanted to.

    Rather than mutate the 2025 result and re-run - which needs a 2025 result
    to exist - this asserts the stronger property that produced the same
    guarantee: no results table the model reads contains pres_2025 at all. The
    only transcription of the outcome lives under `evaluations/`, read by the
    post-election evaluator alone.
    """

    for path in ROOT.rglob("presidential_results_standardized.csv"):
        if any(part in {".git", "archives", "backups"} for part in path.parts):
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "election_id" not in frame.columns:
            continue
        assert not frame["election_id"].astype(str).eq("pres_2025").any(), (
            f"{path} carries a 2025 outcome and is on the model's input path"
        )


def test_the_contract_forecast_declares_it_used_no_outcome() -> None:
    manifest = ROOT / "outputs/prospective_pres_2025_v32/run_manifest.json"
    if not manifest.is_file():
        pytest.skip("no contract-built forecast in this tree")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["performance_metrics_computed"] is False
    contract_record = payload["prospective_feature_contract"]
    assert contract_record["target_election_outcome_fields_used"] == []
    assert contract_record["unclassified_missing_column_behaviour"] == "raise"
