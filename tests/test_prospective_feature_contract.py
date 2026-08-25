"""No active prospective feature may become zero by omission.

The 2025 forecast shipped for four versions with two model-active families
silently zeroed - the whole `regional_accent_*` group and
`major_party_core_eligible` - because the target assembly filled every missing
column with `0.0` under a comment asserting such columns were inert.

Fixing those two columns is not the same as fixing the class. These tests pin
the invariant: a column the target lacks is an outcome, a declared default, or
a model-active family with a builder, and anything else stops the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_issue_engine import prospective_feature_contract as contract

ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": ["pres_2025"] * 3,
            "region_id": ["sido_11", "sido_26", "sido_29"],
            "slot": list("ABC"),
            "bloc": ["더불어민주당", "국민의힘", "제3지대"],
        }
    )


def test_an_unclassified_missing_column_stops_the_run() -> None:
    """The exact shape of the bug: a column nobody classified becoming zero."""

    with pytest.raises(contract.ProspectiveFeatureError) as error:
        contract.resolve(_frame(), ["election_id", "some_new_predictor"])
    assert "no declared kind" in str(error.value)
    assert "some_new_predictor" in str(error.value)


def test_a_model_active_family_without_a_builder_stops_the_run() -> None:
    with pytest.raises(contract.ProspectiveFeatureError) as error:
        contract.resolve(_frame(), ["election_id", "regional_accent_reliability"])
    assert "model-active" in str(error.value)


def test_outcome_columns_become_nan_never_zero() -> None:
    """A zero here is a fabricated result that reads as a real one."""

    out = contract.resolve(_frame(), ["election_id", "actual", "err_pp"])
    for column in ("actual", "err_pp"):
        assert out[column].isna().all(), f"{column} must be NaN, not a value"


def test_declared_defaults_carry_a_stated_reason() -> None:
    for column, (_, reason) in contract.DECLARED_DEFAULTS.items():
        assert isinstance(reason, str) and len(reason.split()) >= 5, (
            f"{column} is defaulted without a usable reason"
        )


def test_a_builder_that_omits_a_column_stops_the_run() -> None:
    def incomplete(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"regional_accent_reliability": np.zeros(len(frame))}, index=frame.index)

    with pytest.raises(contract.ProspectiveFeatureError) as error:
        contract.resolve(
            _frame(),
            ["regional_accent_reliability", "regional_accent_signal"],
            {"regional_accent": incomplete},
        )
    assert "did not produce" in str(error.value)


def test_the_two_families_that_were_silently_zeroed_are_model_active() -> None:
    assert contract.classify("regional_accent_reliability") == "active:regional_accent"
    assert contract.classify("regional_accent_conservative_share") == "active:regional_accent"
    assert contract.classify("major_party_core_eligible") == "active:major_party_core_eligible"


def test_the_contract_run_produces_both_active_families() -> None:
    """Checked against the artifact the contract produced, not against code.

    The *published* demonstration is still V31's, which has the zeros - that is
    the defect this fix exists to remove, and it stays true until V32 is
    promoted. So the invariant is asserted against the contract's own output.
    """

    artifact = ROOT / "outputs/prospective_pres_2025_v32/prediction_stage_audit.csv"
    if not artifact.is_file():
        pytest.skip("the contract run has not been produced in this tree")
    frame = pd.read_csv(artifact, encoding="utf-8-sig", low_memory=False)

    reliability = pd.to_numeric(frame["regional_accent_reliability"], errors="coerce").fillna(0.0)
    assert (reliability.abs() > 1e-12).all(), (
        "the contract run still has rows with no regional accent reliability"
    )
    eligible = frame["major_party_core_eligible"].astype(str).str.lower()
    assert eligible.isin({"true", "1", "1.0"}).sum() == 34, (
        "the two major-party nominees should be core eligible in all 17 regions"
    )


def test_the_published_demonstration_is_recorded_as_defective_until_promoted() -> None:
    """Names the gap rather than letting it pass silently.

    While the pointer still selects a demonstration built before the contract,
    that artifact carries the zeroed families. This test states which artifact
    is active and that the condition is known, so the repository never reports
    a clean bill of health on a forecast that has the defect.
    """

    pointer = json.loads(
        (ROOT / "data/config/current_presidential_model.json").read_text(encoding="utf-8")
    )
    artifact = ROOT / str(pointer["prospective_demonstration"]["artifact"])
    audit = artifact / "prediction_stage_audit.csv"
    if not audit.is_file():
        return
    frame = pd.read_csv(audit, encoding="utf-8-sig", low_memory=False)
    reliability = pd.to_numeric(frame["regional_accent_reliability"], errors="coerce").fillna(0.0)
    clean = bool((reliability.abs() > 1e-12).any())
    record = ROOT / "docs/DIAGNOSIS_2025_ACCENT_ZEROING_20260825.md"
    assert clean or record.is_file(), (
        f"the active demonstration {artifact.name} has the zeroed accent layer and "
        "no diagnosis record explains it"
    )
