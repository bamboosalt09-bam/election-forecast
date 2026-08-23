"""Guards for the fold training-depth diagnostic.

Strict chronological nesting makes the first scored election the least trained.
That is the design working, not a defect, but it means the equal-election macro
mixes folds that are not comparable. These tests pin the property so that a
change to the panel or the nesting rule surfaces here rather than silently
altering how the headline should be read.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import diagnose_fold_training_depth as depth
from scripts.active_model_pointer import active_output_dir, active_version

ACTIVE_DIR = active_output_dir()


def _report() -> pd.DataFrame:
    if not (ACTIVE_DIR / "fold_audit.csv").exists():
        pytest.skip(f"the {active_version().upper()} fold audit is not present")
    return depth.report(ACTIVE_DIR)


def test_training_depth_increases_with_the_chronological_order() -> None:
    """Each later fold must have every earlier fold's training and one more."""

    report = _report()
    counts = report["training_election_count"].tolist()
    assert counts == sorted(counts), "folds must be ordered by training depth"
    assert counts == list(range(counts[0], counts[0] + len(counts))), (
        "chronological nesting adds exactly one election per fold"
    )


def test_the_first_scored_fold_trains_on_the_warmup_alone() -> None:
    report = _report()
    first = report.iloc[0]
    assert first["target_election"] == "pres_2002"
    assert first["training_election_count"] == 1
    assert "pres_1997" in str(first["training_elections"])


def test_the_shallowest_fold_has_a_degenerate_design() -> None:
    """One training election leaves no cross-election variation to be collinear in."""

    report = _report()
    if "raw_max_predictor_vif" not in report.columns:
        pytest.skip("the fold audit does not record predictor VIF")
    first = report.iloc[0]
    deepest = report.iloc[-1]
    assert float(first["raw_max_predictor_vif"]) == pytest.approx(1.0, abs=1e-6)
    assert float(deepest["raw_max_predictor_vif"]) > 1.0, (
        "a fold with real training should show some collinearity"
    )


def test_the_shallowest_fold_dominates_the_national_macro() -> None:
    """If this ever stops being true the headline reads differently."""

    report = _report()
    first = report.iloc[0]
    assert float(first["share_of_national_macro"]) > 0.5, (
        "one fold carrying most of the macro is the fact the README discloses"
    )
    assert report["share_of_national_macro"].sum() == pytest.approx(1.0)


def test_the_worst_national_fold_is_not_the_worst_regional_fold() -> None:
    """The 2002 error is a level error, not a shape error; keep them separated."""

    report = _report()
    worst_national = report.loc[report["national_mae_pp"].idxmax(), "target_election"]
    worst_regional = report.loc[
        report["regional_weighted_mae_pp"].idxmax(), "target_election"
    ]
    assert worst_national == "pres_2002"
    assert worst_regional != worst_national, (
        "conflating shape and level would misdirect the diagnosis"
    )
