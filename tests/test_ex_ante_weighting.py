"""Guards for the ex-ante weighting report.

The headline regional metric weights by the target election's own turnout, which
is not available before the election. These tests pin that the ex-ante
alternatives are computed from data that predates each target, and that the
comparison stays honest about which elections each weighting can cover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_ex_ante_weighting as exante
from scripts.active_model_pointer import active_output_dir, active_version


def _panel() -> pd.DataFrame:
    """Two elections, two regions, with the second region growing."""

    rows = []
    volumes = {
        ("pres_2002", "r1"): 100.0,
        ("pres_2002", "r2"): 300.0,
        ("pres_2007", "r1"): 200.0,
        ("pres_2007", "r2"): 200.0,
    }
    for (election, region), volume in volumes.items():
        for slot, actual, pred in (("A", 0.6, 0.5), ("B", 0.4, 0.5)):
            rows.append(
                {
                    "election_id": election,
                    "region_id": region,
                    "slot": slot,
                    "contest_votes": volume,
                    "actual": actual,
                    "layer_pred": pred,
                }
            )
    return pd.DataFrame(rows)


def test_prior_weights_come_from_the_preceding_election_only() -> None:
    volumes = exante.regional_volumes(_panel())
    prior, substituted = exante.prior_election_weights(volumes)

    assert set(prior["election_id"]) == {"pres_2007"}, "the first election has no prior"
    weights = prior.set_index("region_id")["weight"].to_dict()
    # 2007's weights must be 2002's volumes, not 2007's own
    assert weights == {"r1": 100.0, "r2": 300.0}
    assert substituted.get("pres_2007", 0) == 0


def test_a_region_without_a_predecessor_gets_the_prior_mean_and_is_counted() -> None:
    """세종 first appears in 2012; it must be substituted, not dropped."""

    panel = _panel()
    extra = panel.loc[panel.election_id.eq("pres_2007")].copy()
    extra["region_id"] = "r3"
    panel = pd.concat([panel, extra], ignore_index=True)

    volumes = exante.regional_volumes(panel)
    prior, substituted = exante.prior_election_weights(volumes)
    weights = prior.set_index("region_id")["weight"].to_dict()

    assert substituted["pres_2007"] == 1
    assert weights["r3"] == pytest.approx((100.0 + 300.0) / 2)
    assert set(weights) == {"r1", "r2", "r3"}, "no region may be dropped"


def test_equal_region_weighting_ignores_volume() -> None:
    """A weighting that needs no data must not depend on turnout."""

    panel = _panel()
    by_election, _ = exante.evaluate(panel)
    inflated = panel.copy()
    inflated.loc[inflated.region_id.eq("r2"), "contest_votes"] *= 10.0
    inflated_by_election, _ = exante.evaluate(inflated)

    pd.testing.assert_series_equal(
        by_election["equal_region_pp"],
        inflated_by_election["equal_region_pp"],
    )


def test_the_first_election_has_no_prior_weighting() -> None:
    by_election, _ = exante.evaluate(_panel())
    first = by_election.loc[by_election.election_id.eq("pres_2002")].iloc[0]
    assert np.isnan(first["prior_election_votes_pp"])
    assert not np.isnan(first["contest_votes_pp"])
    assert not np.isnan(first["equal_region_pp"])


def test_the_shipped_headline_is_reproduced_by_the_contest_votes_column() -> None:
    """The report must reproduce the pointer's own regional figure, not a variant."""

    import json

    from scripts.active_model_pointer import POINTER

    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    predictions = active_output_dir() / "nested_predictions.csv"
    if not predictions.exists():
        pytest.skip(f"{active_version().upper()} predictions are not present")
    frame = pd.read_csv(predictions, encoding="utf-8-sig", low_memory=False)
    by_election, _ = exante.evaluate(frame)
    assert by_election["contest_votes_pp"].mean() == pytest.approx(
        pointer["regional_equal_election_macro_mae_pp"], abs=1e-9
    )


def test_ex_ante_weightings_are_not_silently_better_than_the_headline() -> None:
    """If an ex-ante figure ever beat the headline, the claim would need rechecking."""

    predictions = active_output_dir() / "nested_predictions.csv"
    if not predictions.exists():
        pytest.skip(f"{active_version().upper()} predictions are not present")
    frame = pd.read_csv(predictions, encoding="utf-8-sig", low_memory=False)
    by_election, _ = exante.evaluate(frame)
    matched = by_election.loc[by_election["prior_election_votes_pp"].notna()]
    assert matched["contest_votes_pp"].mean() <= matched["prior_election_votes_pp"].mean()
    assert matched["contest_votes_pp"].mean() <= matched["equal_region_pp"].mean()
