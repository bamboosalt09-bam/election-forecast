from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine.direct_party_center import apply_direct_party_center


def test_direct_party_center_preserves_third_candidate_and_total_mass() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_x"] * 3,
            "region_id": ["r1"] * 3,
            "major_party_core_eligible": [True, True, False],
            "direct_party_recent_base": [0.60, 0.40, 0.0],
            "layer_pred": [0.45, 0.35, 0.20],
        }
    )
    result = apply_direct_party_center(
        frame,
        prediction_column="layer_pred",
        gain_by_election={"pres_x": 0.25},
    )

    assert np.isclose(result["layer_pred"].sum(), 1.0)
    assert np.isclose(result.loc[2, "layer_pred"], 0.20)
    assert np.allclose(result["layer_pred"], [0.4575, 0.3425, 0.20])


def test_direct_party_center_is_identity_without_two_eligible_candidates() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_x"] * 2,
            "region_id": ["r1"] * 2,
            "major_party_core_eligible": [True, False],
            "direct_party_recent_base": [0.60, 0.0],
            "layer_pred": [0.70, 0.30],
        }
    )
    result = apply_direct_party_center(
        frame,
        prediction_column="layer_pred",
        gain_by_election={"pres_x": 0.25},
    )
    assert np.allclose(result["layer_pred"], frame["layer_pred"])
