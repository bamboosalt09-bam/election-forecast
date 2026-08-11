from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine import regional_swing_elasticity as swing


def test_maximum_vif_is_computed_from_predictors_only() -> None:
    from scripts.evaluate_preliminary_slot_shadow_nested import _maximum_predictor_vif

    x = np.column_stack([np.arange(10.0), np.arange(10.0) * 2.0, np.ones(10)])
    assert np.isinf(_maximum_predictor_vif(x))


def test_regional_offset_preserves_third_candidate_mass(monkeypatch) -> None:
    profile = pd.DataFrame(
        {
            "region_id": ["sido_30"],
            "source": ["region"],
            "intercept": [0.0],
            "slope": [1.0],
            "offset": [0.4],
            "effective_n": [4.0],
            "reliability": [0.5],
            "events": [4],
        }
    )
    monkeypatch.setattr(swing, "fit_profiles", lambda *args, **kwargs: profile)
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2012"] * 3,
            "region_id": ["sido_30"] * 3,
            "bloc": ["국민의힘", "더불어민주당", "제3지대"],
            "major_party_core_eligible": [True, True, False],
            "layer_pred": [0.40, 0.40, 0.20],
        }
    )
    result = swing.apply_regional_offset(
        frame,
        pd.DataFrame(),
        prediction_column="layer_pred",
        gain_by_election={"pres_2012": 0.2},
    )
    assert np.isclose(result["layer_pred"].sum(), 1.0)
    assert np.isclose(result.loc[2, "layer_pred"], 0.20)
    assert result.loc[0, "layer_pred"] > 0.40
