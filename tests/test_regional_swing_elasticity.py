from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine import regional_swing_elasticity as swing


def test_profile_uses_only_events_before_cutoff() -> None:
    events = pd.DataFrame(
        {
            "election_id": ["e1", "e2", "e3"],
            "event_date": pd.to_datetime(["2000-01-01", "2004-01-01", "2008-01-01"]),
            "region_id": ["sido_30"] * 3,
            "national_logit": [-0.2, 0.0, 3.0],
            "regional_logit": [-0.1, 0.2, -3.0],
            "quality": [1.0, 1.0, 1.0],
        }
    )
    profile = swing.fit_profiles(events, cutoff=pd.Timestamp("2008-01-01"), prior_strength=2.0)
    row = swing.profile_for_region(profile, "sido_30")
    assert row is not None
    assert int(row["events"]) == 2
    assert float(row["slope"]) > 1.0


def test_chungcheong_hierarchy_handles_missing_region() -> None:
    events = pd.DataFrame(
        {
            "election_id": ["e1", "e2", "e1", "e2"],
            "event_date": pd.to_datetime(["2000-01-01", "2004-01-01"] * 2),
            "region_id": ["sido_30", "sido_30", "sido_44", "sido_44"],
            "national_logit": [-0.2, 0.2, -0.2, 0.2],
            "regional_logit": [-0.1, 0.3, -0.3, 0.1],
            "quality": [1.0] * 4,
        }
    )
    profiles = swing.fit_profiles(events, cutoff=pd.Timestamp("2008-01-01"))
    sejong = swing.profile_for_region(profiles, "sido_36")
    assert sejong is not None
    prediction = swing.predict_region_share(sejong, 0.55, method="elasticity")
    assert 0.0 < prediction < 1.0


def test_flat_profile_prediction_is_national_share() -> None:
    assert np.isclose(
        swing.predict_region_share(None, 0.57, method="elasticity"), 0.57
    )
