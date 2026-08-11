from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine.issue_vote_engine import apply_party_tone_gap_prediction_adjustment


def test_party_stance_proxy_rewards_endorsement_and_penalizes_attack(monkeypatch) -> None:
    monkeypatch.setenv("POLL_PROJECT_ENABLE_PARTY_STANCE_PROXY", "1")
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "region_id": "sido_x",
                "slot": "A",
                "party_stance_signal_centered": 0.5,
                "party_tone_confidence": 1.0,
                "manual_valence_coverage": 1.0,
                "party_stance_proxy_available": 1.0,
            },
            {
                "election_id": "pres_x",
                "region_id": "sido_x",
                "slot": "B",
                "party_stance_signal_centered": -0.5,
                "party_tone_confidence": 1.0,
                "manual_valence_coverage": 1.0,
                "party_stance_proxy_available": 1.0,
            },
        ]
    )

    out = apply_party_tone_gap_prediction_adjustment(frame, np.array([0.5, 0.5]))

    assert out[0] > 0.5
    assert out[1] < 0.5


def test_party_stance_proxy_does_not_activate_for_legacy_tone_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_x",
                "region_id": "sido_x",
                "slot": "A",
                "party_stance_signal_centered": 0.9,
                "party_tone_confidence": 1.0,
                "manual_valence_coverage": 1.0,
                "party_stance_proxy_available": 0.0,
            },
            {
                "election_id": "pres_x",
                "region_id": "sido_x",
                "slot": "B",
                "party_stance_signal_centered": -0.9,
                "party_tone_confidence": 1.0,
                "manual_valence_coverage": 1.0,
                "party_stance_proxy_available": 0.0,
            },
        ]
    )

    out = apply_party_tone_gap_prediction_adjustment(frame, np.array([0.5, 0.5]))

    assert np.allclose(out, [0.5, 0.5])
