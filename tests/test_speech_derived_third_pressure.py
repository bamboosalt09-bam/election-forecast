from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine.speech_derived_third_pressure import (
    build_automatic_third_candidate_pressure,
)


DATES = {"pres_test": pd.Timestamp("2024-04-10")}


def _profile() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "slot": "C",
                "candidate_name": "Third",
                "centrist_appeal": 0.64,
                "anti_major_party_appeal": 0.36,
                "available_date": "2024-04-01",
                "confidence": 0.81,
            }
        ]
    )


def _speech(third_bloc: str = "무소속") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "slot": "A",
                "candidate_name": "Conservative",
                "bloc": "국민의힘",
                "available_date": "2024-04-01",
                "confidence": 0.81,
            },
            {
                "election_id": "pres_test",
                "slot": "B",
                "candidate_name": "Liberal",
                "bloc": "더불어민주당",
                "available_date": "2024-04-01",
                "confidence": 0.81,
            },
            {
                "election_id": "pres_test",
                "slot": "C",
                "candidate_name": "Third",
                "bloc": third_bloc,
                "available_date": "2024-04-01",
                "confidence": 0.81,
            },
        ]
    )


def _landscape(conservative: float, liberal: float, progressive: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "slot": "C",
                "candidate_name": "Third",
                "conservative": conservative,
                "liberal": liberal,
                "progressive": progressive,
                "available_date": "2024-04-01",
                "confidence": 0.81,
            }
        ]
    )


def test_pressure_allocates_more_to_closer_major_lane() -> None:
    out = build_automatic_third_candidate_pressure(
        _profile(), _speech(), _landscape(0.9, 0.1, 0.0), DATES
    )
    values = out.set_index("source_slot")

    assert values.loc["A", "transfer_pressure"] > values.loc["B", "transfer_pressure"]
    assert out["transfer_pressure"].sum() == pytest.approx((0.64 * 0.36) ** 0.5)
    assert out["confidence"].tolist() == pytest.approx([0.81, 0.81])


def test_progressive_bloc_is_liberal_lane_fallback_without_landscape() -> None:
    out = build_automatic_third_candidate_pressure(
        _profile(), _speech("진보정당계"), pd.DataFrame(), DATES
    ).set_index("source_slot")

    assert out.loc["B", "ideological_affinity"] == 1.0
    assert out.loc["A", "ideological_affinity"] == 0.0
    assert out.loc["B", "transfer_pressure"] > out.loc["A", "transfer_pressure"]


def test_future_profile_is_excluded() -> None:
    profile = _profile()
    profile["available_date"] = "2024-04-11"

    out = build_automatic_third_candidate_pressure(
        profile, _speech(), _landscape(0.5, 0.5, 0.0), DATES
    )

    assert out.empty
