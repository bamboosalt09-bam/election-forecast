import pandas as pd

from election_forecast.config import ForecastConfig
from election_forecast.party_base import compute_party_base


def test_party_base_computes_region_camp_base() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["e1", "e1"],
            "election_type": ["presidential", "presidential"],
            "election_date": ["2025-01-01", "2025-01-01"],
            "region_id": ["r1", "r1"],
            "party_name": ["p1", "p2"],
            "camp": ["conservative", "liberal"],
            "candidate_name": ["a", "b"],
            "votes": [60, 40],
            "vote_share": [0.6, 0.4],
            "turnout": [0.7, 0.7],
            "available_date": ["2025-01-02", "2025-01-02"],
        }
    )

    party_base = compute_party_base(frame, "2026-01-01", ForecastConfig())

    row = party_base.loc[party_base["region_id"] == "r1"].iloc[0]
    assert row["conservative"] == 0.6
    assert row["liberal"] == 0.4


def test_party_base_prefers_bloc_history_when_target_election_is_available() -> None:
    stale_results = pd.DataFrame(
        {
            "election_id": ["e1", "e1"],
            "election_type": ["presidential", "presidential"],
            "election_date": ["2025-01-01", "2025-01-01"],
            "region_id": ["r1", "r1"],
            "party_name": ["p1", "p2"],
            "camp": ["conservative", "liberal"],
            "candidate_name": ["a", "b"],
            "votes": [80, 20],
            "vote_share": [0.8, 0.2],
            "turnout": [0.7, 0.7],
            "available_date": ["2025-01-02", "2025-01-02"],
        }
    )
    bloc_history = pd.DataFrame(
        [
            {
                "election_id": "assembly_2024_pr",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "더불어민주당",
                "vote_share": 0.65,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "assembly_2024_pr",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "국민의힘",
                "vote_share": 0.25,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "assembly_2024_pr",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "제3지대",
                "vote_share": 0.10,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "assembly_2024_pr",
                "election_type": "assembly_pr",
                "region_id": "r1",
                "bloc": "기타소수정당",
                "vote_share": 0.50,
                "data_quality_weight": 1.0,
            },
        ]
    )

    party_base = compute_party_base(
        stale_results,
        "2026-01-01",
        ForecastConfig(),
        bloc_history=bloc_history,
        target_election_id="pres_2026",
    )

    row = party_base.loc[party_base["region_id"] == "r1"].iloc[0]
    assert row["liberal"] == 0.65
    assert row["conservative"] == 0.25
    assert row["centrist"] == 0.10
    assert row["anti_party"] == 0.0
