from __future__ import annotations

import pandas as pd

from election_forecast.features.region_bloc_prior import (
    CONSERVATIVE_BLOC,
    THIRD_BLOC,
)
from presidential_issue_engine.speech_derived_candidate_regional_base import (
    build_automatic_candidate_regional_base,
)


DATES = {"pres_2017": pd.Timestamp("2017-05-09")}


def _speech() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "A",
                "candidate_name": "Major",
                "bloc": CONSERVATIVE_BLOC,
                "organization_strength": 0.9,
                "available_date": "2017-05-08",
                "confidence": 0.8,
            },
            {
                "election_id": "pres_2017",
                "slot": "C",
                "candidate_name": "Third",
                "bloc": THIRD_BLOC,
                "organization_strength": 0.64,
                "available_date": "2017-05-08",
                "confidence": 0.75,
            },
        ]
    )


def _history() -> pd.DataFrame:
    rows = []
    for region, share in [("r1", 0.5), ("r2", 0.3), ("r3", 0.1)]:
        rows.append(
            {
                "election_id": "assembly_2016_pr",
                "election_type": "assembly_pr",
                "region_id": region,
                "bloc": THIRD_BLOC,
                "vote_share": share,
                "data_quality_weight": 1.0,
            }
        )
    rows.extend(
        [
            {
                "election_id": "assembly_2016_district",
                "election_type": "assembly_district",
                "region_id": "r2",
                "bloc": THIRD_BLOC,
                "vote_share": 0.9,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "assembly_2020_pr",
                "election_type": "assembly_pr",
                "region_id": "r3",
                "bloc": THIRD_BLOC,
                "vote_share": 0.9,
                "data_quality_weight": 1.0,
            },
        ]
    )
    return pd.DataFrame(rows)


def test_latest_prior_direct_party_ballot_identifies_positive_excess() -> None:
    out = build_automatic_candidate_regional_base(_speech(), _history(), DATES)

    assert out["region_id"].tolist() == ["r1"]
    assert out.iloc[0]["regional_affinity"] == 1.0
    assert out.iloc[0]["organization_depth"] == 0.8
    assert out.iloc[0]["confidence"] == 0.75
    assert out.iloc[0]["source_election_ids"] == "assembly_2016_pr"


def test_major_party_candidate_is_not_duplicated_as_personal_base() -> None:
    out = build_automatic_candidate_regional_base(_speech(), _history(), DATES)

    assert set(out["slot"]) == {"C"}
