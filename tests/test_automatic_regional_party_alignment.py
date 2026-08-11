from __future__ import annotations

import pandas as pd

from election_forecast.features.region_bloc_prior import (
    CONSERVATIVE_BLOC,
    LIBERAL_BLOC,
    THIRD_BLOC,
)
from presidential_issue_engine.automatic_regional_party_alignment import (
    SEMANTIC_HISTORY_TYPE_WEIGHTS,
    build_automatic_nonmajor_alignment,
    build_full_history_identity_events,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "assembly_2004_district",
                "election_type": "assembly_district",
                "region_id": "sido_44",
                "bloc": THIRD_BLOC,
                "vote_share": 0.35,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "assembly_2004_district",
                "election_type": "assembly_district",
                "region_id": "sido_11",
                "bloc": THIRD_BLOC,
                "vote_share": 0.05,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "assembly_2000_district",
                "election_type": "assembly_district",
                "region_id": "sido_44",
                "bloc": THIRD_BLOC,
                "vote_share": 0.30,
                "data_quality_weight": 1.0,
            },
            {
                "election_id": "assembly_2000_district",
                "election_type": "assembly_district",
                "region_id": "sido_11",
                "bloc": THIRD_BLOC,
                "vote_share": 0.04,
                "data_quality_weight": 1.0,
            },
        ]
    )


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "slot": "A",
                "candidate_name": "major",
                "bloc": CONSERVATIVE_BLOC,
                "outsider_status": 0.0,
                "available_date": "2007-12-18",
                "confidence": 0.8,
            },
            {
                "election_id": "pres_2007",
                "slot": "C",
                "candidate_name": "regional alternative",
                "bloc": "independent",
                "outsider_status": 0.9,
                "available_date": "2007-12-18",
                "confidence": 0.8,
            },
        ]
    )


def _landscape() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "slot": "A",
                "candidate_name": "major",
                "candidate_role": "final",
                "conservative": 0.35,
                "liberal": 0.30,
                "available_date": "2007-12-18",
                "confidence": 0.8,
            },
            {
                "election_id": "pres_2007",
                "slot": "C",
                "candidate_name": "regional alternative",
                "candidate_role": "final",
                "conservative": 0.55,
                "liberal": 0.20,
                "available_date": "2007-12-18",
                "confidence": 0.8,
            },
        ]
    )


def _bloc_landscape() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bloc": THIRD_BLOC,
                "conservative": 0.40,
                "liberal": 0.20,
            },
            {
                "bloc": LIBERAL_BLOC,
                "conservative": 0.20,
                "liberal": 0.40,
            },
        ]
    )


def test_full_history_reservoir_includes_constituency_elections() -> None:
    events = build_full_history_identity_events(_history())
    assert set(events["election_type"]) == {"assembly_district"}
    chungnam = events.loc[events["region_id"].eq("sido_44")]
    assert chungnam["identity_excess"].gt(0.0).all()


def test_semantic_history_downweights_candidate_ballots() -> None:
    events = build_full_history_identity_events(
        _history(), type_weights=SEMANTIC_HISTORY_TYPE_WEIGHTS
    )
    assert set(events["type_weight"]) == {0.25}
    assert SEMANTIC_HISTORY_TYPE_WEIGHTS["assembly_pr"] == 1.0
    assert SEMANTIC_HISTORY_TYPE_WEIGHTS["metro_council_district"] == 0.18
    assert SEMANTIC_HISTORY_TYPE_WEIGHTS["local_council_district"] == 0.10


def test_target_outcome_cannot_change_automatic_candidate_alignment() -> None:
    base, _ = build_automatic_nonmajor_alignment(
        _history(), _context(), _landscape(), _bloc_landscape()
    )
    contaminated = pd.concat(
        [
            _history(),
            pd.DataFrame(
                [
                    {
                        "election_id": "pres_2007",
                        "election_type": "presidential",
                        "region_id": "sido_44",
                        "bloc": THIRD_BLOC,
                        "vote_share": 0.99,
                        "data_quality_weight": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    changed, _ = build_automatic_nonmajor_alignment(
        contaminated, _context(), _landscape(), _bloc_landscape()
    )
    assert base[["election_id", "candidate_name", "affinity"]].equals(
        changed[["election_id", "candidate_name", "affinity"]]
    )
    assert base.iloc[0]["candidate_name"] == "regional alternative"
