from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine
from presidential_issue_engine.region_bloc_prior import (
    DISTRICT_TERRAIN_TYPE_WEIGHTS,
    attach_bloc_prior,
    compute_bloc_prior,
    election_date,
    load_bloc_history,
    normalize_bloc,
)
from presidential_issue_engine.build_bloc_history_from_nec import REGION_IDS


ORDER = ["e1", "e2", "e3"]


def test_normalize_bloc_maps_historical_party_aliases() -> None:
    assert normalize_bloc("\ud55c\ub098\ub77c\ub2f9") == "\uad6d\ubbfc\uc758\ud798"
    assert normalize_bloc("\ub300\ud1b5\ud569\ubbfc\uc8fc\uc2e0\ub2f9") == "\ub354\ubd88\uc5b4\ubbfc\uc8fc\ub2f9"
    assert normalize_bloc("\ubbfc\uc8fc\uc790\uc720\ub2f9") == "\uad6d\ubbfc\uc758\ud798"
    assert normalize_bloc("\uc0c8\uc815\uce58\uad6d\ubbfc\ud68c\uc758") == "\ub354\ubd88\uc5b4\ubbfc\uc8fc\ub2f9"
    assert normalize_bloc("\uad6d\ubbfc\uc2e0\ub2f9") == "\uc81c3\uc9c0\ub300"


def test_legacy_presidential_dates_and_jeju_alias_are_supported() -> None:
    assert election_date("pres_1992") == pd.Timestamp("1992-12-18")
    assert election_date("pres_1997") == pd.Timestamp("1997-12-18")
    assert REGION_IDS["\uc81c\uc8fc\ub3c4"] == "sido_50"


def test_compute_bloc_prior_uses_only_prior_elections() -> None:
    history = pd.DataFrame(
        [
            ("e1", "presidential", "r1", "A", 0.60, 1.0),
            ("e1", "presidential", "r2", "A", 0.40, 1.0),
            ("e2", "assembly_pr", "r1", "A", 0.70, 1.0),
            ("e2", "assembly_pr", "r2", "A", 0.30, 1.0),
            ("e3", "presidential", "r1", "A", 0.10, 1.0),
            ("e3", "presidential", "r2", "A", 0.90, 1.0),
        ],
        columns=[
            "election_id",
            "election_type",
            "region_id",
            "bloc",
            "vote_share",
            "data_quality_weight",
        ],
    )

    e2_prior = compute_bloc_prior(history, "e2", ORDER)
    e3_prior = compute_bloc_prior(history, "e3", ORDER)

    assert e2_prior.loc[e2_prior["region_id"] == "r1", "bloc_loyalty"].iloc[0] > 0
    assert e3_prior.loc[e3_prior["region_id"] == "r1", "bloc_loyalty"].iloc[0] > 0


def test_compute_bloc_prior_allows_same_year_sources_before_target_date() -> None:
    history = pd.DataFrame(
        [
            ("local_governor_2002", "local_governor", "sido_11", "A", 0.60, 1.0),
            ("local_governor_2002", "local_governor", "sido_26", "A", 0.40, 1.0),
            ("pres_2002", "presidential", "sido_11", "A", 0.10, 1.0),
            ("pres_2002", "presidential", "sido_26", "A", 0.90, 1.0),
        ],
        columns=[
            "election_id",
            "election_type",
            "region_id",
            "bloc",
            "vote_share",
            "data_quality_weight",
        ],
    )

    prior = compute_bloc_prior(history, "pres_2002", ["pres_2002"])

    assert set(prior["region_id"]) == {"sido_11", "sido_26"}
    assert prior.loc[prior["region_id"] == "sido_11", "bloc_loyalty"].iloc[0] > 0


def test_district_terrain_prior_excludes_party_list_and_presidential_rows() -> None:
    history = pd.DataFrame(
        [
            ("assembly_2004_pr", "assembly_pr", "sido_11", "A", 0.10, 1.0),
            ("assembly_2004_pr", "assembly_pr", "sido_26", "A", 0.90, 1.0),
            ("assembly_2004_district", "assembly_district", "sido_11", "A", 0.70, 1.0),
            ("assembly_2004_district", "assembly_district", "sido_26", "A", 0.10, 1.0),
        ],
        columns=[
            "election_id",
            "election_type",
            "region_id",
            "bloc",
            "vote_share",
            "data_quality_weight",
        ],
    )

    prior = compute_bloc_prior(
        history,
        "pres_2007",
        ["pres_2007"],
        election_type_weights=DISTRICT_TERRAIN_TYPE_WEIGHTS,
    )

    assert prior.loc[prior["region_id"] == "sido_11", "bloc_loyalty"].iloc[0] > 0
    assert prior.loc[prior["region_id"] == "sido_26", "bloc_loyalty"].iloc[0] < 0
    assert set(prior["effective_election_count"]) == {1}


def test_attach_bloc_prior_matches_slot_blocs() -> None:
    history = pd.DataFrame(
        [
            ("e1", "presidential", "r1", "A", 0.60, 1.0),
            ("e1", "presidential", "r2", "A", 0.40, 1.0),
            ("e1", "presidential", "r1", "B", 0.40, 1.0),
            ("e1", "presidential", "r2", "B", 0.60, 1.0),
        ],
        columns=[
            "election_id",
            "election_type",
            "region_id",
            "bloc",
            "vote_share",
            "data_quality_weight",
        ],
    )
    frame = pd.DataFrame(
        [
            {"election_id": "e2", "region_id": "r1", "slot": "slot_a", "bloc": "A"},
            {"election_id": "e2", "region_id": "r1", "slot": "slot_b", "bloc": "B"},
        ]
    )

    out = attach_bloc_prior(frame, history, ORDER)

    assert out.loc[out["slot"] == "slot_a", "partisan_prior"].iloc[0] > 0
    assert out.loc[out["slot"] == "slot_b", "partisan_prior"].iloc[0] < 0


def test_issue_engine_splits_concrete_and_general_partisan_prior() -> None:
    frame = pd.DataFrame(
        [
            {"partisan_prior": 0.40, "effective_election_count": 3},
            {"partisan_prior": -0.10, "effective_election_count": 3},
        ]
    )

    out = issue_vote_engine._split_partisan_prior_layers(frame)

    assert out.loc[0, "partisan_prior_raw"] == pytest.approx(0.40)
    assert out.loc[0, "concrete_partisan_prior"] == pytest.approx(
        issue_vote_engine.CONCRETE_PRIOR_CAP
    )
    assert out.loc[0, "general_partisan_prior"] == pytest.approx(
        (0.40 - issue_vote_engine.CONCRETE_PRIOR_CAP)
        * issue_vote_engine.GENERAL_PRIOR_SHRINK
    )
    assert out.loc[0, "partisan_prior"] == pytest.approx(out.loc[0, "partisan_prior_raw"])
    assert out.loc[1, "partisan_prior"] == pytest.approx(-0.10)


def test_issue_engine_builds_pre_2002_warmup_rows() -> None:
    warmup = issue_vote_engine.historical_presidential_warmup_frame()

    assert set(warmup["election_id"]) == {"pres_1992", "pres_1997"}
    assert {"A", "B", "C"}.issubset(set(warmup["slot"]))
    assert set(issue_vote_engine.PREDICTORS).issubset(warmup.columns)


def test_issue_engine_moderates_regional_deviation_by_prior_layers() -> None:
    frame = pd.DataFrame(
        [
            {
                "election_id": "e2",
                "region_id": "r1",
                "slot": "A",
                "concrete_partisan_prior": 0.05,
                "general_partisan_prior": 0.00,
            },
            {
                "election_id": "e2",
                "region_id": "r1",
                "slot": "B",
                "concrete_partisan_prior": -0.05,
                "general_partisan_prior": 0.00,
            },
            {
                "election_id": "e2",
                "region_id": "r2",
                "slot": "A",
                "concrete_partisan_prior": 0.05,
                "general_partisan_prior": 0.00,
            },
            {
                "election_id": "e2",
                "region_id": "r2",
                "slot": "B",
                "concrete_partisan_prior": -0.05,
                "general_partisan_prior": 0.00,
            },
        ]
    )
    pred = pd.Series([0.95, 0.05, 0.55, 0.45])

    out = issue_vote_engine.apply_partisan_layer_prediction_moderation(frame, pred)

    assert out[0] < 0.95
    assert out[1] > 0.05
    assert frame.assign(pred=out).groupby(["election_id", "region_id"])["pred"].sum().iloc[0] == pytest.approx(1.0)


def test_load_bloc_history_falls_back_to_presidential_results() -> None:
    presidential = pd.DataFrame(
        [
            {
                "election_id": "e1",
                "region_id": "r1",
                "slot": "A",
                "party_name": "\ud55c\ub098\ub77c\ub2f9",
                "vote_share": 0.55,
            }
        ]
    )

    history = load_bloc_history("missing_bloc_history.csv", presidential_results=presidential)

    assert history["bloc"].iloc[0] == "\uad6d\ubbfc\uc758\ud798"
    assert history["election_type"].iloc[0] == "presidential"
