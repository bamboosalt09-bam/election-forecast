from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine


def test_scored_contest_scope_loader_excludes_only_available_false_rows(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "scored_contest_scope.csv"
    pd.DataFrame(
        [
            {
                "election_id": "pres_2002",
                "slot": "C",
                "include_in_scored_contest": False,
                "available_date": "2002-12-18",
            },
            {
                "election_id": "pres_2007",
                "slot": "C",
                "include_in_scored_contest": False,
                "available_date": "2007-12-20",
            },
            {
                "election_id": "pres_2017",
                "slot": "C",
                "include_in_scored_contest": True,
                "available_date": "2017-05-08",
            },
        ]
    ).to_csv(path, index=False)
    monkeypatch.setattr(issue_vote_engine, "SCORED_CONTEST_SCOPE", str(path))

    assert issue_vote_engine._load_scored_contest_scope_exclusions() == {
        ("pres_2002", "C")
    }


def test_2002_scored_contest_is_two_way_without_altering_raw_result() -> None:
    raw = pd.read_csv(issue_vote_engine.RESULTS)
    raw_c = raw.loc[
        raw["election_id"].eq("pres_2002") & raw["slot"].eq("C")
    ]
    assert not raw_c.empty
    assert raw_c["is_active_slot"].astype(str).str.lower().eq("true").all()

    frame = issue_vote_engine.assemble()
    context = frame.loc[frame["election_id"].eq("pres_2002")]
    assert set(context["slot"]) == {"A", "B", "C"}
    scored = issue_vote_engine.scored_contest_rows(context)
    assert set(scored["slot"]) == {"A", "B"}
    normalized = issue_vote_engine.normalized_vote_share_target(scored)
    sums = pd.Series(normalized, index=scored.index).groupby(scored["region_id"]).sum()
    assert sums.to_numpy() == pytest.approx(1.0)


def test_scored_scope_changes_only_the_evaluation_flag(tmp_path, monkeypatch) -> None:
    excluded = tmp_path / "excluded.csv"
    included = tmp_path / "included.csv"
    base = {
        "election_id": "pres_2002",
        "slot": "C",
        "available_date": "2002-12-18",
    }
    pd.DataFrame([{**base, "include_in_scored_contest": False}]).to_csv(
        excluded, index=False
    )
    pd.DataFrame([{**base, "include_in_scored_contest": True}]).to_csv(
        included, index=False
    )

    monkeypatch.setattr(issue_vote_engine, "SCORED_CONTEST_SCOPE", str(excluded))
    excluded_frame = issue_vote_engine.assemble().sort_values(
        ["election_id", "region_id", "slot"]
    ).reset_index(drop=True)
    monkeypatch.setattr(issue_vote_engine, "SCORED_CONTEST_SCOPE", str(included))
    included_frame = issue_vote_engine.assemble().sort_values(
        ["election_id", "region_id", "slot"]
    ).reset_index(drop=True)

    assert excluded_frame.drop(columns="is_scored_contest_row").equals(
        included_frame.drop(columns="is_scored_contest_row")
    )
    assert not excluded_frame.loc[
        excluded_frame["election_id"].eq("pres_2002")
        & excluded_frame["slot"].eq("C"),
        "is_scored_contest_row",
    ].any()
    assert included_frame.loc[
        included_frame["election_id"].eq("pres_2002")
        & included_frame["slot"].eq("C"),
        "is_scored_contest_row",
    ].all()
    assert "is_scored_contest_row" not in issue_vote_engine.PREDICTORS


def test_rolling_training_backfills_only_missing_target_slots() -> None:
    train = pd.DataFrame(
        [
            {"election_id": "pres_1997", "slot": "A", "value": 1},
            {"election_id": "pres_1997", "slot": "B", "value": 2},
            {"election_id": "pres_1997", "slot": "C", "value": 3},
            {"election_id": "pres_2002", "slot": "A", "value": 4},
            {"election_id": "pres_2002", "slot": "B", "value": 5},
        ]
    )
    test = pd.DataFrame(
        [
            {"election_id": "pres_2007", "slot": "A"},
            {"election_id": "pres_2007", "slot": "B"},
            {"election_id": "pres_2007", "slot": "C"},
        ]
    )

    selected, residual_mask = issue_vote_engine.rolling_training_with_slot_backfill(
        train,
        test,
        {"pres_1997"},
    )

    assert selected[["election_id", "slot"]].to_records(index=False).tolist() == [
        ("pres_2002", "A"),
        ("pres_2002", "B"),
        ("pres_1997", "C"),
    ]
    assert residual_mask.tolist() == [True, True, False]
