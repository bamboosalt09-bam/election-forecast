from __future__ import annotations

import pandas as pd

from presidential_issue_engine import issue_vote_engine


def test_issue_temporal_salience_boosts_late_rising_issue() -> None:
    salience = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "issue_name": "late_issue",
                "period": "2007-10-01",
                "salience_score": 0.10,
                "available_date": "2007-10-07",
            },
            {
                "election_id": "pres_2007",
                "issue_name": "late_issue",
                "period": "2007-11-26",
                "salience_score": 0.90,
                "available_date": "2007-12-02",
            },
        ]
    )

    out = issue_vote_engine._issue_temporal_salience(salience)

    row = out.iloc[0]
    assert row["salience"] > salience["salience_score"].mean()
    assert row["salience_late_momentum"] > 0


def test_issue_temporal_salience_can_fall_back_to_plain_mean(monkeypatch) -> None:
    monkeypatch.setenv("POLL_PROJECT_DISABLE_ISSUE_TEMPORAL_WEIGHTING", "1")
    salience = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "issue_name": "late_issue",
                "period": "2007-10-01",
                "salience_score": 0.10,
                "available_date": "2007-10-07",
            },
            {
                "election_id": "pres_2007",
                "issue_name": "late_issue",
                "period": "2007-11-26",
                "salience_score": 0.90,
                "available_date": "2007-12-02",
            },
        ]
    )

    out = issue_vote_engine._issue_temporal_salience(salience)

    assert out["salience"].iloc[0] == salience["salience_score"].mean()
    assert out["salience_late_momentum"].iloc[0] == 0.0


def test_issue_residual_stock_decays_to_d1_and_persistent_mega_issue_lingers_longer(
    tmp_path, monkeypatch
) -> None:
    axis = tmp_path / "mega_issue_axis.csv"
    axis.write_text(
        "election_id,mega_event,primary_issue,secondary_issue,axis_weight,regime_axis_weight,available_date,activation_method,notes\n"
        "pres_2017,event,regime_change,,1.0,0.0,2017-04-01,automatic,test\n",
        encoding="utf-8",
    )
    taxonomy = tmp_path / "mega_issue_taxonomy.csv"
    taxonomy.write_text(
        "election_id,mega_event,shock_type,severity,national_scope,persistence,polarization,target_specificity,available_date,confidence,notes\n"
        "pres_2017,event,institutional_crisis,1.0,1.0,1.0,1.0,1.0,2017-04-01,1.0,test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_AXIS", str(axis))
    monkeypatch.setattr(issue_vote_engine, "MEGA_ISSUE_TAXONOMY", str(taxonomy))
    salience = pd.DataFrame(
        [
            {"election_id": "pres_2017", "issue_name": "regime_change", "period": "2017-04-01", "salience_score": 0.8, "available_date": "2017-04-07"},
            {"election_id": "pres_2017", "issue_name": "plain_issue", "period": "2017-04-01", "salience_score": 0.8, "available_date": "2017-04-07"},
        ]
    )

    out = issue_vote_engine._issue_temporal_salience(salience).set_index("issue_name")

    assert out.loc["regime_change", "salience_residual_stock"] > 0.0
    assert out.loc["regime_change", "salience_residual_stock"] > out.loc["plain_issue", "salience_residual_stock"]


def test_issue_temporal_salience_excludes_post_cutoff_period_even_if_marked_available() -> None:
    salience = pd.DataFrame(
        [
            {"election_id": "pres_2017", "issue_name": "valid", "period": "2017-05-01", "salience_score": 0.4, "available_date": "2017-05-02"},
            {"election_id": "pres_2017", "issue_name": "future", "period": "2017-05-10", "salience_score": 1.0, "available_date": "2017-05-02"},
        ]
    )

    out = issue_vote_engine._issue_temporal_salience(salience)

    assert set(out["issue_name"]) == {"valid"}
