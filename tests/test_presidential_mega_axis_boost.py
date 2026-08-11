from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine


def test_mega_axis_boost_multiplies_only_configured_election_issue(tmp_path, monkeypatch) -> None:
    axis = tmp_path / "mega_issue_axis.csv"
    axis.write_text(
        "\n".join(
            [
                "election_id,mega_event,primary_issue,secondary_issue,axis_weight,regime_axis_weight,available_date,activation_method,notes",
                "pres_2022,event,regime_change,corruption_integrity,1.5,1.2,2022-03-01,manual,",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_AXIS", str(axis))
    monkeypatch.setenv("POLL_PROJECT_MEGA_AXIS_BOOST_STRENGTH", "1")
    monkeypatch.setattr(issue_vote_engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(tmp_path / "missing.csv"))
    frame = pd.DataFrame(
        [
            {"election_id": "pres_2022", "issue_name": "regime_change", "salience": 0.2},
            {"election_id": "pres_2022", "issue_name": "corruption_integrity", "salience": 0.2},
            {"election_id": "pres_2022", "issue_name": "housing", "salience": 0.2},
            {"election_id": "pres_2017", "issue_name": "regime_change", "salience": 0.2},
        ]
    )

    out = issue_vote_engine._apply_mega_axis_salience_boost(frame)
    values = dict(zip(zip(out["election_id"], out["issue_name"]), out["salience"]))

    assert values[("pres_2022", "regime_change")] == pytest.approx(0.3)
    assert values[("pres_2022", "corruption_integrity")] == pytest.approx(0.24)
    assert values[("pres_2022", "housing")] == 0.2
    assert values[("pres_2017", "regime_change")] == 0.2


def test_mega_axis_boost_uses_election_intensity(tmp_path, monkeypatch) -> None:
    axis = tmp_path / "mega_issue_axis.csv"
    axis.write_text(
        "\n".join(
            [
                "election_id,mega_event,primary_issue,secondary_issue,axis_weight,regime_axis_weight,available_date,activation_method,notes",
                "pres_2022,event,regime_change,,1.5,0.0,2022-03-01,manual,",
            ]
        ),
        encoding="utf-8",
    )
    intensity = tmp_path / "mega_issue_intensity.csv"
    intensity.write_text(
        "\n".join(
            [
                "election_id,mega_issue_intensity,available_date,notes",
                "pres_2022,2.0,2022-03-01,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_AXIS", str(axis))
    monkeypatch.setattr(issue_vote_engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(intensity))
    monkeypatch.setenv("POLL_PROJECT_MEGA_AXIS_BOOST_STRENGTH", "0.1")
    frame = pd.DataFrame([{"election_id": "pres_2022", "issue_name": "regime_change", "salience": 0.2}])

    out = issue_vote_engine._apply_mega_axis_salience_boost(frame)

    assert out.loc[0, "salience"] == pytest.approx(0.22)


def test_issue_epoch_importance_applies_confident_multiplier(tmp_path, monkeypatch) -> None:
    epoch = tmp_path / "issue_epoch_importance.csv"
    epoch.write_text(
        "\n".join(
            [
                "election_id,issue_name,importance_multiplier,available_date,confidence,notes",
                "pres_2022,housing,1.5,2022-03-01,0.5,test",
                "pres_2022,economy_growth,2.0,2022-04-01,1.0,future",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "ISSUE_EPOCH_IMPORTANCE", str(epoch))
    frame = pd.DataFrame(
        [
            {"election_id": "pres_2022", "issue_name": "housing", "salience": 0.2},
            {"election_id": "pres_2022", "issue_name": "economy_growth", "salience": 0.2},
        ]
    )

    out = issue_vote_engine._apply_issue_epoch_importance(frame)
    values = dict(zip(out["issue_name"], out["salience"]))

    assert values["housing"] == pytest.approx(0.2125)
    assert values["economy_growth"] == pytest.approx(0.2)
