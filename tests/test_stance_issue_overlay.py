import pandas as pd

from presidential_issue_engine import issue_vote_engine as engine


def test_active_stance_issue_overlay_is_bounded_by_default(monkeypatch) -> None:
    monkeypatch.delenv("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH", raising=False)
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "issue_name": ["housing"],
            "slot": ["A"],
            "salience": [2.0],
            "emphasis_within": [-0.5],
        }
    )

    result = engine._apply_stance_issue_overlay(frame)

    assert 0.95 <= result.loc[0, "stance_salience_multiplier"] <= 1.05
    assert 0.98 <= result.loc[0, "stance_link_multiplier"] <= 1.02
    assert result.loc[0, "salience"] > 0.0
    assert result.loc[0, "emphasis_within"] < 0.0


def test_stance_issue_overlay_is_identity_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH", "off")
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "issue_name": ["housing"],
            "slot": ["A"],
            "salience": [2.0],
            "emphasis_within": [-0.5],
        }
    )
    result = engine._apply_stance_issue_overlay(frame)
    assert result.loc[0, "salience"] == 2.0
    assert result.loc[0, "emphasis_within"] == -0.5


def test_stance_issue_overlay_changes_magnitude_not_sign(monkeypatch, tmp_path) -> None:
    path = tmp_path / "overlay.csv"
    pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "issue_name": ["housing"],
            "slot": ["A"],
            "salience_multiplier": [1.10],
            "link_multiplier": [0.96],
            "available_date": ["2022-03-08"],
        }
    ).to_csv(path, index=False)
    monkeypatch.setattr(
        engine,
        "THROUGH_2022_REDERIVED_LAYER_CONFIG",
        {
            "overlay_gain": 0.1,
            "third_competitiveness_gate_enabled": False,
            "third_character_multiplier_enabled": False,
        },
    )
    monkeypatch.setenv("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH", str(path))
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "issue_name": ["housing"],
            "slot": ["A"],
            "salience": [2.0],
            "emphasis_within": [-0.5],
        }
    )
    result = engine._apply_stance_issue_overlay(frame)
    assert result.loc[0, "salience"] == 2.2
    assert result.loc[0, "emphasis_within"] < 0.0
    assert result.loc[0, "emphasis_within"] == -0.48


def test_stance_issue_overlay_excludes_rows_after_forecast_cutoff(monkeypatch, tmp_path) -> None:
    path = tmp_path / "future_overlay.csv"
    pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "issue_name": ["housing"],
            "slot": ["A"],
            "salience_multiplier": [1.10],
            "link_multiplier": [1.04],
            "available_date": ["2022-03-09"],
        }
    ).to_csv(path, index=False)
    monkeypatch.setattr(
        engine,
        "THROUGH_2022_REDERIVED_LAYER_CONFIG",
        {
            "overlay_gain": 0.1,
            "third_competitiveness_gate_enabled": False,
            "third_character_multiplier_enabled": False,
        },
    )
    monkeypatch.setenv("POLL_PROJECT_STANCE_ISSUE_OVERLAY_PATH", str(path))
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "issue_name": ["housing"],
            "slot": ["A"],
            "salience": [2.0],
            "emphasis_within": [-0.5],
        }
    )

    result = engine._apply_stance_issue_overlay(frame)

    assert result.loc[0, "salience"] == 2.0
    assert result.loc[0, "emphasis_within"] == -0.5
