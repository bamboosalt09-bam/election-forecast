from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine
from presidential_issue_engine.issue_vote_engine import (
    _apply_macro_phrase_bonus,
    _apply_macro_speech_strength,
    _kospi_context_features,
)
from scripts.import_kospi_history import parse_kospi_text
from scripts.fetch_bok_kospi_daily import normalize_bok_kospi_rows


def test_bok_kospi_normalizer_keeps_official_close_only() -> None:
    rows = [
        {"TIME": "20220307", "DATA_VALUE": "2651.31"},
        {"TIME": "20220308", "DATA_VALUE": "2622.40"},
        {"TIME": "20220308", "DATA_VALUE": "2622.40"},
    ]

    out = normalize_bok_kospi_rows(rows)

    assert len(out) == 2
    assert out["date"].tolist() == [pd.Timestamp("2022-03-07"), pd.Timestamp("2022-03-08")]
    assert out["close"].tolist() == pytest.approx([2651.31, 2622.40])
    assert out[["open", "high", "low", "volume"]].isna().all().all()
    assert out["source"].eq("Bank of Korea ECOS").all()
    assert out["ohlc_quality_flag"].eq("official_close_only").all()
    assert out["available_date"].equals(out["date"])


def test_kospi_text_parser_deduplicates_and_preserves_quality_flags() -> None:
    date_label = "2022\ub144 03\uc6d4 08\uc77c"
    text = "\n".join(
        [
            f"{date_label}\t2,600.00\t2,610.00\t2,620.00\t2,590.00\t10.00M\t-0.50%",
            f"{date_label}\t2,600.00\t2,610.00\t2,620.00\t2,590.00\t10.00M\t-0.50%",
            "2022\ub144 03\uc6d4 07\uc77c\t2,613.00\t2,610.00\t2,612.00\t2,600.00\t1.00B\t+0.20%",
        ]
    )

    out = parse_kospi_text(text, "fixture.txt")

    assert len(out) == 2
    assert out["volume"].max() == 1_000_000_000.0
    assert out.loc[out["date"].eq(pd.Timestamp("2022-03-07")), "ohlc_quality_flag"].iloc[0] == (
        "source_range_inconsistent"
    )


def test_kospi_context_excludes_election_day_and_penalizes_responsible_slot() -> None:
    dates = pd.bdate_range("2020-01-01", "2021-03-01")
    close = np.linspace(120.0, 80.0, len(dates))
    kospi = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "available_date": dates,
        }
    )
    kospi.loc[kospi["date"].eq(pd.Timestamp("2021-03-01")), "close"] = 1000.0
    alignment = pd.DataFrame(
        [
            {"election_id": "pres_x", "slot": "A", "economic_responsibility_score": 1.0},
            {"election_id": "pres_x", "slot": "B", "economic_responsibility_score": -1.0},
        ]
    )

    out = _kospi_context_features(kospi, alignment, {"pres_x": "2021-03-01"})
    effects = dict(zip(out["slot"], out["kospi_context_effect"]))

    assert set(out["kospi_latest_date"]) == {"2021-02-26"}
    assert out["kospi_close"].iloc[0] < 100.0
    assert out["kospi_market_stress_index"].iloc[0] > 0.0
    assert effects["A"] < 0.0
    assert effects["B"] > 0.0


def test_active_engine_exposes_time_varying_macro_diagnostics_without_direct_predictor() -> None:
    out = issue_vote_engine.assemble()

    assert "macro_context_signal" not in issue_vote_engine.PREDICTORS
    assert "macro_context_signal" not in out.columns
    fixed_dataset = (
        Path(__file__).resolve().parents[1]
        / "presidential_issue_engine"
        / "fixed_dataset"
    )
    active_kospi_sources = (
        fixed_dataset / "kospi_election_context.csv",
        fixed_dataset / "kospi_daily.csv",
    )
    if any(path.exists() for path in active_kospi_sources):
        assert out["kospi_latest_date"].ne("not_available").any()
    else:
        assert out["kospi_latest_date"].eq("not_available").all()
    epoch = out.groupby("election_id", as_index=False)[
        ["economy_epoch_weight", "housing_epoch_weight"]
    ].first()
    assert np.allclose(epoch[["economy_epoch_weight", "housing_epoch_weight"]].sum(axis=1), 1.0)
    assert epoch["economy_epoch_weight"].nunique() > 1
    assert np.allclose(
        out[
            [
                "growth_within_economy_weight",
                "trade_within_economy_weight",
                "kospi_within_economy_weight",
                "interest_rate_within_economy_weight",
            ]
        ].sum(axis=1),
        1.0,
    )


def test_macro_issue_reinforcement_changes_salience_not_vote_share(monkeypatch) -> None:
    reinforcement = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "issue_name": "economy_growth",
                "macro_issue_multiplier": 1.10,
            }
        ]
    )
    monkeypatch.setattr(issue_vote_engine, "_macro_issue_reinforcement_table", lambda: reinforcement)
    frame = pd.DataFrame(
        [
            {"election_id": "pres_2022", "issue_name": "economy_growth", "salience": 0.5},
            {"election_id": "pres_2022", "issue_name": "security_nk", "salience": 0.5},
            {"election_id": "pres_2022", "issue_name": "economy_growth", "salience": 0.0},
        ]
    )

    out = issue_vote_engine._apply_macro_issue_reinforcement(frame)

    assert out["salience"].tolist() == pytest.approx([0.55, 0.5, 0.0])
    assert "pred" not in out.columns


def test_macro_speech_strength_and_phrase_bonus_preserve_zero_inputs() -> None:
    reinforcement = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "issue_name": "housing",
                "macro_speech_strength_multiplier": 1.08,
                "macro_phrase_bonus_multiplier": 1.05,
            }
        ]
    )
    speech = pd.DataFrame(
        [
            {"election_id": "pres_2022", "issue_name": "housing", "salience_score": 0.5},
            {"election_id": "pres_2022", "issue_name": "housing", "salience_score": 0.0},
            {"election_id": "pres_2022", "issue_name": "security_nk", "salience_score": 0.5},
        ]
    )
    phrase = pd.DataFrame(
        [
            {"election_id": "pres_2022", "slot": "A", "issue_name": "housing", "emphasis_within": 0.4},
            {"election_id": "pres_2022", "slot": "A", "issue_name": "housing", "emphasis_within": 0.0},
            {"election_id": "pres_2022", "slot": "A", "issue_name": "security_nk", "emphasis_within": 0.4},
        ]
    )

    speech_out = _apply_macro_speech_strength(speech, reinforcement)
    phrase_out = _apply_macro_phrase_bonus(phrase, reinforcement)

    assert speech_out["salience_score"].tolist() == pytest.approx([0.54, 0.0, 0.5])
    assert phrase_out["emphasis_within"].tolist() == pytest.approx([0.42, 0.0, 0.4])


def test_macro_issue_epoch_weights_are_bounded_and_time_varying() -> None:
    out = issue_vote_engine._macro_issue_reinforcement_table()
    epoch = out.groupby("election_id", as_index=False)[
        ["economy_epoch_weight", "housing_epoch_weight"]
    ].first()

    assert np.allclose(epoch[["economy_epoch_weight", "housing_epoch_weight"]].sum(axis=1), 1.0)
    assert epoch["economy_epoch_weight"].nunique() > 1
    assert np.allclose(
        out[
            [
                "growth_within_economy_weight",
                "trade_within_economy_weight",
                "kospi_within_economy_weight",
                "interest_rate_within_economy_weight",
            ]
        ].sum(axis=1),
        1.0,
    )
    assert out["macro_speech_strength_multiplier"].between(1.0, 1.08).all()
    assert out["macro_phrase_bonus_multiplier"].between(1.0, 1.05).all()
    assert out["macro_issue_multiplier"].between(1.0, 1.15).all()
    assert out["macro_issue_multiplier"].gt(1.0).any()
