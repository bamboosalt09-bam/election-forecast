from __future__ import annotations

import pytest

from scripts import compute_forecast_baselines as baselines


def test_frozen_v23_baselines_match_documented_macros() -> None:
    predictions, history, _ = baselines._read_inputs()
    by_election = baselines.compute_baselines(predictions, history)
    summary = baselines.build_summary(by_election)
    macro = summary["macro_mae_pp"]

    assert by_election["election_id"].tolist() == [
        "pres_2002",
        "pres_2007",
        "pres_2012",
        "pres_2017",
        "pres_2022",
    ]
    assert macro["model"] == pytest.approx(3.3678986784747)
    assert macro["persistence"] == pytest.approx(13.011470607394397)
    assert macro["uniform_national_swing"] == pytest.approx(8.86104408733107)


def test_uniform_national_swing_is_marked_oracle_aided() -> None:
    predictions, history, _ = baselines._read_inputs()
    summary = baselines.build_summary(baselines.compute_baselines(predictions, history))

    assert summary["uniform_national_swing_is_oracle_aided"] is True
    assert summary["skill_tests"]["vs_uniform_national_swing"]["n"] == 5
