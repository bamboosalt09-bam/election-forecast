from __future__ import annotations

import json

import pandas as pd

from scripts import build_active_v24_predictive_intervals as intervals


def test_active_v24_intervals_are_chronological_and_nested(tmp_path) -> None:
    payload = intervals.build(n_sim=1_000, seed=7_024, output_dir=tmp_path)

    detail = pd.read_csv(tmp_path / "national_predictive_intervals.csv")
    assert len(detail) == 44
    assert set(detail["election_id"]) == set(intervals.ORDER[1:])
    assert not detail["training_elections"].str.contains("pres_2025").any()
    assert not detail["target_outcome_used_to_construct_bounds"].any()

    election_position = {name: index for index, name in enumerate(intervals.ORDER)}
    for row in detail.itertuples(index=False):
        training = str(row.training_elections).split("|")
        assert all(
            election_position[election] < election_position[row.election_id]
            for election in training
        )

    for _, group in detail.groupby(["election_id", "slot"]):
        ordered = group.sort_values("nominal_level")
        assert ordered["lower_share"].is_monotonic_decreasing
        assert ordered["upper_share"].is_monotonic_increasing

    manifest = json.loads(
        (tmp_path / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    assert payload == manifest
    assert manifest["residual_scale_policy"] == "fixed_unscaled_not_selected_on_coverage"
    assert manifest["post_2022_outcomes_used"] is False
