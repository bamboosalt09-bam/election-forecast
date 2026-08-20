from __future__ import annotations

import json

from scripts import build_active_v25_predictive_intervals as intervals


def test_active_v25_intervals_use_v25_point_artifact(tmp_path) -> None:
    payload = intervals.build(n_sim=2_000, output_dir=tmp_path)

    assert payload["model_version"] == "v25"
    assert payload["schema"] == "active_v25_national_predictive_intervals_v1"
    assert payload["post_2022_outcomes_used"] is False
    assert payload["target_outcomes_used_to_construct_bounds"] is False
    assert payload["input"].endswith(
        "outputs/active_presidential_nested_v25/nested_predictions.csv"
    )
    written = json.loads(
        (tmp_path / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    assert written["model_version"] == "v25"
