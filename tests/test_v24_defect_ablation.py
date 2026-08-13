from __future__ import annotations

import json

import numpy as np
import pandas as pd

from presidential_issue_engine.v24_defect_ablation import (
    apply_prior_selected_response_with_fixed_caps,
)
from scripts import run_v24_defect_ablation


def test_fixed_contest_caps_are_forwarded_to_response() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2002", "pres_2007"],
            "region_id": ["sido_11", "sido_11"],
            "source_slot": ["A", "A"],
            "layer_pred": [0.55, 0.60],
            "actual": [0.55, 0.60],
            "contest_votes": [1.0, 1.0],
        }
    )
    regimes = pd.DataFrame(
        {
            "election_id": ["pres_2002", "pres_2007"],
            "dominance_activation": [0.0, 0.0],
        }
    )
    calls: list[tuple[float, float]] = []

    def response(data: pd.DataFrame, _regimes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        calls.append((kwargs["log_shift_cap"], kwargs["swing_log_shift_cap"]))
        return data.copy()

    result, audit = apply_prior_selected_response_with_fixed_caps(
        frame,
        regimes,
        prediction_column="layer_pred",
        apply_response=response,
        election_order=("pres_2002", "pres_2007"),
        log_shift_cap=0.40,
        swing_log_shift_cap=0.50,
    )

    assert len(result) == len(frame)
    assert calls == [(0.40, 0.50), (0.40, 0.50)]
    assert np.allclose(audit["fixed_log_shift_cap"], 0.40)
    assert np.allclose(audit["fixed_swing_log_shift_cap"], 0.50)


def test_v24_ablation_configs_are_experimental(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_v24_defect_ablation, "CONFIG_DIR", tmp_path)
    paths = run_v24_defect_ablation.build_configs()

    assert set(paths) == set(run_v24_defect_ablation.VARIANTS)
    for name, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["active"] is False
        assert payload["promotion"]["status"] == "experimental_not_active"
        assert payload["promotion"]["single_change"] == name

    a1 = json.loads(paths["a1_remove_rif"].read_text(encoding="utf-8"))
    assert "rif" not in a1["predictors"]
    d1 = json.loads(
        paths["d1_disable_dead_general_identity"].read_text(encoding="utf-8")
    )
    assert d1["structural_layers"]["general_regional_identity"]["enabled"] is False
