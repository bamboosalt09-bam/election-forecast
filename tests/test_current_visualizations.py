"""Guards against publishing superseded model figures as current evidence."""

from pathlib import Path

import numpy as np
import pytest

from presidential_issue_engine import make_poster_figures as figures


def test_current_inputs_are_v28_and_separate_forecast_from_history():
    assert figures.ACTIVE_DIR.name == "active_presidential_nested_v28"
    assert figures.FORECAST_DIR.name == "prospective_pres_2025_v28"
    assert not figures._history()["election_id"].astype(str).str.contains("2025").any()
    forecast = figures._forecast()
    assert set(forecast["election_id"]) == {"pres_2025"}
    assert np.allclose(forecast.groupby("region_id")["predicted_share"].sum(), 1.0)


def test_map_license_source_and_archive_are_pinned():
    assert "vuski/admdongkor" in figures.MAP_URL
    assert "ver20250401" in figures.MAP_URL
    assert len(figures.MAP_SHA256) == 64
    assert set(figures.SOURCE_SIDO_TO_REGION.values()) == set(figures._regions()["region_id"])
    assert set(figures.PIE_ANCHORS) == set(figures.SOURCE_SIDO_TO_REGION.values())
    assert figures.SLOT_COLORS == {"A": "#0878D1", "B": "#E33D3D", "C": "#F28C28"}


def test_map_rejects_non_compositional_forecast(monkeypatch):
    original = figures.pd.read_csv

    def altered(path, *args, **kwargs):
        frame = original(path, *args, **kwargs)
        if Path(path).name == "prospective_predictions.csv":
            frame.loc[frame.index[0], "predicted_share"] += 0.01
        return frame

    monkeypatch.setattr(figures.pd, "read_csv", altered)
    with pytest.raises(ValueError, match="compositional"):
        figures._forecast()


def test_active_generator_has_no_superseded_slot_importance_constants():
    source = Path(figures.__file__).read_text(encoding="utf-8")
    for obsolete in ("MAE_LADDER", "R2_LADDER", "VARIABLE_IMPORTANCE", "slotA_prior", "slotB_prior"):
        assert obsolete not in source


def test_readme_does_not_present_legacy_figures_as_current():
    readme = (figures.ROOT / "README.md").read_text(encoding="utf-8")
    assert "poster_figures/12_baseline_comparison.png" not in readme
    assert "poster_figures/14_prospective_forecast_v25.png" not in readme
    assert "poster_figures/v28_pres_2025_regional_map.png" in readme
