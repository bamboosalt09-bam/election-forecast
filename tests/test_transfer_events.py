import pandas as pd
import pytest

from election_forecast.presidential.transfer import apply_transfer_adjustments, load_transfer_events


def _utilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": ["pres_2022"] * 8,
            "region_id": ["sgg_001"] * 4 + ["sgg_002"] * 4,
            "region_name": ["r1"] * 4 + ["r2"] * 4,
            "province": ["p"] * 8,
            "slot": ["A", "B", "C", "alpha"] * 2,
            "is_active_slot": [True] * 8,
            "model_name": ["balanced"] * 8,
            "utility": [0.0] * 8,
        }
    )


def test_transfer_adjusts_target_source_and_alpha() -> None:
    events = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "event_date": ["2022-03-08"],
            "available_date": ["2022-03-08"],
            "source_slot": ["C"],
            "target_slot": ["A"],
            "region_id": ["sgg_001"],
            "transfer_strength": [0.2],
            "transfer_rate": [0.5],
            "abstention_rate": [0.25],
            "notes": ["sample"],
        }
    )

    adjusted, contributions = apply_transfer_adjustments(_utilities(), load_transfer_events(events, "pres_2022"))

    r1 = adjusted.loc[adjusted["region_id"] == "sgg_001"].set_index("slot")
    assert r1.loc["A", "utility"] == 0.1
    assert r1.loc["C", "utility"] == pytest.approx(-0.15)
    assert r1.loc["alpha", "utility"] == 0.05
    assert len(contributions) == 3


def test_transfer_region_all_applies_to_all_regions() -> None:
    events = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "event_date": ["2022-03-08"],
            "available_date": ["2022-03-08"],
            "source_slot": ["C"],
            "target_slot": ["B"],
            "region_id": ["ALL"],
            "transfer_strength": [0.1],
            "transfer_rate": [0.5],
            "abstention_rate": [0.0],
            "notes": ["sample"],
        }
    )

    adjusted, _ = apply_transfer_adjustments(_utilities(), load_transfer_events(events, "pres_2022"))

    b_rows = adjusted.loc[adjusted["slot"] == "B"]
    assert set(b_rows["utility"]) == {0.05}


def test_transfer_events_reject_missing_availability() -> None:
    events = pd.DataFrame(
        {
            "election_id": ["pres_2022"],
            "event_date": ["2022-03-08"],
            "available_date": [None],
            "source_slot": ["C"],
            "target_slot": ["A"],
            "region_id": ["ALL"],
            "transfer_strength": [0.1],
            "transfer_rate": [0.5],
            "abstention_rate": [0.0],
        }
    )

    with pytest.raises(ValueError, match="missing or invalid"):
        load_transfer_events(events, "pres_2022", "2022-03-08")
