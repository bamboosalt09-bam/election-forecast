import pandas as pd

from election_forecast.presidential.schemas import SLOTS, normalize_slots


def test_presidential_slots_load_as_abcalpha() -> None:
    slots = normalize_slots(pd.read_csv("data/presidential/candidate_slots.csv"))
    pres_2022 = slots.loc[slots["election_id"] == "pres_2022"]

    assert set(pres_2022["slot"]) == set(SLOTS)


def test_presidential_2012_c_slot_is_inactive() -> None:
    slots = normalize_slots(pd.read_csv("data/presidential/candidate_slots.csv"))
    c_slot = slots.loc[(slots["election_id"] == "pres_2012") & (slots["slot"] == "C")].iloc[0]

    assert bool(c_slot["is_active_slot"]) is False
