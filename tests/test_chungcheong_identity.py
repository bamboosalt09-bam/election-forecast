from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine import chungcheong_identity as identity


DATES = {
    "event_old": pd.Timestamp("2000-01-01"),
    "event_future": pd.Timestamp("2010-01-01"),
    "pres_test": pd.Timestamp("2005-01-01"),
}


def _date(value: object) -> pd.Timestamp | None:
    return DATES.get(str(value))


def test_identity_events_remove_national_third_party_component() -> None:
    rows = []
    for region, share in (("sido_11", 0.10), ("sido_30", 0.30), ("sido_43", 0.20)):
        rows.append(
            {
                "election_id": "event_old",
                "election_type": "assembly_pr",
                "region_id": region,
                "bloc": identity.THIRD_BLOC,
                "vote_share": share,
                "data_quality_weight": 1.0,
            }
        )
    events = identity.build_identity_events(pd.DataFrame(rows), date_resolver=_date)
    values = events.set_index("region_id")["identity_excess"]
    assert values["sido_11"] == 0.0
    assert np.isclose(values["sido_30"], 0.10)
    assert values["sido_43"] == 0.0


def test_profiles_exclude_future_events() -> None:
    events = pd.DataFrame(
        [
            {
                "election_id": "event_old",
                "event_date": pd.Timestamp("2000-01-01"),
                "region_id": "sido_30",
                "identity_excess": 0.20,
                "quality": 1.0,
                "type_weight": 1.0,
            },
            {
                "election_id": "event_future",
                "event_date": pd.Timestamp("2010-01-01"),
                "region_id": "sido_30",
                "identity_excess": 0.40,
                "quality": 1.0,
                "type_weight": 1.0,
            },
        ]
    )
    profiles = identity.fit_identity_profiles(
        events, cutoff=pd.Timestamp("2005-01-01"), prior_strength=0.01
    )
    direct = profiles.loc[profiles["region_id"].eq("sido_30")].iloc[0]
    assert direct["reservoir"] == 0.20
    assert direct["events"] == 1


def test_routing_is_pit_safe_chung_only_and_mass_conserving(monkeypatch) -> None:
    monkeypatch.setattr(identity, "election_date", lambda _: pd.Timestamp("2005-01-01"))
    frame = pd.DataFrame(
        [
            {"election_id": "pres_test", "region_id": region, "candidate_name": name, "pred": pred}
            for region in ("sido_30", "sido_11")
            for name, pred in (("A", 0.45), ("B", 0.55))
        ]
    )
    events = pd.DataFrame(
        [
            {
                "election_id": "event_old",
                "event_date": pd.Timestamp("2000-01-01"),
                "region_id": "sido_30",
                "identity_excess": 0.30,
                "quality": 1.0,
                "type_weight": 1.0,
            }
        ]
    )
    alignment = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "region_scope": "chungcheong",
                "candidate_name": "A",
                "affinity": 1.0,
                "confidence": 1.0,
                "available_date": "2004-01-01",
                "evidence_type": "test",
            },
            {
                "election_id": "pres_test",
                "region_scope": "chungcheong",
                "candidate_name": "B",
                "affinity": 1.0,
                "confidence": 1.0,
                "available_date": "2006-01-01",
                "evidence_type": "future",
            },
        ]
    )
    out, _ = identity.apply_identity_routing(
        frame,
        events,
        pd.DataFrame(),
        alignment,
        prediction_column="pred",
        gain=1.0,
        shift_cap=0.08,
        prior_strength=0.01,
    )
    chung = out.loc[out["region_id"].eq("sido_30")]
    other = out.loc[out["region_id"].eq("sido_11")]
    assert np.isclose(chung["pred"].sum(), 1.0)
    assert chung.loc[chung["candidate_name"].eq("A"), "pred"].iloc[0] > 0.45
    assert other["pred"].tolist() == [0.45, 0.55]


def test_candidate_regional_base_can_route_identity(monkeypatch) -> None:
    monkeypatch.setattr(identity, "election_date", lambda _: pd.Timestamp("2005-01-01"))
    frame = pd.DataFrame(
        [
            {"election_id": "pres_test", "region_id": "sido_43", "candidate_name": "A", "pred": 0.7},
            {"election_id": "pres_test", "region_id": "sido_43", "candidate_name": "C", "pred": 0.3},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "election_id": "event_old",
                "event_date": pd.Timestamp("2000-01-01"),
                "region_id": "sido_43",
                "identity_excess": 0.25,
                "quality": 1.0,
                "type_weight": 1.0,
            }
        ]
    )
    base = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "region_id": "sido_43",
                "candidate_name": "C",
                "regional_affinity": 0.8,
                "organization_depth": 0.8,
                "confidence": 0.8,
                "available_date": "2004-12-31",
            }
        ]
    )
    out, audit = identity.apply_identity_routing(
        frame,
        events,
        base,
        pd.DataFrame(),
        prediction_column="pred",
        gain=1.0,
        shift_cap=0.08,
        prior_strength=0.01,
    )
    assert out.loc[out["candidate_name"].eq("C"), "pred"].iloc[0] > 0.3
    assert audit.iloc[0]["evidence"] == "candidate_regional_base"
