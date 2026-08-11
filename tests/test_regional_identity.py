from __future__ import annotations

import numpy as np
import pandas as pd

from presidential_issue_engine import regional_identity as identity


DATES = {
    "old": pd.Timestamp("2000-01-01"),
    "future": pd.Timestamp("2010-01-01"),
    "pres_test": pd.Timestamp("2005-01-01"),
}


def _date(value: object) -> pd.Timestamp | None:
    return DATES.get(str(value))


def test_distinctiveness_uses_total_variation_from_event_median() -> None:
    rows = []
    distributions = {
        "sido_11": (0.50, 0.50),
        "sido_26": (0.80, 0.20),
        "sido_41": (0.50, 0.50),
    }
    for region, shares in distributions.items():
        for bloc, share in zip(("국민의힘", "더불어민주당"), shares):
            rows.append(
                {
                    "election_id": "old",
                    "election_type": "assembly_pr",
                    "region_id": region,
                    "bloc": bloc,
                    "vote_share": share,
                    "data_quality_weight": 1.0,
                }
            )
    events = identity.build_distinctiveness_events(pd.DataFrame(rows), date_resolver=_date)
    values = events.set_index("region_id")["distinctiveness"]
    assert np.isclose(values["sido_11"], 0.0)
    assert np.isclose(values["sido_26"], 0.30)
    assert np.isclose(values["sido_41"], 0.0)


def test_profiles_exclude_future_events_and_chungcheong() -> None:
    events = pd.DataFrame(
        [
            {
                "election_id": "old",
                "event_date": pd.Timestamp("2000-01-01"),
                "region_id": "sido_26",
                "distinctiveness": 0.20,
                "quality": 1.0,
                "type_weight": 1.0,
            },
            {
                "election_id": "future",
                "event_date": pd.Timestamp("2010-01-01"),
                "region_id": "sido_26",
                "distinctiveness": 0.50,
                "quality": 1.0,
                "type_weight": 1.0,
            },
            {
                "election_id": "old",
                "event_date": pd.Timestamp("2000-01-01"),
                "region_id": "sido_43",
                "distinctiveness": 0.40,
                "quality": 1.0,
                "type_weight": 1.0,
            },
        ]
    )
    profiles = identity.fit_distinctiveness_profiles(
        events, cutoff=pd.Timestamp("2005-01-01"), prior_strength=0.01
    )
    assert profiles["region_id"].tolist() == ["sido_26"]
    assert profiles.iloc[0]["distinctiveness"] == 0.20
    assert profiles.iloc[0]["events"] == 1


def test_routing_is_pit_safe_non_chung_and_mass_conserving(monkeypatch) -> None:
    monkeypatch.setattr(identity, "election_date", lambda _: pd.Timestamp("2005-01-01"))
    frame = pd.DataFrame(
        [
            {"election_id": "pres_test", "region_id": region, "candidate_name": name, "pred": pred}
            for region in ("sido_26", "sido_43")
            for name, pred in (("A", 0.45), ("B", 0.55))
        ]
    )
    events = pd.DataFrame(
        [
            {
                "election_id": "old",
                "event_date": pd.Timestamp("2000-01-01"),
                "region_id": region,
                "distinctiveness": 0.30,
                "quality": 1.0,
                "type_weight": 1.0,
            }
            for region in ("sido_26", "sido_43")
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "region_id": "sido_26",
                "candidate_name": "A",
                "regional_affinity": 1.0,
                "organization_depth": 1.0,
                "confidence": 1.0,
                "available_date": "2004-12-31",
            },
            {
                "election_id": "pres_test",
                "region_id": "sido_26",
                "candidate_name": "B",
                "regional_affinity": 1.0,
                "organization_depth": 1.0,
                "confidence": 1.0,
                "available_date": "2006-01-01",
            },
        ]
    )
    out, _ = identity.apply_regional_identity_routing(
        frame,
        events,
        evidence,
        prediction_column="pred",
        gain=1.0,
        shift_cap=0.04,
        prior_strength=0.01,
    )
    busan = out.loc[out["region_id"].eq("sido_26")]
    chung = out.loc[out["region_id"].eq("sido_43")]
    assert np.isclose(busan["pred"].sum(), 1.0)
    assert busan.loc[busan["candidate_name"].eq("A"), "pred"].iloc[0] > 0.45
    assert chung["pred"].tolist() == [0.45, 0.55]


def test_regional_mismatch_selects_donor_before_compatible_candidate(monkeypatch) -> None:
    monkeypatch.setattr(identity, "election_date", lambda _: pd.Timestamp("2005-01-01"))
    frame = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "region_id": "sido_29",
                "candidate_name": "L",
                "candidate_camp": "camp_liberal",
                "regional_accent_liberal_share": 0.70,
                "regional_accent_conservative_share": 0.05,
                "regional_accent_centrist_share": 0.10,
                "pred": 0.60,
            },
            {
                "election_id": "pres_test",
                "region_id": "sido_29",
                "candidate_name": "R",
                "candidate_camp": "camp_conservative",
                "regional_accent_liberal_share": 0.70,
                "regional_accent_conservative_share": 0.05,
                "regional_accent_centrist_share": 0.10,
                "pred": 0.10,
            },
            {
                "election_id": "pres_test",
                "region_id": "sido_29",
                "candidate_name": "C",
                "candidate_camp": "camp_centrist",
                "regional_accent_liberal_share": 0.70,
                "regional_accent_conservative_share": 0.05,
                "regional_accent_centrist_share": 0.10,
                "pred": 0.30,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "election_id": "old",
                "event_date": pd.Timestamp("2000-01-01"),
                "region_id": "sido_29",
                "distinctiveness": 0.30,
                "quality": 1.0,
                "type_weight": 1.0,
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "region_id": "sido_29",
                "candidate_name": "C",
                "regional_affinity": 1.0,
                "organization_depth": 1.0,
                "confidence": 1.0,
                "available_date": "2004-12-31",
            }
        ]
    )
    out, _ = identity.apply_regional_identity_routing(
        frame,
        events,
        evidence,
        prediction_column="pred",
        gain=1.0,
        shift_cap=0.04,
        prior_strength=0.01,
    )
    values = out.set_index("candidate_name")["pred"]
    assert np.isclose(values["L"], 0.60)
    assert values["R"] < 0.10
    assert values["C"] > 0.30
