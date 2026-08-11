"""Naver DataLab salience collector — offline (HTTP injected)."""

from __future__ import annotations

import pandas as pd

from news_collector.sources.datalab import (
    apply_salience_to_events,
    build_keyword_groups,
    collect_issue_salience,
    normalize_salience,
    rescale_by_anchor,
)


def test_build_keyword_groups_batches_with_anchor() -> None:
    km = {f"issue_{i}": ["kw"] for i in range(6)}  # 6 issues, batch 4 + anchor
    reqs = build_keyword_groups(km, anchor_name="선거", batch_size=4)
    assert len(reqs) == 2
    assert all(req[-1]["groupName"] == "선거" for req in reqs)  # anchor last in each
    assert len(reqs[0]) == 5 and len(reqs[1]) == 3  # 4+anchor, then 2+anchor


def test_rescale_by_anchor_makes_anchor_mean_one() -> None:
    series = {
        "housing": [("2022-02-01", 20.0), ("2022-02-08", 40.0)],
        "선거": [("2022-02-01", 10.0), ("2022-02-08", 30.0)],  # mean=20
    }
    out = rescale_by_anchor(series, "선거")
    assert "선거" not in out
    assert out["housing"] == [("2022-02-01", 1.0), ("2022-02-08", 2.0)]  # /20


def test_normalize_salience_minmax() -> None:
    rescaled = {"a": [("w1", 1.0), ("w2", 2.0)], "b": [("w1", 4.0)]}
    df = normalize_salience(rescaled, "pres_2022")
    assert df["salience_score"].max() == 1.0  # peak (b/w1=4) → 1.0
    assert df.loc[df["issue_name"] == "a", "salience_score"].tolist() == [0.25, 0.5]


def test_collect_issue_salience_with_injected_poster() -> None:
    def fake_poster(body, cid, secret):
        # echo each group with a flat ratio = len(groupName) so it's deterministic
        out = []
        for g in body["keywordGroups"]:
            ratio = 50.0 if g["groupName"] == "선거" else 25.0
            out.append({"title": g["groupName"], "data": [{"period": "2022-02-07", "ratio": ratio}]})
        return out

    km = {"housing": ["부동산"], "security_nk": ["대북"]}
    df = collect_issue_salience(
        km, "2022-02-01", "2022-02-28", "pres_2022",
        client_id="x", client_secret="y", poster=fake_poster,
    )
    assert set(df["issue_name"]) == {"housing", "security_nk"}
    assert df["salience_score"].between(0, 1).all()


def test_apply_salience_to_events_overwrites_by_week() -> None:
    events = pd.DataFrame(
        [{"election_id": "pres_2022", "issue_name": "housing", "available_date": "2022-02-22",
          "slot": "A", "salience_score": 0.0}]
    )
    salience = pd.DataFrame(
        [{"election_id": "pres_2022", "issue_name": "housing", "period": "2022-02-20", "salience_score": 0.8}]
    )
    out = apply_salience_to_events(events, salience)
    assert out.loc[0, "salience_score"] == 0.8  # curated slot kept, salience filled
    assert out.loc[0, "slot"] == "A"
