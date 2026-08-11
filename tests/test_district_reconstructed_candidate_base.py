from __future__ import annotations

import pandas as pd

from presidential_issue_engine.district_reconstructed_candidate_base import (
    build_district_reconstructed_candidate_base,
)


def _district_rows() -> pd.DataFrame:
    rows = []
    for region_id, region_name, district, major_share, third_share in [
        ("sido_11", "서울특별시", "갑", 0.60, 0.20),
        ("sido_11", "서울특별시", "을", 0.20, 0.10),
        ("sido_29", "광주광역시", "갑", 0.20, 0.65),
        ("sido_29", "광주광역시", "을", 0.25, 0.60),
    ]:
        for bloc, candidate, share in [
            ("더불어민주당", f"major-{district}-{region_id}", major_share),
            ("제3지대", f"third-{district}-{region_id}", third_share),
            ("국민의힘", f"other-{district}-{region_id}", 1.0 - major_share - third_share),
        ]:
            rows.append(
                {
                    "election_id": "assembly_2016_district",
                    "election_date": "2016-04-13",
                    "region_id": region_id,
                    "region_name": region_name,
                    "district_name": district,
                    "party_name": bloc,
                    "bloc": bloc,
                    "candidate_name": candidate,
                    "candidate_votes": 100 * share,
                    "district_valid_votes": 100,
                    "candidate_vote_share": share,
                    "candidate_won": share == max(major_share, third_share, 1.0 - major_share - third_share),
                    "available_date": "2016-04-14",
                }
            )
    # Make the target major candidate the strong candidate in Seoul district 갑.
    rows[0]["candidate_name"] = "후보A"
    return pd.DataFrame(rows)


def _candidate_history() -> pd.DataFrame:
    common = {
        "target_election_id": "pres_2017",
        "target_election_date": "2017-05-09",
        "source_election_date": "2016-04-13",
        "source_sg_id": "20160413",
        "source_election_name": "국회의원선거",
        "source_sg_typecode": "2",
        "source_candidate_id": "id",
        "source_municipality_name": "",
        "source_status": "등록",
        "source_job": "정치인",
        "source_career1": "",
        "source_career2": "",
        "source_is_prior": True,
        "prior_election_won": "Y",
        "available_date": "2016-04-14",
        "entity_match_confidence": 1.0,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "target_slot": "A",
                "target_candidate_name": "후보A",
                "target_party_name": "더불어민주당",
                "source_region_name": "서울특별시",
                "source_district_name": "갑",
                "source_party_name": "더불어민주당",
            },
            {
                **common,
                "target_slot": "C",
                "target_candidate_name": "후보C",
                "target_party_name": "국민의당",
                "source_region_name": "서울특별시",
                "source_district_name": "없는구",
                "source_party_name": "국민의당",
            },
        ]
    )


def test_personal_excess_and_nonmajor_party_organization_are_separate() -> None:
    context = pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "A",
                "candidate_name": "후보A",
                "bloc": "더불어민주당",
                "organization_strength": 0.9,
                "available_date": "2017-05-08",
                "confidence": 0.8,
            },
            {
                "election_id": "pres_2017",
                "slot": "C",
                "candidate_name": "후보C",
                "bloc": "제3지대",
                "organization_strength": 0.8,
                "available_date": "2017-05-08",
                "confidence": 0.8,
            },
            {
                "election_id": "pres_2017",
                "slot": "C",
                "candidate_name": "후보C",
                "bloc": "제3지대",
                "organization_strength": 1.0,
                "available_date": "2017-05-10",
                "confidence": 1.0,
            },
        ]
    )
    regional, _ = build_district_reconstructed_candidate_base(
        _candidate_history(), _district_rows(), context
    )
    personal = regional.loc[
        regional["candidate_name"].eq("후보A")
        & regional["region_id"].eq("sido_11")
    ].iloc[0]
    assert personal["personal_constituency_signal"] > 0.0
    assert personal["party_district_organization_signal"] == 0.0

    third = regional.loc[regional["candidate_name"].eq("후보C")]
    gwangju = third.loc[third["region_id"].eq("sido_29")].iloc[0]
    seoul_rows = third.loc[third["region_id"].eq("sido_11")]
    seoul_signal = (
        0.0
        if seoul_rows.empty
        else float(seoul_rows.iloc[0]["party_district_organization_signal"])
    )
    assert gwangju["party_district_organization_signal"] > seoul_signal
    # The post-election context row must not replace the dated pre-election row.
    assert gwangju["confidence"] == 0.8


def test_footprint_control_limits_single_constituency_province_spillover() -> None:
    context = pd.DataFrame(
        [
            {
                "election_id": "pres_2017",
                "slot": "A",
                "candidate_name": "후보A",
                "bloc": "더불어민주당",
                "organization_strength": 0.9,
                "available_date": "2017-05-08",
                "confidence": 0.8,
            }
        ]
    )
    legacy, _ = build_district_reconstructed_candidate_base(
        _candidate_history(), _district_rows(), context
    )
    controlled, components = build_district_reconstructed_candidate_base(
        _candidate_history(),
        _district_rows(),
        context,
        footprint_controlled=True,
    )
    legacy_signal = float(
        legacy.loc[legacy["candidate_name"].eq("후보A"), "personal_constituency_signal"].iloc[0]
    )
    controlled_signal = float(
        controlled.loc[
            controlled["candidate_name"].eq("후보A"),
            "personal_constituency_signal",
        ].iloc[0]
    )

    assert 0.0 < controlled_signal < legacy_signal
    personal = components.loc[components["component"].eq("personal_constituency")]
    assert personal["scope_weight"].between(0.0, 1.0, inclusive="neither").all()
    assert (personal["signal"] < personal["raw_signal"]).all()
