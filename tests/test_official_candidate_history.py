from __future__ import annotations

import pandas as pd

from presidential_issue_engine.official_candidate_history import (
    build_candidate_reference,
    build_official_candidate_regional_base,
    resolve_candidate_history,
)


def _record(
    *,
    name: str,
    birthday: str,
    sg_id: str,
    party: str,
    typecode: str,
    region: str,
    won: str,
) -> dict[str, str]:
    return {
        "name": name,
        "birthday": birthday,
        "sgId": sg_id,
        "jdName": party,
        "sgTypecode": typecode,
        "sdName": region,
        "sggName": "테스트구",
        "wiwName": "",
        "huboid": f"id-{sg_id}",
        "elctNm": "국회의원선거" if typecode == "2" else "대통령선거",
        "status": "등록",
        "job": "정치인",
        "career1": "경력",
        "career2": "",
        "elcoYn": won,
    }


def test_reference_reads_identity_columns_only() -> None:
    results = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "slot": "A",
                "candidate_name": "후보",
                "party_name": "정당",
                "votes": 123,
                "vote_share": 0.9,
            },
            {
                "election_id": "pres_2022",
                "slot": "alpha",
                "candidate_name": "기타후보 합산",
                "party_name": "",
                "votes": 1,
                "vote_share": 0.1,
            },
        ]
    )
    out = build_candidate_reference(results)
    assert list(out.columns) == [
        "election_id",
        "target_election_date",
        "slot",
        "candidate_name",
        "party_name",
    ]
    assert out.iloc[0]["target_election_date"] == "2022-03-09"
    assert out["candidate_name"].tolist() == ["후보"]


def test_entity_resolution_masks_target_outcome_and_excludes_post_cutoff() -> None:
    reference = pd.DataFrame(
        [
            {
                "election_id": "pres_2022",
                "target_election_date": "2022-03-09",
                "slot": "A",
                "candidate_name": "홍길동",
                "party_name": "정당A",
            }
        ]
    )
    records = {
        "홍길동": [
            _record(
                name="홍길동",
                birthday="19600101",
                sg_id="20160413",
                party="정당A",
                typecode="2",
                region="부산광역시",
                won="Y",
            ),
            _record(
                name="홍길동",
                birthday="19600101",
                sg_id="20220309",
                party="정당A",
                typecode="1",
                region="전국",
                won="Y",
            ),
            _record(
                name="홍길동",
                birthday="19600101",
                sg_id="20250603",
                party="정당B",
                typecode="1",
                region="전국",
                won="N",
            ),
            _record(
                name="홍길동",
                birthday="19700101",
                sg_id="20180613",
                party="다른정당",
                typecode="5",
                region="서울특별시",
                won="N",
            ),
        ]
    }
    history, audit = resolve_candidate_history(
        reference,
        records,
        source_url="https://example.test",
        max_source_date="2022-12-31",
    )
    assert audit.iloc[0]["entity_match_method"] == "target_election_party_birthday"
    assert set(history["birthday"]) == {"19600101"}
    assert "20250603" not in set(history["source_sg_id"])
    target = history.loc[history["source_sg_id"].eq("20220309")].iloc[0]
    assert not bool(target["source_is_prior"])
    assert target["prior_election_won"] == ""


def test_regional_base_uses_only_prior_dated_regional_records() -> None:
    history = pd.DataFrame(
        [
            {
                "target_election_id": "pres_2022",
                "target_election_date": "2022-03-09",
                "target_slot": "A",
                "target_candidate_name": "후보",
                "source_election_date": "2016-04-13",
                "source_election_name": "국회의원선거",
                "source_sg_typecode": "2",
                "source_sg_id": "20160413",
                "source_region_name": "부산광역시",
                "source_is_prior": True,
                "prior_election_won": "Y",
                "entity_match_confidence": 1.0,
                "available_date": "2016-04-14",
            },
            {
                "target_election_id": "pres_2022",
                "target_election_date": "2022-03-09",
                "target_slot": "A",
                "target_candidate_name": "후보",
                "source_election_date": "2022-03-09",
                "source_election_name": "대통령선거",
                "source_sg_typecode": "1",
                "source_sg_id": "20220309",
                "source_region_name": "전국",
                "source_is_prior": False,
                "prior_election_won": "",
                "entity_match_confidence": 1.0,
                "available_date": "2022-03-09",
            },
        ]
    )
    out = build_official_candidate_regional_base(history)
    assert len(out) == 1
    assert out.iloc[0]["region_id"] == "sido_26"
    assert 0.0 < out.iloc[0]["regional_affinity"] <= 0.8
    assert out.iloc[0]["source_election_ids"] == "20160413"

