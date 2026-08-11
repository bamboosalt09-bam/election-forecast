"""Offline tests for VoteXmntckInfoInqireService2 parsing."""

from __future__ import annotations

import httpx

from news_collector.sources.nec_vote_api import call_api, count_item_candidate_rows, parse_items
from presidential_issue_engine.import_nec_vote_api_history import _items_to_history


def test_parse_xml_items_and_total_count() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <response>
      <header><resultCode>INFO-00</resultCode><resultMsg>NORMAL SERVICE</resultMsg></header>
      <body>
        <items>
          <item>
            <sgId>20200415</sgId>
            <sgTypecode>7</sgTypecode>
            <sdName>서울특별시</sdName>
            <jd01>미래한국당</jd01>
            <hbj01></hbj01>
            <dugsu01>1,000</dugsu01>
          </item>
        </items>
        <totalCount>1</totalCount>
      </body>
    </response>
    """
    items, total = parse_items(xml)

    assert total == 1
    assert items[0]["sdName"] == "서울특별시"
    assert items[0]["dugsu01"] == "1,000"


def test_count_item_candidate_rows_explodes_numbered_fields() -> None:
    rows = count_item_candidate_rows(
        {
            "sgId": "20200415",
            "sgTypecode": "7",
            "sdName": "서울특별시",
            "jd01": "미래한국당",
            "hbj01": "",
            "dugsu01": "1,000",
            "jd02": "더불어시민당",
            "hbj02": "",
            "dugsu02": "2,000",
        }
    )

    assert [row["party"] for row in rows] == ["미래한국당", "더불어시민당"]
    assert [row["votes"] for row in rows] == [1000.0, 2000.0]


def test_count_item_candidate_rows_ignores_zero_filled_empty_slots() -> None:
    rows = count_item_candidate_rows(
        {
            "jd01": "정당A",
            "hbj01": "후보A",
            "dugsu01": "100",
            "jd02": "",
            "hbj02": "",
            "dugsu02": "0",
        },
        max_candidates=2,
    )

    assert len(rows) == 1
    assert rows[0]["candidate"] == "후보A"


def test_items_to_history_normalizes_bloc_share_by_region() -> None:
    history = _items_to_history(
        [
            {
                "sgId": "20200415",
                "sgTypecode": "7",
                "sdName": "서울특별시",
                "wiwName": "합계",
                "jd01": "미래한국당",
                "dugsu01": "1,000",
                "jd02": "더불어시민당",
                "dugsu02": "2,000",
            }
        ],
        election_id="assembly_2020_pr",
        election_type="assembly_pr",
        data_quality_weight=1.0,
    )

    out = history.set_index("bloc")["vote_share"].to_dict()
    assert round(out["국민의힘"], 6) == round(1 / 3, 6)
    assert round(out["더불어민주당"], 6) == round(2 / 3, 6)


def test_call_api_http_error_does_not_echo_service_key(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> httpx.Response:
        request = httpx.Request("GET", url)
        return httpx.Response(401, text="Unauthorized", request=request)

    monkeypatch.setattr(httpx, "get", fake_get)

    try:
        call_api("operation", service_key="secret-key")
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")

    assert "401" in message
    assert "secret-key" not in message
