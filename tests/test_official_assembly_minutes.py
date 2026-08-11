from datetime import date

from presidential_issue_engine.official_assembly_minutes import (
    conservative_available_date,
    parse_committees,
    parse_meetings,
    parse_pdf_creation_datetime,
    parse_session_windows,
    parse_speaker_blocks,
)


def test_parse_hierarchy_and_meeting() -> None:
    main = """
    <a data-class="1" data-sess="424">
      제424회 (2025. 04. 04. ~ 2025. 05. 03.)
    </a>
    """
    assert parse_session_windows(main) == {
        "424": (date(2025, 4, 4), date(2025, 5, 3))
    }

    committees = parse_committees(
        '<a data-class="2" data-cmit="AA">국회운영위원회</a>', "2"
    )
    assert committees[0].committee_code == "AA"
    meetings = parse_meetings(
        '<a data-id="54513" data-sess="424" '
        'title="국회운영위원회 제1차 (2025. 04. 15.)"></a>',
        class_id="2",
        committee_code="AA",
        committee_name="국회운영위원회",
    )
    assert meetings[0].minutes_id == "54513"
    assert meetings[0].meeting_date == date(2025, 4, 15)


def test_pdf_creation_date_controls_availability() -> None:
    pdf = b"/CreationDate (D:20250610135136+09'00')"
    assert parse_pdf_creation_datetime(pdf).isoformat() == "2025-06-10T13:51:36"
    available, basis, created = conservative_available_date(
        pdf, collected_on=date(2026, 8, 10), safety_lag_days=1
    )
    assert available == date(2025, 6, 11)
    assert basis == "official_pdf_creation_plus_safety_lag"
    assert created == "2025-06-10T13:51:36"


def test_missing_pdf_date_fails_closed() -> None:
    available, basis, created = conservative_available_date(
        b"%PDF-no-metadata", collected_on=date(2026, 8, 10)
    )
    assert available == date(2026, 8, 10)
    assert basis == "collection_date_fallback_not_historical"
    assert created == ""


def test_speaker_parser_uses_only_attributed_utterances() -> None:
    html = """
    <h2>제22대국회 제424회 제1차 국회운영위원회 (2025.04.15.)</h2>
    <div class="speaker" data-mem_id="6693" data-name="박찬대" data-pos="위원장">
      <label>위원장 박찬대 선택</label>
      <span class="spk_sub">첫 번째 발언입니다.</span>
      <span class="spk_sub">두 번째 발언입니다.</span>
    </div>
    <div>회의록 외부 안내문</div>
    """
    blocks = parse_speaker_blocks(html)
    assert len(blocks) == 1
    assert blocks[0].speaker_name == "박찬대"
    assert blocks[0].text == "첫 번째 발언입니다. 두 번째 발언입니다."
    assert "선택" not in blocks[0].text
