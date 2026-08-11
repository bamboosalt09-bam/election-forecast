from presidential_issue_engine.region_bloc_prior import normalize_bloc
from scripts.build_assembly_stance_pilot import quotas_for
from scripts.evaluate_stance_pilot_3000_sensitivity import (
    _speaker_bloc_lookup,
    bounded_log_strength,
    meeting_quarter,
    neutral_content_flags,
    normalize_speaker_name,
)


def test_validation5000_profile_has_exact_requested_size() -> None:
    quotas = quotas_for("validation5000", 5_000)
    assert sum(quotas.values()) == 5_000
    assert quotas["person_directional"] == 500
    assert quotas["party_directional"] == 500


def test_neutral_context_helpers_extract_analysis_without_vote_direction() -> None:
    assert neutral_content_flags("이 문제의 원인과 영향을 분석하고 대책이 필요합니다") == (1, 1)
    assert neutral_content_flags("회의를 시작하겠습니다") == (0, 0)
    assert meeting_quarter("2021-08-17") == "2021-Q3"
    assert meeting_quarter("unknown") == ""
    assert bounded_log_strength(0, 100) == 0.0
    assert bounded_log_strength(100, 100) == 1.0


def test_normalize_speaker_name_removes_common_assembly_roles() -> None:
    assert normalize_speaker_name("조수진 위원") == "조수진"
    assert normalize_speaker_name("안상수 의원") == "안상수"
    assert normalize_speaker_name("보건복지부장관 김근태") == "김근태"


def test_speaker_bloc_lookup_spans_historical_assemblies() -> None:
    lookup = _speaker_bloc_lookup()
    assert lookup[("17", "안상수")] == normalize_bloc("한나라당")
    assert lookup[("21", "고민정")] == normalize_bloc("더불어민주당")
