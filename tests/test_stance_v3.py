import pytest

from election_forecast.stance_v3 import (
    classify_issue_character,
    compose_v3_input,
    ownership_abstention,
)


def test_nearest_context_uses_only_closest_side() -> None:
    row = {
        "target_type": "person",
        "target_name": "홍길동",
        "text_excerpt": "홍길동의 정책을 묻습니다.",
        "context_before": "먼 앞 문장",
        "context_after": "가까운 뒤 문장",
        "context_gap_before": 4,
        "context_gap_after": 1,
    }
    result = compose_v3_input(row, "nearest_context")
    assert "가까운 뒤 문장" in result
    assert "먼 앞 문장" not in result
    assert "홍길동" not in result


def test_risk_aware_representation_drops_context_for_reported_speech() -> None:
    row = {
        "target_type": "party",
        "text_excerpt": "누군가 지지한다고 말했습니다.",
        "context_before": "별도 주장",
        "context_gap_before": 1,
    }
    result = compose_v3_input(row, "risk_aware_nearest")
    assert "별도 주장" not in result


def test_ownership_abstains_on_metalinguistic_example() -> None:
    label, reason = ownership_abstention(
        "예컨대 박근혜를 지지한다, 문재인을 지지한다고 표현한다.", "positive"
    )
    assert label == "neutral"
    assert reason == "metalinguistic_example"


def test_ownership_retains_direct_support() -> None:
    label, reason = ownership_abstention("저는 이 정책을 적극 지지합니다.", "positive")
    assert label == "positive"
    assert reason == "retained"


def test_issue_character_distinguishes_information_from_evaluation() -> None:
    informational = classify_issue_character(0.10, 0.80, 0.10, confidence_quality=0.75)
    negative = classify_issue_character(0.70, 0.20, 0.10, confidence_quality=0.75)

    assert informational["issue_character"] == "informational_context"
    assert informational["informational_score"] == 0.8
    assert informational["character_score"] < 0.0
    assert informational["character_multiplier"] < 1.0
    assert negative["issue_character"] == "negative_accountability"
    assert negative["accountability_score"] == pytest.approx(0.6)
    assert negative["character_score"] > 0.0
    assert negative["character_multiplier"] > 1.0


def test_issue_character_marks_two_sided_directional_discourse_as_polarized() -> None:
    result = classify_issue_character(0.40, 0.20, 0.40, confidence_quality=0.75)

    assert result["issue_character"] == "polarized_contest"
    assert result["polarized_score"] == 0.8
    assert result["polarization"] == 0.8
    assert result["character_multiplier"] > 1.0
