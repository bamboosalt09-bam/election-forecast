from election_forecast.stance_context_model import compose_context_input, weak_context_label


def test_context_input_masks_target_identity_and_keeps_boundaries() -> None:
    value = compose_context_input(
        {
            "target_type": "person",
            "target_name": "홍길동",
            "target_alias": "길동 후보",
            "context_before": "홍길동 관련 질의입니다.",
            "text_excerpt": "길동 후보의 정책을 비판합니다.",
            "context_after": "홍길동에게 답변을 요구합니다.",
        }
    )
    assert "홍길동" not in value
    assert "길동 후보" not in value
    assert "[TARGET]" in value
    assert "[CURRENT]" in value
    assert "[BEFORE]" in value
    assert "[AFTER]" in value


def test_weak_context_label_separates_reform_topic_from_attack() -> None:
    label = weak_context_label(
        {
            "text_excerpt": "부정부패 사범을 엄단하고 비리를 근절하는 대책을 시행했습니다.",
            "rule_stance_polarity": -1,
        }
    )
    assert label.label == "neutral"
    assert label.reason == "anti_corruption_policy_not_attack"


def test_weak_context_label_detects_defense_inside_critical_question() -> None:
    label = weak_context_label(
        {
            "text_excerpt": "대응하지 못했는데도 책임이 없다고 하는 것이 맞는 것 아니겠습니까?",
            "rule_stance_polarity": 1,
        }
    )
    assert label.label == "negative"
    assert label.reason == "defense_inside_rebuttal_or_question"

