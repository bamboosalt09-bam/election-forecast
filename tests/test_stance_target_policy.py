from election_forecast.stance_target_policy import target_aware_decision


def _row(text: str, *, target_type: str = "person", target_name: str = "박근혜", legacy: str = "attack") -> dict[str, object]:
    return {
        "text_excerpt": text,
        "target_type": target_type,
        "target_name": target_name,
        "rule_stance_label": legacy,
    }


def test_metalinguistic_candidate_examples_are_neutral() -> None:
    decision = target_aware_decision(
        _row("예컨대 박근혜를 지지한다, 문재인을 지지한다, 그리고 비방한다", legacy="endorse"),
        model_label="positive",
        model_probability=0.9,
        model_margin=0.8,
    )
    assert decision.label == "neutral"


def test_negated_criticism_reception_is_neutral() -> None:
    decision = target_aware_decision(
        _row("우리가 꼭 문재인 정부를 비판한다고만 받아들이지 마시고 논의해야 한다", target_name="문재인"),
        model_label="negative",
        model_probability=0.9,
        model_margin=0.8,
    )
    assert decision.label == "neutral"


def test_reported_demand_without_evaluation_is_neutral() -> None:
    decision = target_aware_decision(
        _row(
            "안상수 한나라당 원내대표께서 간사 사퇴 요구를 하셨다고 이야기를 들었습니다.",
            target_type="party",
            target_name="한나라당",
            legacy="neutral",
        ),
        model_label="negative",
        model_probability=0.9,
        model_margin=0.8,
    )
    assert decision.label == "neutral"


def test_alignment_with_target_proposal_is_positive() -> None:
    decision = target_aware_decision(
        _row(
            "우리 한나라당과 총재가 주장한 것처럼 부정부패 추방을 위한 제도개혁을 해야 합니다.",
            target_type="party",
            target_name="한나라당",
        ),
        model_label="negative",
        model_probability=0.4,
        model_margin=0.01,
    )
    assert decision.label == "positive"


def test_high_confidence_override_requires_explicit_opt_in() -> None:
    row = _row("박근혜 정부의 정책 실패가 명백합니다.", legacy="neutral")
    conservative = target_aware_decision(
        row,
        model_label="negative",
        model_probability=0.9,
        model_margin=0.8,
    )
    override = target_aware_decision(
        row,
        model_label="negative",
        model_probability=0.9,
        model_margin=0.8,
        allow_high_confidence_model_override=True,
    )
    assert conservative.label == "neutral"
    assert override.label == "negative"
    assert override.used_model_override
