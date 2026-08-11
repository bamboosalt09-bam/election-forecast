import numpy as np

from scripts.evaluate_external_nli_cascade import (
    CascadePolicy,
    apply_cascade,
    compose_hypotheses,
    compose_premise,
)
from scripts.evaluate_external_nli_role_veto import VetoPolicy, apply_role_veto


def test_external_nli_prompts_are_target_specific() -> None:
    row = {
        "target_name": "정부",
        "text_excerpt": "정부 정책의 책임을 묻습니다.",
        "context_before": "경제 문제를 논의했습니다.",
        "context_after": "답변을 요구합니다.",
        "agenda": "경제정책",
    }

    premise = compose_premise(row)
    hypotheses = compose_hypotheses(row)

    assert "[현재 발언]" in premise
    assert "[앞 문맥]" in premise
    assert len(premise) < 2_500
    assert len(hypotheses) == 6
    assert "정부" in hypotheses[0]
    assert "정부" in hypotheses[3]


def test_cascade_abstains_when_any_semantic_gate_is_weak() -> None:
    stance = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
        ]
    )
    policy = CascadePolicy(0.70, 0.70, 0.70, 0.20)

    prediction = apply_cascade(
        [0.90, 0.60, 0.90, 0.90],
        [0.90, 0.90, 0.60, 0.90],
        np.vstack([stance[:3], [[0.45, 0.10, 0.45]]]),
        policy,
    )

    assert prediction.tolist() == ["negative", "neutral", "neutral", "neutral"]


def test_external_role_veto_never_creates_or_reverses_direction() -> None:
    source = np.asarray(["negative", "positive", "neutral"])
    prediction = apply_role_veto(
        source,
        np.asarray([0.9, 0.4, 0.9]),
        np.asarray([0.9, 0.9, 0.9]),
        VetoPolicy(0.7, 0.7),
    )

    assert prediction.tolist() == ["negative", "neutral", "neutral"]
