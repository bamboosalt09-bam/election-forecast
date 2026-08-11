import numpy as np

from election_forecast.stance_text_model import (
    apply_rule_correction,
    legacy_rule_label,
)
from scripts.extract_assembly_stance_rows import classify_stance


def test_legacy_text_label_matches_extractor_polarity() -> None:
    texts = [
        "이 정책을 강력히 비판한다.",
        "경제 문제를 논의하겠습니다.",
        "이 정책을 적극 지지합니다.",
        "그 주장은 사실이 아니라고 반박한다.",
        "문제가 없다고 주장했지만 정책 실패를 비판한다.",
    ]
    polarity_labels = {-1: "negative", 0: "neutral", 1: "positive"}
    for text in texts:
        assert legacy_rule_label(text) == polarity_labels[classify_stance(text)[1]]


def test_rule_correction_does_not_invent_direction_from_neutral_by_default() -> None:
    classes = np.asarray(["negative", "neutral", "positive"])
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
        ]
    )
    corrected = apply_rule_correction(
        probabilities,
        classes,
        ["neutral", "positive", "negative"],
        min_override_probability=0.40,
        min_probability_margin=0.05,
    )
    assert corrected.tolist() == ["neutral", "negative", "neutral"]

