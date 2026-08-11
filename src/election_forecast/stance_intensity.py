"""Derive continuous stance strength and neutral-information labels.

The three-way context classifier estimates direction.  It was not trained on
linguistic intensity or informativeness, so those values remain transparent
shadow features rather than gold labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


_ANALYSIS = re.compile(
    r"원인|영향|결과|전망|현황|대책|방안|대안|필요|때문|따라서|분석|평가|통계|자료|조사"
)
_IMPACT = re.compile(
    r"문제|위기|심각|우려|의혹|논란|실패|악화|부족|침체|부담|불안|피해|위법|부패|"
    r"폭등|폭락|개선|회복|증가|감소|성장|강화|확대|축소|성과|안정|해결|정상화"
)
_EVIDENCE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|퍼센트|원|억|조|명|건|배|년|개월)|"
    r"보고서|통계청|한국은행|감사원|위원회|연구원|자료에 따르면"
)
_PROCEDURAL = re.compile(
    r"의사일정|개의하겠습니다|산회를 선포|상정합니다|가결되었음을|회의록|"
    r"질의하시기 바랍니다|답변하여 주시기 바랍니다|다음 안건"
)
_EMPHASIS = re.compile(
    r"매우|대단히|극히|결코|절대로|명백히|엄중|중대|참담|최악|강력히|"
    r"전적으로|반드시|도저히|심각|철저히|단호히"
)


@dataclass(frozen=True)
class StanceIntensity:
    positive_strength: float
    negative_strength: float
    positive_label: str
    negative_label: str
    directional_score: float
    directional_strength: float
    emphasis_score: float


@dataclass(frozen=True)
class NeutralInformation:
    content_score: float
    neutral_information_score: float
    label: str
    analysis_flag: int
    impact_flag: int
    evidence_flag: int
    procedural_flag: int


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def strength_label(value: float) -> str:
    """Map a directional evidence score to a stable descriptive bin."""
    if value < 0.08:
        return "absent"
    if value < 0.20:
        return "weak"
    if value < 0.40:
        return "moderate"
    return "strong"


def information_label(value: float) -> str:
    if value < 0.08:
        return "none"
    if value < 0.20:
        return "low"
    if value < 0.40:
        return "medium"
    return "high"


def stance_intensity(
    probability_negative: float,
    probability_neutral: float,
    probability_positive: float,
    text: object,
) -> StanceIntensity:
    """Convert class posteriors into mutually exclusive directional strength.

    The positive-minus-negative posterior contrast prevents simultaneous
    positive and negative mass from being counted twice.  Emphasis can change
    magnitude by at most 20%; it cannot create a direction.
    """
    probabilities = np.asarray(
        [probability_negative, probability_neutral, probability_positive], dtype=float
    )
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        probabilities[:] = (0.0, 1.0, 0.0)
    else:
        probabilities /= total
    p_negative, _, p_positive = probabilities
    directional_score = float(p_positive - p_negative)
    text_value = str(text or "")
    emphasis_count = len(_EMPHASIS.findall(text_value)) + min(text_value.count("!"), 2)
    emphasis_score = _clip(emphasis_count / 3.0)
    directional_strength = _clip(abs(directional_score) * (1.0 + 0.20 * emphasis_score))
    positive_strength = directional_strength if directional_score > 0.0 else 0.0
    negative_strength = directional_strength if directional_score < 0.0 else 0.0
    return StanceIntensity(
        positive_strength=positive_strength,
        negative_strength=negative_strength,
        positive_label=strength_label(positive_strength),
        negative_label=strength_label(negative_strength),
        directional_score=float(np.sign(directional_score) * directional_strength),
        directional_strength=directional_strength,
        emphasis_score=emphasis_score,
    )


def neutral_information(
    probability_neutral: float,
    text: object,
    *,
    issue_name: object = "",
    context_before: object = "",
    context_after: object = "",
) -> NeutralInformation:
    """Estimate substantive information carried by a neutral statement.

    This score never supplies vote direction.  It may only support an already
    observed directional signal for the same election and issue.
    """
    text_value = str(text or "").strip()
    analysis_flag = int(bool(_ANALYSIS.search(text_value)))
    impact_flag = int(bool(_IMPACT.search(text_value)))
    evidence_flag = int(bool(_EVIDENCE.search(text_value)))
    procedural_flag = int(bool(_PROCEDURAL.search(text_value)))
    length_score = _clip((len(text_value) - 15.0) / 140.0)
    issue_flag = int(bool(str(issue_name or "").strip()))
    context_flag = int(bool(str(context_before or "").strip() or str(context_after or "").strip()))
    content_score = (
        0.30 * length_score
        + 0.22 * analysis_flag
        + 0.18 * impact_flag
        + 0.18 * evidence_flag
        + 0.08 * issue_flag
        + 0.04 * context_flag
    )
    if procedural_flag and not (analysis_flag or impact_flag or evidence_flag):
        content_score *= 0.30
    content_score = _clip(content_score)
    neutral_score = _clip(float(probability_neutral) * content_score)
    return NeutralInformation(
        content_score=content_score,
        neutral_information_score=neutral_score,
        label=information_label(neutral_score),
        analysis_flag=analysis_flag,
        impact_flag=impact_flag,
        evidence_flag=evidence_flag,
        procedural_flag=procedural_flag,
    )
