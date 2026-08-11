"""Conservative target-aware stance resolution for person and party mentions."""

from __future__ import annotations

import re
from dataclasses import dataclass


_SPACE = re.compile(r"\s+")
_METALINGUISTIC_EXAMPLE = re.compile(
    r"(?:예컨대|예를\s*들어|가령).{0,160}(?:지지|찬성|반대|비방|비판|공격)"
)
_NEGATED_RECEPTION = re.compile(
    r"(?:비판|비방|공격)(?:한다고|하는\s*것으로)(?:만)?\s*(?:받아들이지|보지|생각하지)"
)
_REPORT_ONLY = re.compile(
    r"(?:이야기|말|주장|요구|발언)(?:을|를)?\s*(?:들었습니다|들었다|했다고\s*합니다|"
    r"하였다고\s*합니다|전해졌습니다)|보도되었습니다|알려졌습니다"
)
_SPEAKER_EVALUATION = re.compile(
    r"부당|잘못|문제|타당|옳|납득할\s*수\s*없|비판|지지|찬성|반대|규탄|책임|"
    r"실패|불법|의혹|훌륭|성과|환영"
)
_ALIGNMENT = re.compile(
    r"(?:주장한\s*것처럼|제안한\s*대로|요구한\s*대로|방안에\s*따라|입장과\s*같이)"
)
_NORMATIVE_ACTION = re.compile(r"해야|필요|추진|개혁|개선|강화|확대|도입|실현")
_DIRECT_NEGATIVE = re.compile(
    r"정책\s*실패|실패|잘못|불법|비리|뇌물|무능|은폐|사퇴해야|퇴진해야|책임져야|규탄"
)
_DIRECT_POSITIVE = re.compile(
    r"적극\s*지지|지지합니다|찬성합니다|환영합니다|높이\s*평가|성과를\s*냈|훌륭"
)


@dataclass(frozen=True)
class TargetAwareDecision:
    label: str
    reason: str
    used_model_override: bool = False


def _clean(value: object) -> str:
    return _SPACE.sub(" ", "" if value is None else str(value)).strip()


def generic_legacy_label(value: object) -> str:
    label = _clean(value)
    if label == "attack":
        return "negative"
    if label in {"defend", "endorse"}:
        return "positive"
    return "neutral"


def target_aware_decision(
    row: dict[str, object],
    *,
    model_label: str,
    model_probability: float,
    model_margin: float,
    allow_high_confidence_model_override: bool = False,
) -> TargetAwareDecision:
    """Resolve stance while keeping the legacy rule as the conservative anchor."""
    legacy = generic_legacy_label(row.get("rule_stance_label", ""))
    target_type = _clean(row.get("target_type", ""))
    if target_type not in {"person", "party"}:
        return TargetAwareDecision(legacy, "non_candidate_target_legacy")

    text = _clean(row.get("text_excerpt", ""))
    target_name = _clean(row.get("target_name", ""))
    if not target_name or target_name not in text:
        return TargetAwareDecision("neutral", "target_not_in_current_sentence")

    if _METALINGUISTIC_EXAMPLE.search(text):
        return TargetAwareDecision("neutral", "metalinguistic_stance_example")
    if _NEGATED_RECEPTION.search(text):
        return TargetAwareDecision("neutral", "negated_stance_reception")
    if _REPORT_ONLY.search(text) and not _SPEAKER_EVALUATION.search(text):
        return TargetAwareDecision("neutral", "reported_stance_without_speaker_evaluation")

    compact = text.replace(" ", "")
    self_target = f"우리{target_name}" in compact or f"저희{target_name}" in compact
    if (self_target or target_name in text) and _ALIGNMENT.search(text) and _NORMATIVE_ACTION.search(text):
        return TargetAwareDecision("positive", "alignment_with_target_proposal")

    if allow_high_confidence_model_override and legacy == "neutral" and model_label != "neutral":
        direct_cue = (
            model_label == "negative" and bool(_DIRECT_NEGATIVE.search(text))
        ) or (
            model_label == "positive" and bool(_DIRECT_POSITIVE.search(text))
        )
        if model_probability >= 0.80 and model_margin >= 0.60 and direct_cue:
            return TargetAwareDecision(model_label, "high_confidence_direct_model_override", True)

    return TargetAwareDecision(legacy, "legacy_anchor")
