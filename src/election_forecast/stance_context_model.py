"""Context composition and weak supervision for Korean stance modeling."""

from __future__ import annotations

import re
from dataclasses import dataclass


_SPACE = re.compile(r"\s+")
_DIRECT_NEGATIVE = re.compile(
    r"유감|심각|믿을 수 없|능력.{0,8}없|제출.{0,8}않|위반|의혹|피해|"
    r"책임.{0,8}(져|지셔|있)|적절하지 않|잘못.{0,8}(생각|판단)|아니겠습니까|문제라고"
)
_DIRECT_POSITIVE = re.compile(
    r"노고|격려|감사|같은 생각|환영|높이 평가|뜻깊|잘 해|잘하고|성과를|찬성"
)
_REFORM_CONTEXT = re.compile(r"척결|근절|엄단|예방|처단|바로잡|단속|투명|개선|강화")
_CORRUPTION_TOPIC = re.compile(r"부정부패|공직비리|부패사범|비리 척결")
_REPORT_OR_QUOTE = re.compile(
    r"[‘’“”\"']|라고|다고|라는|이라며|발언|주장|보도|회신|발표|판결|자료|보고"
)
_CONDITIONAL_DEFENSE = re.compile(
    r"문제가 없다고|책임이 없다고|잘못이 없다고|정당하다고|문제가 없다면|책임이 없다면"
)
_QUESTION = re.compile(r"[?？]|습니까|아닌가|것 아니|겠습니까")
_FIRST_PERSON = re.compile(r"저는|제가|본 위원|저희는|생각합니다|봅니다|말씀드립니다")
_NEGATIVE_AFTER_DEFENSE = re.compile(r"그런데|하지만|그러나|불구하고|무책임|실패|유감|심각|잘못")


@dataclass(frozen=True)
class WeakContextLabel:
    label: str
    confidence: float
    reason: str


def _clean(value: object) -> str:
    return _SPACE.sub(" ", "" if value is None else str(value)).strip()


def _mask_target(text: str, target_name: str, target_alias: str) -> str:
    masked = text
    for value in sorted({target_name, target_alias}, key=len, reverse=True):
        if value:
            masked = masked.replace(value, " [TARGET] ")
    return _clean(masked)


def compose_context_input(row: dict[str, object]) -> str:
    target_name = _clean(row.get("target_name", ""))
    target_alias = _clean(row.get("target_alias", ""))
    current = _mask_target(_clean(row.get("text_excerpt", "")), target_name, target_alias)
    before = _mask_target(_clean(row.get("context_before", "")), target_name, target_alias)
    after = _mask_target(_clean(row.get("context_after", "")), target_name, target_alias)
    target_type = _clean(row.get("target_type", "none")) or "none"
    # Repeating CURRENT gives it more TF-IDF mass than optional context while
    # retaining explicit section boundaries.
    return (
        f"[TARGET_TYPE={target_type}] [CURRENT] {current} [CURRENT_REPEAT] {current} "
        f"[BEFORE] {before} [AFTER] {after}"
    )


def weak_context_label(row: dict[str, object]) -> WeakContextLabel:
    text = _clean(row.get("text_excerpt", ""))
    before = _clean(row.get("context_before", ""))
    after = _clean(row.get("context_after", ""))
    context = f"{before} {text} {after}"
    try:
        polarity = int(float(row.get("rule_stance_polarity", 0) or 0))
    except (TypeError, ValueError):
        polarity = 0
    legacy = {-1: "negative", 0: "neutral", 1: "positive"}.get(polarity, "neutral")

    direct_negative = bool(_DIRECT_NEGATIVE.search(text))
    direct_positive = bool(_DIRECT_POSITIVE.search(text))
    if direct_negative and direct_positive:
        return WeakContextLabel("neutral", 0.35, "mixed_direct_cues")

    if legacy == "positive" and _CONDITIONAL_DEFENSE.search(text):
        if _QUESTION.search(text) or _NEGATIVE_AFTER_DEFENSE.search(context):
            return WeakContextLabel("negative", 0.62, "defense_inside_rebuttal_or_question")
        if _REPORT_OR_QUOTE.search(text) and not _FIRST_PERSON.search(text):
            return WeakContextLabel("neutral", 0.55, "reported_defense")

    if legacy == "negative" and _CORRUPTION_TOPIC.search(text) and _REFORM_CONTEXT.search(text):
        if direct_positive:
            return WeakContextLabel("positive", 0.58, "praise_for_anti_corruption_action")
        if not _FIRST_PERSON.search(text) and not _QUESTION.search(text):
            return WeakContextLabel("neutral", 0.62, "anti_corruption_policy_not_attack")

    if legacy == "neutral":
        if direct_negative:
            return WeakContextLabel("negative", 0.58, "implicit_direct_criticism")
        if direct_positive:
            return WeakContextLabel("positive", 0.58, "implicit_direct_support")

    if legacy != "neutral" and _REPORT_OR_QUOTE.search(text) and not _FIRST_PERSON.search(text):
        return WeakContextLabel(legacy, 0.40, "reported_direction_retained_low_confidence")
    return WeakContextLabel(legacy, 0.52 if legacy == "neutral" else 0.60, "legacy_context_consistent")

