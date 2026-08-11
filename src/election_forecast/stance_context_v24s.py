"""Lexical-boundary and stance-conflict abstentions for V24-S."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_context_v22 import _AFFILIATED_OR_LOCAL, _CENTRAL_GOVERNMENT_REFERENCE
from election_forecast.stance_context_v23s import apply_pragmatic_role_gate_v23s
from election_forecast.stance_precision import clean_text


_POSITIVE_EFFECT_CONFLICT = re.compile(
    r"(?:효과를\s*볼\s*수\s*있었|효과가\s*있었|성과를\s*거두|"
    r"회복에\s*기여|활성화에\s*기여|가능하게\s*했)"
)
_NEUTRAL_GOVERNMENT_RESPONSE = re.compile(
    r"정부는.{0,220}(?:위기|상황|문제)에\s*대응하여.{0,220}"
    r"(?:대책을\s*시행|프로그램을\s*마련|구조조정을\s*추진)"
)
_EXPLICIT_GOVERNMENT_CRITICISM = re.compile(
    r"정부.{0,160}(?:실패|실정|무능|잘못|무책임|부패|파탄|방치|왜곡)"
)
_GENERIC_INSTITUTIONAL_TENSION = re.compile(
    r"(?:정권을\s*대표하는\s*)?(?:대통령\s*및\s*)?정부의\s*입장과\s*"
    r"중앙은행의\s*입장이\s*상충"
)
_TRUNCATED_FRAGMENT = re.compile(r"(?:도|은|는|이|가)\s*다\s*$")
_OTHER_PEOPLE_EMOTION = re.compile(
    r"(?:사람들|국민들|주민들|투자자들)(?:도|이|가|은|는)?.{0,180}"
    r"정부.{0,160}(?:분노|허탈|실망|원망|불신)"
)
_SPEAKER_EVALUATION_BEFORE_REPORT = re.compile(
    r"(?:문제|잘못|실패|부당|심각).{0,50}(?:여기|생각|판단|보)"
)
_NEUTRAL_SUPPORT_CONDITION = re.compile(
    r"정부\s*지원금이\s*끊기면.{0,220}(?:유지(?:를)?\s*못|불안\s*문제가\s*발생)"
)
_PAST_DEICTIC_GOVERNMENT = re.compile(
    r"(?:그때|당시).{0,120}정부(?:대책|정책).{0,120}(?:미흡|실패|잘못)"
)
_HISTORICAL_CRISIS_CONTEXT = re.compile(r"대공황|대침체|세계대전|외환위기")
_GENERIC_GOVERNMENT_MECHANISM = re.compile(
    r"정부가.{0,180}(?:조장|유지|확대).{0,180}(?:결국|그래서).{0,80}위기"
)
_FALSE_GOVERNMENT_COMPOUNDS = re.compile(r"천정부지")


def lexical_role_reasons_v24s(row: Mapping[str, object], prediction: str) -> tuple[str, ...]:
    """Return V24-S abstentions for independently observed V15 errors."""

    current = clean_text(row.get("text_excerpt", ""))
    before = clean_text(row.get("context_before", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    reasons: list[str] = []

    if prediction == "negative" and _POSITIVE_EFFECT_CONFLICT.search(current):
        reasons.append("positive_effect_conflicts_with_negative_v24s")
    if (
        _NEUTRAL_GOVERNMENT_RESPONSE.search(current)
        and not _EXPLICIT_GOVERNMENT_CRITICISM.search(current)
    ):
        reasons.append("neutral_government_response_v24s")
    if _GENERIC_INSTITUTIONAL_TENSION.search(current):
        reasons.append("generic_institutional_tension_v24s")
    if len(current) < 80 and _TRUNCATED_FRAGMENT.search(current):
        reasons.append("truncated_evaluative_fragment_v24s")
    public_report = _OTHER_PEOPLE_EMOTION.search(current)
    if public_report and not _SPEAKER_EVALUATION_BEFORE_REPORT.search(
        current[: public_report.start()]
    ):
        reasons.append("other_people_own_emotion_v24s")
    if _NEUTRAL_SUPPORT_CONDITION.search(current):
        reasons.append("neutral_support_condition_v24s")
    if _PAST_DEICTIC_GOVERNMENT.search(current):
        reasons.append("past_deictic_government_v24s")

    if target_type == "government":
        combined = f"{current} {before}"
        cleaned = _FALSE_GOVERNMENT_COMPOUNDS.sub("", combined)
        cleaned = _AFFILIATED_OR_LOCAL.sub("", cleaned)
        if not _CENTRAL_GOVERNMENT_REFERENCE.search(cleaned):
            reasons.append("government_only_inside_false_compound_v24s")
        if (
            _HISTORICAL_CRISIS_CONTEXT.search(before)
            and _GENERIC_GOVERNMENT_MECHANISM.search(current)
            and not re.search(r"(?:현|이|우리|현재의?)\s*정부|현\s*정권", current)
        ):
            reasons.append("generic_historical_crisis_mechanism_v24s")
    return tuple(dict.fromkeys(reasons))


def apply_lexical_role_gate_v24s(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Apply frozen V23-S followed by V24-S precision abstentions."""

    output, v23s_reasons, resolution = apply_pragmatic_role_gate_v23s(frame, prediction)
    encoded: list[str] = []
    for index, row in enumerate(frame):
        reasons = [reason for reason in v23s_reasons[index].split("|") if reason]
        reasons.extend(lexical_role_reasons_v24s(row, str(output[index])))
        reasons = list(dict.fromkeys(reasons))
        encoded.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded, resolution
