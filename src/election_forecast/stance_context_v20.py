"""Strict ownership policy for high-precision stance V20."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_context_v19 import contextual_owner_resolution_reasons_v19
from election_forecast.stance_precision import clean_text


_GOVERNMENT_OFFICIAL = re.compile(
    r"(?:^|\s)(?:국가안보실(?:장|제?\d+차장)|외교부장관|국무총리|총리|"
    r"[가-힣]{2,20}부장관|장관|차관|대통령비서실장|청와대수석)(?:\s|$)"
)
_AMBASSADOR = re.compile(r"(?:^|\s)주[가-힣]+대한민국대사(?:\s|$)")
_TARGET_ADMINISTRATION = re.compile(r"[가-힣]{2,4}\s*(?:정부|정권)")
_PARTY_SELF_POSITION = re.compile(
    r"(?:우리|저희)(?:도|는|가|\s*당)?.{0,80}[가-힣]{2,20}당(?:도|은|는|이|가)"
    r".{0,140}(?:협력|지지|찬성|환영|지원|추진|노력)"
)
_IMPERSONAL_REPORT = re.compile(
    r"(?:평가|지적|거론|혹평|주장|분석|경고)(?:(?:을|를)\s*)?"
    r"(?:받고|받아|되고|되어|돼|하고\s*있|해\s*왔|했습니다|되고\s*있)"
)
_REPORT_CONTINUATION = re.compile(r"^(?:그\s*원인으로|그에\s*따르면|이러한\s*평가는|그\s*평가는)")
_PREVIOUS_EXTERNAL_REPORT = re.compile(
    r"(?:평가|지적|거론|혹평|주장|분석|경고).{0,60}(?:하고\s*있|되고\s*있|받고\s*있)"
)
_REPORTED_PUBLIC_REACTION = re.compile(
    r"(?:농어민|서민|국민|유권자|시장|기업|투자자).{0,180}"
    r"(?:상실감|불신|공분|걱정|원망|한탄|평가를\s*받|지지|반대)"
)
_HISTORICAL_GOVERNMENT_SCOPE = re.compile(
    r"(?:역대|여러|과거의\s*여러)\s*정부|민주\s*정부\s*\d+년"
)
_CONDITIONAL_DIRECTION = re.compile(
    r"(?:만일|만약|않으면|잃는다면|한다면|하게\s*되면|오락가락하게\s*되면)"
    r".{0,240}(?:실패|지지도|지지율|신뢰|불확실|혼란|위기|무능|대가|잘못)"
)
_GENERAL_GOVERNMENT_PROPOSITION = re.compile(
    r"나라가\s*(?:망하|잘못되).{0,120}(?:정부와|정부의).{0,100}(?:부패|비리|정경유착)"
)
_NON_EVALUATIVE_CONFUSION = re.compile(
    r"(?:기업과\s*투자자|기업과\s*투자자,\s*정부).{0,100}(?:모두\s*)?혼란"
)
_ANALYTICAL_CAUSAL_PROJECTION = re.compile(
    r"(?:대책도\s*없이|대책\s*없이|준비도\s*없이|근거도\s*없이|"
    r"뚜렷한\s*대책도\s*없이).{0,220}"
    r"(?:신뢰를\s*떨어뜨|불확실성을\s*증폭|혼란에\s*빠뜨|위기를\s*가중)"
)
_EXPLICIT_FIRST_PERSON_POSITIVE = re.compile(
    r"(?:저는|제가|본\s*(?:의원|위원)은).{0,220}"
    r"(?:지지|찬성|환영|높이\s*평가|긍정적으로\s*평가|성과|훌륭)"
)


def contextual_strict_owner_reasons_v20(row: Mapping[str, object]) -> tuple[str, ...]:
    """Add role, report-chain, conditional, and historical-scope abstentions."""

    reasons = list(contextual_owner_resolution_reasons_v19(row))
    current = clean_text(row.get("text_excerpt", ""))
    before = clean_text(row.get("context_before", ""))
    speaker = clean_text(row.get("speaker", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    if _GOVERNMENT_OFFICIAL.search(speaker) and (
        target_type == "government" or _TARGET_ADMINISTRATION.search(current)
    ):
        reasons.append("government_official_self_position_v20")
    if _AMBASSADOR.search(speaker) and re.search(r"(?:우리\s*)?정부의", current):
        reasons.append("diplomat_reports_external_position")
    if target_type == "party" and _PARTY_SELF_POSITION.search(current):
        reasons.append("party_self_position_v20")
    if _IMPERSONAL_REPORT.search(current):
        reasons.append("impersonal_reported_direction_v20")
    if _REPORT_CONTINUATION.search(current) and _PREVIOUS_EXTERNAL_REPORT.search(before):
        reasons.append("continued_external_report_v20")
    if _REPORTED_PUBLIC_REACTION.search(current):
        reasons.append("reported_public_reaction_v20")
    if target_type == "government" and _HISTORICAL_GOVERNMENT_SCOPE.search(current):
        reasons.append("historical_government_scope_v20")
    if _CONDITIONAL_DIRECTION.search(current):
        reasons.append("conditional_direction_v20")
    if target_type == "government" and _GENERAL_GOVERNMENT_PROPOSITION.search(current):
        reasons.append("generic_government_proposition_v20")
    if _NON_EVALUATIVE_CONFUSION.search(current):
        reasons.append("non_evaluative_confusion_v20")
    if _ANALYTICAL_CAUSAL_PROJECTION.search(current):
        reasons.append("analytical_causal_projection_v20")
    return tuple(dict.fromkeys(reasons))


def apply_contextual_strict_owner_gate_v20(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Apply V20 and require explicit first-person ownership for positives."""

    output = np.asarray(prediction, dtype=str).astype("<U8")
    if len(frame) != len(output):
        raise ValueError("frame and prediction lengths do not match")
    encoded: list[str] = []
    for index, row in enumerate(frame):
        reasons = list(contextual_strict_owner_reasons_v20(row))
        current = clean_text(row.get("text_excerpt", ""))
        if output[index] == "positive" and not _EXPLICIT_FIRST_PERSON_POSITIVE.search(current):
            reasons.append("positive_owner_not_explicit_v20")
        reasons = list(dict.fromkeys(reasons))
        encoded.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded
