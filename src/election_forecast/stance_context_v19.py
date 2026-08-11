"""Attribution-owner and government-scope gates for stance V19."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_context_v18 import contextual_assertion_reasons_v18
from election_forecast.stance_precision import clean_text


_LOCAL_GOVERNMENT = re.compile(
    r"(?:지방|광역|기초|시[ㆍ·]?도|특별자치)\s*(?:자치)?정부|지방정부"
)
_SURVEY_OWNED_STANCE = re.compile(
    r"(?:갤럽|여론조사|설문조사|조사에\s*따르면|조사\s*결과).{0,180}"
    r"(?:\d+(?:[.]\d+)?\s*%|퍼센트|국민|응답자).{0,180}"
    r"(?:실패|지지|찬성|반대|신뢰|불신|평가|잘못)"
)
_COLLECTIVE_SUBJECT = re.compile(
    r"(?:기업(?:인|들|하는\s*사람들)?|투자자(?:들)?|시장|국민(?:들)?|"
    r"유권자(?:들)?|응답자(?:들)?)(?:은|는|이|가|도|에서는|에서)?\s*"
    r".{0,140}(?:정부|정권|대통령|정책).{0,100}"
    r"(?:의심|불신|신뢰|지지|찬성|반대|비판|규탄|실패|잘못)"
)
_GOVERNMENT_POLICY_IS_SUPPORT_OBJECT = re.compile(
    r"(?:우리\s*)?정부의\s*.{0,140}(?:정책|노력|입장|비핵화|평화|조치|대응)"
    r".{0,80}(?:지지|환영|찬성)(?:하|했|합니다|하고|한다|받|보내|표하)"
)
_SPEAKER_FIRST_PERSON_SUPPORT = re.compile(
    r"(?:저는|본\s*(?:의원|위원)은|제가).{0,180}"
    r"(?:우리\s*)?정부의.{0,100}(?:지지|환영|찬성)"
)
_CURRENT_GOVERNMENT_CORRECTION_CONTEXT = re.compile(
    r"현\s*정부(?:는|도|가|\s*역시).{0,220}"
    r"(?:해결|중단|폐기|수정|조정|축소|재검토|대책)"
)
_UNNAMED_PRIOR_POLICY_EVALUATION = re.compile(
    r"(?:사업|정책).{0,100}정부가.{0,100}(?:내놓|추진|도입).{0,80}"
    r"(?:잘못|실패)|정부가.{0,100}(?:내놓|추진|도입).{0,80}"
    r"(?:잘못|실패).{0,60}(?:사업|정책)"
)


def contextual_owner_resolution_reasons_v19(
    row: Mapping[str, object],
) -> tuple[str, ...]:
    """Add general owner and government-scope abstentions to frozen V18."""

    reasons = list(contextual_assertion_reasons_v18(row))
    current = clean_text(row.get("text_excerpt", ""))
    before = clean_text(row.get("context_before", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    if target_type == "government" and _LOCAL_GOVERNMENT.search(current):
        reasons.append("local_government_not_national_target")
    if _SURVEY_OWNED_STANCE.search(current):
        reasons.append("survey_owns_direction")
    if _COLLECTIVE_SUBJECT.search(current):
        reasons.append("collective_actor_owns_direction")
    if (
        target_type == "government"
        and _GOVERNMENT_POLICY_IS_SUPPORT_OBJECT.search(current)
        and not _SPEAKER_FIRST_PERSON_SUPPORT.search(current)
    ):
        reasons.append("government_policy_is_support_object")
    if (
        target_type == "government"
        and _CURRENT_GOVERNMENT_CORRECTION_CONTEXT.search(before)
        and _UNNAMED_PRIOR_POLICY_EVALUATION.search(current)
    ):
        reasons.append("unnamed_prior_policy_not_current_government")
    return tuple(dict.fromkeys(reasons))


def apply_contextual_owner_resolution_gate_v19(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Apply V19 to unchanged frozen base predictions."""

    output = np.asarray(prediction, dtype=str).astype("<U8")
    if len(frame) != len(output):
        raise ValueError("frame and prediction lengths do not match")
    encoded_reasons: list[str] = []
    for index, row in enumerate(frame):
        reasons = contextual_owner_resolution_reasons_v19(row)
        encoded_reasons.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded_reasons
