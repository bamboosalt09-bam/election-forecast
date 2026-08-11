"""Speaker-role and temporal-scope attribution gates for stance V17."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_context_v16 import contextual_attribution_reasons_v16
from election_forecast.stance_precision import clean_text


_ANSWER_REQUEST = re.compile(
    r"(?:판단|생각|조치|행위).{0,45}(?:되는지|인지|었는지).{0,100}"
    r"(?:답변해?|밝혀|말씀해)\s*주시기\s*바랍니다"
)
_REPORTED_EXPERT = re.compile(
    r"(?:교수|전문가|학자)(?:님)?(?:께서|이|가|은|는).{0,180}"
    r"(?:판단|평가|주장|지적|말씀)(?:을\s*)?(?:하셨|했|했습니다|하였습니다)"
)
_GOVERNMENT_REPRESENTATIVE = re.compile(
    r"(?:^|\s)(?:대통령|국무총리|총리|[가-힣]{2,20}부장관|장관|차관|청장|처장|"
    r"대통령비서실장|청와대수석)(?:\s|$)"
)
_TARGET_REPRESENTATIVE_REPORT = re.compile(
    r"(?:대통령|국무총리|총리|장관).{0,180}(?:송구|책임|실패|잘못).{0,100}"
    r"(?:말|생각|밝히|했|했습니다|하였습니다)"
)
_FOREIGN_OR_FUTURE_GOVERNMENT = re.compile(
    r"(?:미국|일본|중국|러시아|영국|프랑스|독일|체코|네덜란드|캐나다|호주|인도)\s*"
    r"(?:신\s*)?(?:정부|행정부)|(?:다음|차기|향후)\s*정부"
)
_EXTERNAL_SUPPORT_RECEIVED = re.compile(
    r"(?:국제사회|국민).{0,120}(?:우리\s*)?정부.{0,100}"
    r"(?:지지|환영|신뢰).{0,40}(?:받|보내|표하)"
)
_FOLLOWING_REBUTTAL = re.compile(
    r"(?:탓|비판|책임).{0,120}(?:유감|부당|잘못|아니)"
)


def contextual_speaker_scope_reasons_v17(
    row: Mapping[str, object],
) -> tuple[str, ...]:
    """Add speaker-role, question, and target-time-scope abstentions."""

    reasons = list(contextual_attribution_reasons_v16(row))
    current = clean_text(row.get("text_excerpt", ""))
    after = clean_text(row.get("context_after", ""))
    speaker = clean_text(row.get("speaker", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    if _ANSWER_REQUEST.search(current):
        reasons.append("answer_request_not_owned_stance")
    if _REPORTED_EXPERT.search(current):
        reasons.append("reported_expert_stance")
    if target_type == "government" and _GOVERNMENT_REPRESENTATIVE.search(speaker):
        reasons.append("government_representative_self_position")
    if target_type == "government" and _TARGET_REPRESENTATIVE_REPORT.search(current):
        reasons.append("reported_target_representative_position")
    if target_type == "government" and _FOREIGN_OR_FUTURE_GOVERNMENT.search(current):
        reasons.append("foreign_or_future_government_scope")
    if target_type == "government" and _EXTERNAL_SUPPORT_RECEIVED.search(current):
        reasons.append("external_support_report")
    if after and _FOLLOWING_REBUTTAL.search(after):
        reasons.append("claim_rebutted_in_following_context_v17")
    return tuple(dict.fromkeys(reasons))


def apply_contextual_speaker_scope_gate_v17(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Apply the V17 gate to the unchanged frozen base predictions."""

    output = np.asarray(prediction, dtype=str).astype("<U8")
    if len(frame) != len(output):
        raise ValueError("frame and prediction lengths do not match")
    encoded_reasons: list[str] = []
    for index, row in enumerate(frame):
        reasons = contextual_speaker_scope_reasons_v17(row)
        encoded_reasons.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded_reasons
