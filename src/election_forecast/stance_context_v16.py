"""Attribution gates learned from the independently locked V15 audit."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_context_v15 import (
    contextual_ownership_abstention_reasons_v15,
)
from election_forecast.stance_precision import clean_text


_EXTERNAL_OWNED_STANCE = re.compile(
    r"(?:일부\s*)?(?:야당(?:과\s*언론)?|언론|전문가|학자|"
    r"국민(?:과\s*국제사회)?|국제사회|여론|응답자|"
    r"미국(?:\s*(?:정부|행정부))?|일본(?:\s*정부)?|중국(?:\s*정부)?|"
    r"러시아(?:\s*정부)?|영국(?:\s*정부)?|프랑스(?:\s*정부)?|"
    r"독일(?:\s*정부)?|체코(?:\s*정부)?|네덜란드(?:\s*정부)?|"
    r"유럽연합|EU)\s*(?:은|는|이|가|도|들이|에서는|에서|또한)\s*"
    r".{0,160}?(?:실패|탓으로\s*돌리|비판|규탄|지적|평가|주장|단언|"
    r"지지|찬성|환영|반대|신뢰)"
)
_PAST_GOVERNMENT = re.compile(
    r"(?:지난|과거|전임)\s*.{0,35}(?:정부|정권|대통령)|"
    r"[가-힣]{2,12}\s*전\s*대통령|"
    r"당시\s*.{0,35}(?:정부|정권)"
)
_FOREIGN_GOVERNMENT = re.compile(
    r"(?:미국|일본|중국|러시아|영국|프랑스|독일|체코|네덜란드|"
    r"캐나다|호주|인도|북한)\s*(?:정부|행정부|당국)"
)
_SELF_POSITION = re.compile(r"(?:지지|찬성|환영|반대|협력|규탄|비판)(?:하|할|했|합니다|한다)")


def _target_subject_self_position(row: Mapping[str, object]) -> bool:
    target_type = clean_text(row.get("target_type", "none")) or "none"
    if target_type not in {"person", "party"}:
        return False
    current = clean_text(row.get("text_excerpt", ""))
    targets = {
        clean_text(row.get("target_name", "")),
        clean_text(row.get("target_alias", "")),
    }
    for target in (value for value in targets if value):
        subject = re.search(
            rf"{re.escape(target)}\s*(?:후보|대통령|측|캠프|당)?\s*(?:은|는|도|에서는|께서는)",
            current,
        )
        if subject and _SELF_POSITION.search(current[subject.end() : subject.end() + 180]):
            return True
    return False


def contextual_attribution_reasons_v16(
    row: Mapping[str, object],
) -> tuple[str, ...]:
    """Add external-owner and wrong-government abstentions to V15."""

    reasons = list(contextual_ownership_abstention_reasons_v15(row))
    current = clean_text(row.get("text_excerpt", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    if _EXTERNAL_OWNED_STANCE.search(current):
        reasons.append("external_actor_owns_direction")
    if target_type == "government" and _PAST_GOVERNMENT.search(current):
        reasons.append("historical_government_not_current_target")
    if target_type == "government" and _FOREIGN_GOVERNMENT.search(current):
        reasons.append("foreign_government_not_current_target")
    if _target_subject_self_position(row):
        reasons.append("target_subject_owns_self_position")
    return tuple(dict.fromkeys(reasons))


def apply_contextual_attribution_gate_v16(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Apply the V16 attribution policy to frozen base predictions."""

    output = np.asarray(prediction, dtype=str).astype("<U8")
    if len(frame) != len(output):
        raise ValueError("frame and prediction lengths do not match")
    encoded_reasons: list[str] = []
    for index, row in enumerate(frame):
        reasons = contextual_attribution_reasons_v16(row)
        encoded_reasons.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded_reasons
