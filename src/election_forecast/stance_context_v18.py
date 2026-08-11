"""Temporal-regime and assertion-status attribution gates for stance V18."""

from __future__ import annotations

import re
from datetime import date
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from election_forecast.stance_context_v17 import contextual_speaker_scope_reasons_v17
from election_forecast.stance_precision import clean_text


_NAMED_REGIMES = (
    (re.compile(r"(?:김영삼|문민)\s*(?:정부|정권)"), date(1993, 2, 25), date(1998, 2, 24)),
    (re.compile(r"(?:김대중|국민의정부)\s*(?:정부|정권)?"), date(1998, 2, 25), date(2003, 2, 24)),
    (re.compile(r"(?:노무현|참여)\s*(?:정부|정권)"), date(2003, 2, 25), date(2008, 2, 24)),
    (re.compile(r"(?:이명박|MB)\s*(?:정부|정권)"), date(2008, 2, 25), date(2013, 2, 24)),
    (re.compile(r"박근혜\s*(?:정부|정권)"), date(2013, 2, 25), date(2017, 5, 9)),
    (re.compile(r"문재인\s*(?:정부|정권)"), date(2017, 5, 10), date(2022, 5, 9)),
)
_GOVERNING_SELF_COMMITMENT = re.compile(
    r"(?:우리\s*당과\s*정부|정부와\s*우리\s*당).{0,160}"
    r"(?:추진|노력|시행|실현|마련|하겠습니다|하도록\s*하겠습니다)"
)
_UNRESOLVED_ATTRIBUTION_QUESTION = re.compile(
    r"(?:잘못|실패|책임|문제).{0,100}(?:인가|것인가|아닌가)(?:\s|,|\.|$)|"
    r"(?:인가|것인가|아닌가).{0,100}(?:잘못|실패|책임|문제)"
)
_ANALYTICAL_SUPPORT_PROJECTION = re.compile(
    r"(?:경제가\s*어려워질수록|정책을\s*쓰게\s*되면|그렇게\s*되면).{0,180}"
    r"(?:정부|여당|대통령).{0,60}(?:지지도|지지율).{0,50}(?:떨어|하락)|"
    r"(?:정부|여당|대통령).{0,60}(?:지지도|지지율).{0,50}(?:떨어|하락).{0,140}"
    r"(?:정국|경제).{0,30}(?:불안|악순환)"
)
_CONDITIONAL_EVALUATION = re.compile(
    r"(?:않으면|한다면|하게\s*되면|잘못되면).{0,160}"
    r"(?:나쁜|무능한|실패한)\s*(?:정부|정권|정책)"
)
_REPORTED_POLITICAL_ACTORS = re.compile(
    r"[가-힣]{2,4}\s*(?:서울)?시장도\s*그렇고.{0,80}"
    r"[가-힣]{2,4}\s*의원도\s*그렇고.{0,160}(?:탓|책임|실패)"
)


def _meeting_date(row: Mapping[str, object]) -> date | None:
    value = pd.to_datetime(row.get("meeting_date", ""), errors="coerce")
    return None if pd.isna(value) else value.date()


def _mentions_noncurrent_named_regime(row: Mapping[str, object]) -> bool:
    if clean_text(row.get("target_type", "none")) != "government":
        return False
    current = clean_text(row.get("text_excerpt", ""))
    meeting = _meeting_date(row)
    if meeting is None:
        return False
    return any(
        pattern.search(current) and not (start <= meeting <= end)
        for pattern, start, end in _NAMED_REGIMES
    )


def contextual_assertion_reasons_v18(
    row: Mapping[str, object],
) -> tuple[str, ...]:
    """Add date-aware regime and non-assertion abstentions to frozen V17."""

    reasons = list(contextual_speaker_scope_reasons_v17(row))
    current = clean_text(row.get("text_excerpt", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    if _mentions_noncurrent_named_regime(row):
        reasons.append("named_noncurrent_government")
    if target_type == "government" and _GOVERNING_SELF_COMMITMENT.search(current):
        reasons.append("governing_party_self_commitment")
    if _UNRESOLVED_ATTRIBUTION_QUESTION.search(current):
        reasons.append("unresolved_attribution_question")
    if _ANALYTICAL_SUPPORT_PROJECTION.search(current):
        reasons.append("analytical_support_projection")
    if _CONDITIONAL_EVALUATION.search(current):
        reasons.append("conditional_evaluation")
    if _REPORTED_POLITICAL_ACTORS.search(current):
        reasons.append("reported_political_actors")
    return tuple(dict.fromkeys(reasons))


def apply_contextual_assertion_gate_v18(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Apply the V18 gate to the unchanged frozen base predictions."""

    output = np.asarray(prediction, dtype=str).astype("<U8")
    if len(frame) != len(output):
        raise ValueError("frame and prediction lengths do not match")
    encoded_reasons: list[str] = []
    for index, row in enumerate(frame):
        reasons = contextual_assertion_reasons_v18(row)
        encoded_reasons.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded_reasons
