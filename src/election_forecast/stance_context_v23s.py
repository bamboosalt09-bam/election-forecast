"""Pragmatic-role and historical-scope gates for stance classifier V23-S."""

from __future__ import annotations

import re
from datetime import date
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from election_forecast.stance_context_v22 import apply_grammatical_target_gate_v22
from election_forecast.stance_precision import clean_text


_CRITICISM_OBJECT = re.compile(
    r"(?:정부|정권|청와대)(?:를|을)\s*(?:비난|공격|비판)하는\s*(?:소재|도구|수단|구실)"
)
_SINGULAR_PUBLIC_REPORT = re.compile(
    r"국민(?:은|이|가).{0,120}(?:정부|정권|청와대).{0,100}"
    r"(?:믿지\s*못|불신|신뢰하지\s*않|외면|반대)"
)
_COMMITTEE_COLLECTIVE_REPORT = re.compile(
    r"(?:많은|여러)\s*위원(?:님)?들이.{0,180}(?:걱정|우려|평가|지적)"
)
_NEUTRAL_GOVERNMENT_HYPOTHESIS = re.compile(
    r"정부가\s*만약.{0,260}(?:한다면|하면|하게\s*되면|한다고\s*하면)"
)
_CURRENT_GOVERNMENT = re.compile(r"(?:현|이|우리|현재의?)\s*정부|현\s*정권")
_ADMINISTRATION = re.compile(
    r"(?P<name>YS|DJ|MB|김영삼|김대중|노무현|이명박|박근혜|문재인)\s*(?:정부|정권)"
)
_ADMINISTRATION_END = {
    "YS": date(1998, 2, 25),
    "김영삼": date(1998, 2, 25),
    "DJ": date(2003, 2, 25),
    "김대중": date(2003, 2, 25),
    "노무현": date(2008, 2, 25),
    "MB": date(2013, 2, 25),
    "이명박": date(2013, 2, 25),
    "박근혜": date(2017, 5, 10),
    "문재인": date(2022, 5, 10),
}


def _only_historical_named_governments(row: Mapping[str, object], current: str) -> bool:
    if clean_text(row.get("target_type", "none")) != "government":
        return False
    if _CURRENT_GOVERNMENT.search(current):
        return False
    meeting = pd.to_datetime(row.get("meeting_date", ""), errors="coerce")
    if pd.isna(meeting):
        return False
    names = [match.group("name") for match in _ADMINISTRATION.finditer(current)]
    if not names:
        return False
    meeting_date = meeting.date()
    return all(_ADMINISTRATION_END[name] < meeting_date for name in names)


def pragmatic_role_reasons_v23s(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return V23-S abstentions for the five V14 residual error types."""

    current = clean_text(row.get("text_excerpt", ""))
    reasons: list[str] = []
    if _CRITICISM_OBJECT.search(current):
        reasons.append("government_is_criticism_object_v23s")
    if _SINGULAR_PUBLIC_REPORT.search(current):
        reasons.append("singular_public_owns_direction_v23s")
    if _COMMITTEE_COLLECTIVE_REPORT.search(current):
        reasons.append("committee_collective_owns_direction_v23s")
    if _NEUTRAL_GOVERNMENT_HYPOTHESIS.search(current):
        reasons.append("neutral_government_hypothesis_v23s")
    if _only_historical_named_governments(row, current):
        reasons.append("only_historical_named_government_v23s")
    return tuple(dict.fromkeys(reasons))


def apply_pragmatic_role_gate_v23s(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Apply V22 followed by V23-S pragmatic-role abstentions."""

    output, v22_reasons, resolution = apply_grammatical_target_gate_v22(frame, prediction)
    encoded: list[str] = []
    for index, row in enumerate(frame):
        reasons = [reason for reason in v22_reasons[index].split("|") if reason]
        reasons.extend(pragmatic_role_reasons_v23s(row))
        reasons = list(dict.fromkeys(reasons))
        encoded.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded, resolution
