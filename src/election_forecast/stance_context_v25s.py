"""Victim-role, factual-mention, and hypothetical-agent gates for V25-S."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_context_v24s import apply_lexical_role_gate_v24s
from election_forecast.stance_precision import clean_text


_BUDGET_SIZE_FACT = re.compile(
    r"정부예산이\s*[\d,.]+\s*조인데\s*공기업이\s*[\d,.]+\s*조"
)
_IMPEACHMENT_EVENT_DESCRIPTION = re.compile(
    r"(?:대통령\s*)?탄핵소추\s*의결로\s*권한이\s*정지.{0,160}"
    r"(?:혼란|국가위기)\s*상황"
)
_TARGET_IS_ATTACK_VICTIM = re.compile(
    r"(?:망가뜨리|악화시켜|타격을\s*주).{0,180}"
    r"(?:정부|정권).{0,180}(?:재집권할\s*수\s*없|정부를\s*바꾸|정권을\s*교체)"
)
_TARGET_DISRUPTED_BY_EXTERNAL_EVENT = re.compile(
    r"(?:문서유출|외부\s*공격|해킹|사건)\s*때문에.{0,160}"
    r"(?:대통령|정부).{0,160}(?:차질|역량을\s*분산|지장).{0,100}(?:안타깝|우려)"
)
_MACRO_HYPOTHETICAL_RISK = re.compile(
    r"(?:경우에|경우|다면|떨어진다면).{0,180}"
    r"(?:경제파탄|재정파탄|위기).{0,50}(?:올\s*수밖에|발생할\s*수|우려)"
)
_REPORTED_COMMON_VIEW = re.compile(r"(?:것|정부|정권)(?:이)?다라는\s*것이\s*중론입니다")
_HYPOTHETICAL_GOVERNMENT_EXECUTOR = re.compile(
    r"(?:분배론(?:을)?|분배를)\s*강조하면.{0,160}정부가.{0,160}"
    r"(?:수단도\s*많지\s*않|할\s*수\s*있는\s*시대는\s*아니).{0,320}(?:염려|악순환)"
)


def semantic_role_reasons_v25s(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return V25-S abstentions for independently observed V16 errors."""

    current = clean_text(row.get("text_excerpt", ""))
    reasons: list[str] = []
    if _BUDGET_SIZE_FACT.search(current):
        reasons.append("government_only_in_budget_size_fact_v25s")
    if _IMPEACHMENT_EVENT_DESCRIPTION.search(current):
        reasons.append("neutral_impeachment_event_description_v25s")
    if _TARGET_IS_ATTACK_VICTIM.search(current):
        reasons.append("assigned_target_is_attack_victim_v25s")
    if _TARGET_DISRUPTED_BY_EXTERNAL_EVENT.search(current):
        reasons.append("assigned_target_disrupted_by_external_event_v25s")
    if _MACRO_HYPOTHETICAL_RISK.search(current):
        reasons.append("macro_hypothetical_risk_v25s")
    if _REPORTED_COMMON_VIEW.search(current):
        reasons.append("reported_common_view_v25s")
    if _HYPOTHETICAL_GOVERNMENT_EXECUTOR.search(current):
        reasons.append("government_is_hypothetical_executor_v25s")
    return tuple(dict.fromkeys(reasons))


def apply_semantic_role_gate_v25s(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Apply frozen V24-S followed by V25-S semantic-role abstentions."""

    output, v24s_reasons, resolution = apply_lexical_role_gate_v24s(frame, prediction)
    encoded: list[str] = []
    for index, row in enumerate(frame):
        reasons = [reason for reason in v24s_reasons[index].split("|") if reason]
        reasons.extend(semantic_role_reasons_v25s(row))
        reasons = list(dict.fromkeys(reasons))
        encoded.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded, resolution
