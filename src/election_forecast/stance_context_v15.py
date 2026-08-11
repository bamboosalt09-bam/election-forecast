"""Additional ownership abstention for the full-corpus stance experiment.

V14 already rejects questions, quotations, external attribution, and most
target self-reports.  V15 keeps that policy frozen and adds one general Korean
syntax check: when the assigned target is the topic of a clause and owns a
reporting predicate, the clause describes the target's position rather than
the current speaker's stance toward the target.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_precision import (
    ambiguity_abstention_reasons,
    clean_text,
)


_TARGET_ROLE_SUFFIX = r"(?:후보|대통령|정부|정권|측|캠프|당)?"
_TARGET_TOPIC_PARTICLE = r"(?:은|는|도|에서는|에서|께서는|께서)"
_TARGET_OWNED_REPORT = re.compile(
    r"(?:진단|시인|말하|이야기|평가|설명|발표|밝히|주장|기대|지지|환영)"
    r"(?:했|하였|했다|하였다|했습니다|하였습니다|했다고|하였다고|하고\s*있)"
)


def _target_forms(row: Mapping[str, object]) -> tuple[str, ...]:
    values = {
        clean_text(row.get("target_name", "")),
        clean_text(row.get("target_alias", "")),
    }
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def target_owned_report_reason(row: Mapping[str, object]) -> str:
    """Detect a target-owned report that V14's shorter window can miss."""

    target_type = clean_text(row.get("target_type", "none")) or "none"
    if target_type not in {"person", "party"}:
        return ""
    current = clean_text(row.get("text_excerpt", ""))
    for target in _target_forms(row):
        actor = re.compile(
            rf"{re.escape(target)}\s*{_TARGET_ROLE_SUFFIX}\s*"
            rf"{_TARGET_TOPIC_PARTICLE}\s*",
        )
        match = actor.search(current)
        if not match:
            continue
        tail = current[match.start() : match.end() + 180]
        if _TARGET_OWNED_REPORT.search(tail):
            return "target_subject_owns_reported_stance"
    return ""


def contextual_ownership_abstention_reasons_v15(
    row: Mapping[str, object],
) -> tuple[str, ...]:
    """Return V14 reasons plus the extended target-ownership reason."""

    reasons = list(ambiguity_abstention_reasons(row))
    ownership = target_owned_report_reason(row)
    if ownership:
        reasons.append(ownership)
    return tuple(dict.fromkeys(reasons))


def apply_contextual_ownership_gate_v15(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Apply the V15 gate while preserving neutral abstentions."""

    output = np.asarray(prediction, dtype=str).astype("<U8")
    if len(frame) != len(output):
        raise ValueError("frame and prediction lengths do not match")
    encoded_reasons: list[str] = []
    for index, row in enumerate(frame):
        reasons = contextual_ownership_abstention_reasons_v15(row)
        encoded_reasons.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded_reasons
