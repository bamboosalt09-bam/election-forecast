"""Grammatical-role and central-government referent gates for stance V22."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_context_v21 import apply_discourse_target_gate_v21
from election_forecast.stance_precision import clean_text


_PARTY_CRITICISM_ACTION = re.compile(
    r"(?:경질|사퇴|폐기|수정|사과|책임|철회).{0,50}(?:요구|촉구)|"
    r"(?:비판|규탄|책임을\s*추궁)"
)
_GOVERNMENT_POLICY_BENEFICIARY = re.compile(
    r"(?:정부의|정부가\s*추진하는).{0,100}"
    r"(?:국가균형발전|정책|사업|계획|목표|방침).{0,120}(?:저해|방해|훼손|가로막)"
)
_AFFILIATED_OR_LOCAL = re.compile(
    r"정부(?:투자|출연|산하)기관|준정부기관|지방자치단체|지방정부|지자체"
)
_CENTRAL_GOVERNMENT_REFERENCE = re.compile(
    r"(?<!준)(?<!지방)(?<!투자)(?<!출연)정부(?!기관)|정권|청와대|행정부|내각"
)
_HYPOTHETICAL_FAILURE = re.compile(
    r"(?:했거나|있거나|없거나|못한\s*경우|안\s*되어\s*있거나).{0,220}"
    r"(?:경우|실패(?:하는)?\s*요인)|"
    r"(?:경우|경우에는).{0,140}(?:실패(?:하는)?\s*요인|실패할\s*수)"
)
_COLLECTIVE_REPORT = re.compile(
    r"(?:노동계|국민들|유권자들|시민들)(?:이|가|은|는)?.{0,180}"
    r"(?:정부|정책|정권).{0,120}(?:불신|비판|반대|말하고|평가하고|지적하고)|"
    r"(?:노동계|국민들|유권자들|시민들)(?:이|가|은|는)?.{0,180}"
    r"(?:말하고|평가하고|비판하고|지적하고)"
)


def grammatical_target_reasons_v22(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return target-role abstentions not covered by V21."""

    current = clean_text(row.get("text_excerpt", ""))
    before = clean_text(row.get("context_before", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    target_name = clean_text(row.get("target_name", ""))
    reasons: list[str] = []

    if (
        target_type == "party"
        and target_name
        and target_name in current
        and _PARTY_CRITICISM_ACTION.search(current[current.find(target_name) :])
    ):
        reasons.append("party_owns_criticism_not_negative_target_v22")

    if target_type == "government" and _GOVERNMENT_POLICY_BENEFICIARY.search(current):
        reasons.append("government_policy_is_beneficiary_v22")

    if target_type == "government":
        without_affiliates = _AFFILIATED_OR_LOCAL.sub("", current)
        if _AFFILIATED_OR_LOCAL.search(current) and not _CENTRAL_GOVERNMENT_REFERENCE.search(
            without_affiliates
        ):
            reasons.append("affiliated_or_local_body_not_central_government_v22")
        combined = f"{current} {before}"
        without_affiliates = _AFFILIATED_OR_LOCAL.sub("", combined)
        if not _CENTRAL_GOVERNMENT_REFERENCE.search(without_affiliates):
            reasons.append("central_government_referent_absent_v22")

    if _HYPOTHETICAL_FAILURE.search(current):
        reasons.append("generic_hypothetical_failure_v22")

    if _COLLECTIVE_REPORT.search(current):
        reasons.append("collective_report_owns_direction_v22")

    return tuple(dict.fromkeys(reasons))


def apply_grammatical_target_gate_v22(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Apply V21 and then V22 grammatical-role abstentions."""

    output, v21_reasons, resolution = apply_discourse_target_gate_v21(frame, prediction)
    encoded: list[str] = []
    for index, row in enumerate(frame):
        reasons = [reason for reason in v21_reasons[index].split("|") if reason]
        reasons.extend(grammatical_target_reasons_v22(row))
        reasons = list(dict.fromkeys(reasons))
        encoded.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded, resolution
