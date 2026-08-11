"""Discourse-owner and evaluation-target resolver for shadow stance V21."""

from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

import numpy as np

from election_forecast.stance_context_v20 import (
    _EXPLICIT_FIRST_PERSON_POSITIVE,
    contextual_strict_owner_reasons_v20,
)
from election_forecast.stance_precision import clean_text


_DEMONSTRATIVE_REFERENCE = re.compile(r"^(?:이처럼|이와\s*같이|그처럼|그러한|이러한)")
_NAMED_PRIOR_GOVERNMENT = re.compile(
    r"(?:[가-힣]{2,4}(?:[ㆍ·][가-힣]{2,4})*\s*(?:정부|정권)|"
    r"(?:과거|이전|전임|역대)\s*(?:정부|정권))"
)
_CONDITIONAL_FAILURE = re.compile(
    r"(?:못하면|못\s*할\s*경우|하지\s*못하면|하지\s*않으면|잡지\s*못하면)"
    r".{0,220}(?:실패한|무능한|성공한\s*[^.]{0,40}아니|끝날\s*수|파탄)"
)
_COMMITTEE_STAFF = re.compile(r"(?:수석)?전문위원|입법조사관|위원회\s*(?:직원|조사관)")
_REPORTED_OPINION_SUMMARY = re.compile(
    r"(?:의견|견해|평가|지적)(?:도|이|가|은|는)?\s*(?:있었습니다|있었으며|제시되었습니다|나왔습니다)"
)
_TARGET_SELF_QUOTE = re.compile(
    r"(?:께서|가|이|은|는)?\s*(?:말씀하신|말씀하셨|말한|밝힌|주장한|지적한|강조한)"
)
_PUBLIC_OBSERVATION = re.compile(
    r"(?:국민|유권자|서민|시장|기업|투자자).{0,180}"
    r"(?:받아들이지|믿지|외면|불신|반대|거부).{0,120}"
    r"(?:관측|전망|평가|분석)(?:입니다|이다|됩니다|되고\s*있)"
)
_PARTY_SUPPORT_CLAUSE = re.compile(r"(?:예측|강조|경고|요구|제안|지적|주장|노력)")
_CONTRAST = re.compile(r"(?:하지만|그러나|반면|그런데|했(?:었)?지만|하였지만)")
_GOVERNMENT_NEGATIVE = re.compile(
    r"(?:정부|정권).{0,180}(?:실패|실정|거짓말|무능|미적미적|오락가락|잘못|악화|파탄|부실)"
)
_PRIOR_GOVERNMENT_BLAME = re.compile(
    r"(?:참여|과거|이전|전임|그전의?)\s*정부.{0,180}(?:실정|실패|잘못|책임|거품|악화|만들)"
)
_PARTY_CONCESSION = re.compile(r"(?:도|은|는).{0,80}(?:부인할\s*수\s*없|인정하지\s*않을\s*수\s*없)")
_DEFENSIVE_RESPONSIBILITY = re.compile(
    r"(?:떠안|책임밖에\s*없|책임만\s*있|물려받|초래한\s*것이\s*아니)"
)
_FIRST_PERSON_SUPPORT = re.compile(
    r"(?:저는|제가|본\s*(?:의원|위원|인은)).{0,140}"
    r"(?:100\s*%\s*)?(?:동의|지지|찬성|환영|높이\s*평가|긍정적으로\s*평가)"
)


def _target_in_text(target_name: str, text: str) -> bool:
    target = clean_text(target_name).strip()
    if not target or target in {"none", "정부", "청와대"}:
        return target in text
    return target in text


def _explicit_target_owned_positive(row: Mapping[str, object], current: str) -> bool:
    target_name = clean_text(row.get("target_name", ""))
    return bool(_target_in_text(target_name, current) and _FIRST_PERSON_SUPPORT.search(current))


def resolve_discourse_target_v21(row: Mapping[str, object]) -> dict[str, object]:
    """Resolve owner, assertion, scope, and target match before polarity use."""

    current = clean_text(row.get("text_excerpt", ""))
    before = clean_text(row.get("context_before", ""))
    speaker = clean_text(row.get("speaker", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    target_name = clean_text(row.get("target_name", ""))
    owner = "speaker"
    assertion = "asserted"
    government_scope = "current_or_explicit"
    referent_source = "current_sentence"
    target_match = "matched"
    reasons: list[str] = []

    if _CONDITIONAL_FAILURE.search(current):
        assertion = "conditional"
        reasons.append("conditional_failure_projection_v21")

    if _COMMITTEE_STAFF.search(speaker) and _REPORTED_OPINION_SUMMARY.search(current):
        owner = "reported_hearing_opinion"
        assertion = "reported"
        reasons.append("committee_staff_reports_others_v21")

    if target_type == "person" and _target_in_text(target_name, current):
        target_position = current.find(target_name)
        after_target = current[target_position + len(target_name) : target_position + len(target_name) + 80]
        if _TARGET_SELF_QUOTE.search(after_target):
            owner = "quoted_target"
            target_match = "other_evaluation_object"
            reasons.append("target_person_is_quote_owner_v21")

    if _PUBLIC_OBSERVATION.search(current):
        owner = "reported_public"
        assertion = "reported"
        reasons.append("reported_public_observation_v21")

    if (
        target_type == "government"
        and _DEMONSTRATIVE_REFERENCE.search(current)
        and _NAMED_PRIOR_GOVERNMENT.search(before)
    ):
        government_scope = "inherited_named_historical"
        referent_source = "previous_sentence"
        target_match = "historical_not_generic_current"
        reasons.append("demonstrative_inherits_historical_government_v21")

    if target_type == "party" and _target_in_text(target_name, current):
        target_position = current.find(target_name)
        contrast = _CONTRAST.search(current, target_position)
        if contrast:
            target_clause = current[target_position : contrast.start()]
            other_clause = current[contrast.end() :]
            if _PARTY_SUPPORT_CLAUSE.search(target_clause) and _GOVERNMENT_NEGATIVE.search(other_clause):
                target_match = "government_after_party_contrast"
                reasons.append("party_mentioned_but_government_evaluated_v21")
        if _PRIOR_GOVERNMENT_BLAME.search(current) and _PARTY_CONCESSION.search(
            current[target_position:]
        ):
            target_match = "prior_government_blame_party_concession"
            reasons.append("party_only_concedes_prior_government_blame_v21")

    if target_type == "person" and _target_in_text(target_name, current):
        target_position = current.find(target_name)
        target_window = current[target_position : target_position + 220]
        if _DEFENSIVE_RESPONSIBILITY.search(target_window) and _PRIOR_GOVERNMENT_BLAME.search(current):
            target_match = "target_defended_prior_government_blamed"
            reasons.append("target_defended_by_prior_government_blame_v21")

    return {
        "stance_owner": owner,
        "assertion_status": assertion,
        "government_scope": government_scope,
        "referent_source": referent_source,
        "target_match": target_match,
        "abstention_reasons": tuple(dict.fromkeys(reasons)),
    }


def apply_discourse_target_gate_v21(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Apply frozen V20 plus V21 target resolution to base predictions."""

    output = np.asarray(prediction, dtype=str).astype("<U8")
    if len(frame) != len(output):
        raise ValueError("frame and prediction lengths do not match")
    encoded_reasons: list[str] = []
    encoded_resolution: list[str] = []
    for index, row in enumerate(frame):
        current = clean_text(row.get("text_excerpt", ""))
        reasons = list(contextual_strict_owner_reasons_v20(row))
        if output[index] == "positive" and not (
            _EXPLICIT_FIRST_PERSON_POSITIVE.search(current)
            or _explicit_target_owned_positive(row, current)
        ):
            reasons.append("positive_owner_not_explicit_v20")
        resolution = resolve_discourse_target_v21(row)
        reasons.extend(resolution["abstention_reasons"])
        reasons = list(dict.fromkeys(reasons))
        encoded_reasons.append("|".join(reasons))
        encoded_resolution.append(
            json.dumps(
                {key: value for key, value in resolution.items() if key != "abstention_reasons"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        if reasons:
            output[index] = "neutral"
    return output, encoded_reasons, encoded_resolution
