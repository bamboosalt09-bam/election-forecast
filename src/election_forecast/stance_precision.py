"""Precision-first Korean parliamentary stance classification helpers.

The forecast must not turn neutral discourse into a directional signal, and a
positive/negative reversal is more harmful than abstaining.  This module keeps
those decisions explicit: directionality and polarity are estimated
separately, then a conservative policy can abstain to neutral.  Neutral rows
retain an independent information score.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.stats import beta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


LABELS = np.asarray(["negative", "neutral", "positive"])
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class StanceAdoptionPolicy:
    """Operational gate for a bounded, selective stance overlay.

    Coverage is diagnostic rather than a pass/fail criterion: abstention makes
    the overlay less useful, not less safe. The risk bound is paired with a
    maximum per-row vote-share effect at integration time.
    """

    max_observed_harmful_errors: int = 0
    max_harmful_error_upper_95: float = 0.05
    min_independent_directional_emissions: int = 59
    max_absolute_vote_share_effect: float = 0.001


def stance_adoption_assessment(
    metrics: Mapping[str, float | int],
    *,
    independent_audit: bool,
    target_attribution_audited: bool,
    point_in_time_audited: bool,
    rolling_non_degradation: bool,
    policy: StanceAdoptionPolicy = StanceAdoptionPolicy(),
) -> dict[str, object]:
    """Evaluate evidence and integration checks without a recall threshold."""

    emitted = int(metrics.get("predicted_directional_rows", 0))
    checks = {
        "independent_audit": bool(independent_audit),
        "target_attribution_audited": bool(target_attribution_audited),
        "point_in_time_audited": bool(point_in_time_audited),
        "rolling_non_degradation": bool(rolling_non_degradation),
        "zero_observed_harmful_errors": int(metrics.get("harmful_error_count", -1))
        <= policy.max_observed_harmful_errors,
        "harmful_error_upper_95_within_limit": float(
            metrics.get("harmful_error_upper_95", 1.0)
        )
        <= policy.max_harmful_error_upper_95,
        "enough_independent_directional_emissions": emitted
        >= policy.min_independent_directional_emissions,
    }
    return {
        "classifier_quality_gate_passed": bool(all(checks.values())),
        "checks": checks,
        "quality_gate": {
            **asdict(policy),
            "coverage_is_diagnostic_only": True,
            "coverage_threshold": None,
            "risk_rationale": (
                "The 5% label-risk bound is paired with a 0.1%p absolute "
                "per-row vote-share effect cap."
            ),
        },
    }

_QUESTION = re.compile(r"\?|습니까|나요|는지요|어떻습니까|아닙니까|주시겠습니까")
_QUOTATION = re.compile(r"[\"'‘’“”]|라고\s+(?:말|주장|발언|답변)|다는\s+(?:말|주장|보도)")
_REPORTING = re.compile(r"자료|보고|발표|통계|조사|기록|집계|확인|파악|설명|소명")
_NUMERIC_EVIDENCE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|퍼센트|명|건|원|억|조|년|월|일)")
_PROCEDURAL = re.compile(r"제출|보고해|답변해|확인해|검토해|조사해|소명해|밝혀\s*주")
_NEGATION = re.compile(
    r"아니|않|없|(?<!잘)못(?:\s|하|되|된|할|한|했|했습니다)?|말라|금지|부인|반박"
)
_CONTRAST = re.compile(r"하지만|그러나|그런데|반면|불구하고|오히려|다만")
_SUPPORT = re.compile(r"지지|찬성|환영|긍정적|잘\s*했|잘하고|공감|동의|격려|감사")
_CRITICISM = re.compile(
    r"비판|규탄|사퇴|해임|실패|잘못|무능|부패|불법|위법|책임|문제|우려|혼란|괴롭"
)
_DIRECT_POSITION = re.compile(r"저는|본\s*위원은|우리\s*(?:당|는)|생각합니다|봅니다|입장입니다")
_METALINGUISTIC = re.compile(
    r"예컨대|예를\s*들|가령|라는\s*(?:표현|말|주장)|이라고\s*(?:표현|말|주장)"
)
_EXTERNAL_ATTRIBUTION = re.compile(
    r"(?:야당|언론|전문가|학자|국민|응답자|여론|설문|조사|기관).{0,160}"
    r"(?:비판|주장|평가|대답|응답|분석|전망|지적|단언|보고|질책|분노|답답)"
)
_REPORTED_FRAME = re.compile(
    r"프레임|라고\s*(?:대답|평가|주장|단언)|다는\s*(?:평가|주장|보도)"
)
_TARGET_SELF_POSITION = re.compile(
    r"(?:우리\s*)?(?:정부|정권|청와대).{0,50}"
    r"(?:환영|지지|찬성|반대).{0,40}(?:발표|입장|결의|조치|정책)"
)
_THIRD_PARTY_SUPPORTS_TARGET = re.compile(
    r"^.{0,50}(?:정부|당국|측).{0,80}우리\s*정부.{0,40}(?:환영|지지|찬성|반대)"
)
_TARGET_SELF_REPORT = re.compile(
    r"(?:정부|정권|청와대|대통령).{0,30}(?:가|는|도|에서|께서|께서도).{0,80}"
    r"(?:진단|시인|말하|이야기|평가|설명|발표|지지|환영|기대)"
)
_IMPERSONAL_REPORT = re.compile(
    r"(?:비판|지적|평가).{0,35}(?:많|제기|되고\s*있)|"
    r"(?:지적|비판|평가)이\s*제기"
)
_TARGET_BENEFICIARY = re.compile(r"정부를\s*위해서라도|후보를\s*위해서라도")
_OTHER_GOVERNMENT_ATTACKS = re.compile(
    r"(?:현재|지금|현)\s*(?:정부|정권).{0,50}(?:공격|비판|지적)"
)
_PAST_REGIME_ONLY = re.compile(r"과거\s*정권|전임\s*대통령")
_CURRENT_REGIME_MARKER = re.compile(r"현\s*(?:정부|정권|대통령)|현재\s*(?:정부|정권)")
_THIRD_COUNTRY_SUPPORT = re.compile(
    r"^[가-힣]{2,20}(?:는|은).{0,120}우리\s*정부.{0,60}(?:지지|환영|평가)"
)
_REPORTED_REFERENCE = re.compile(
    r"발언.{0,30}소개|기사.{0,30}내용|매도.{0,20}(?:하고|하).*있"
)
_REBUTTAL_OPENING = re.compile(
    r"^(?:그런|그렇|그러나|하지만).{0,20}(?:것은|사실|아니|이미)"
)
_GOVERNMENT_TARGET = re.compile(
    r"(?<!부)정부|정권|청와대|대통령|국무총리|기획재정부|교육부(?!문)|국토부|정부당국"
)


def clean_text(value: object) -> str:
    return _SPACE.sub(" ", "" if value is None else str(value)).strip()


def _mask_target(text: str, target_name: str, target_alias: str) -> str:
    masked = text
    for value in sorted({target_name, target_alias}, key=len, reverse=True):
        if value:
            masked = masked.replace(value, " [TARGET] ")
    return clean_text(masked)


def compose_precision_input(row: Mapping[str, object], mode: str) -> str:
    """Build a metadata-light current/context representation."""

    target_name = clean_text(row.get("target_name", ""))
    target_alias = clean_text(row.get("target_alias", ""))
    current = _mask_target(clean_text(row.get("text_excerpt", "")), target_name, target_alias)
    before = _mask_target(clean_text(row.get("context_before", "")), target_name, target_alias)
    after = _mask_target(clean_text(row.get("context_after", "")), target_name, target_alias)
    agenda = clean_text(row.get("agenda", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    base = f"[TARGET_TYPE={target_type}] [CURRENT] {current} [CURRENT_REPEAT] {current}"
    if mode == "current_only":
        return base
    if mode == "current_context":
        return f"{base} [BEFORE] {before} [AFTER] {after} [AGENDA] {agenda}"
    if mode == "risk_aware_context":
        risk = bool(_QUESTION.search(current) or _QUOTATION.search(current))
        if risk:
            return base
        return f"{base} [BEFORE] {before} [AFTER] {after} [AGENDA] {agenda}"
    raise ValueError(f"unsupported precision representation: {mode}")


def compose_embedding_input(row: Mapping[str, object], mode: str) -> str:
    """Build natural-language input for a fixed sentence embedding model."""

    target_name = clean_text(row.get("target_name", ""))
    target_alias = clean_text(row.get("target_alias", ""))
    current = _mask_target(clean_text(row.get("text_excerpt", "")), target_name, target_alias)
    if mode == "current_only":
        return current
    if mode != "risk_aware_context":
        raise ValueError(f"unsupported embedding representation: {mode}")
    if risk_flags([current])[0]:
        return current
    before = _mask_target(clean_text(row.get("context_before", "")), target_name, target_alias)
    after = _mask_target(clean_text(row.get("context_after", "")), target_name, target_alias)
    parts = [part for part in (before, current, after) if part]
    return " [CONTEXT] ".join(parts)


def ambiguity_abstention_reasons(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return hard abstention reasons for target/ownership ambiguity.

    These checks run after a directional model. They intentionally sacrifice
    recall when the text does not establish that the speaker owns a stance
    toward the assigned target.
    """

    current = clean_text(row.get("text_excerpt", ""))
    after = clean_text(row.get("context_after", ""))
    target_type = clean_text(row.get("target_type", "none")) or "none"
    target_name = clean_text(row.get("target_name", ""))
    target_alias = clean_text(row.get("target_alias", ""))
    reasons: list[str] = []

    if target_type == "government":
        target_explicit = bool(_GOVERNMENT_TARGET.search(current))
    else:
        target_explicit = any(
            value and value in current for value in (target_name, target_alias)
        )
    if not target_explicit:
        reasons.append("target_not_explicit")
    if _QUESTION.search(current):
        reasons.append("question")
    if _QUOTATION.search(current):
        reasons.append("quotation_owner_unknown")
    if _EXTERNAL_ATTRIBUTION.search(current) or _REPORTED_FRAME.search(current):
        reasons.append("external_or_reported_stance")
    if _IMPERSONAL_REPORT.search(current):
        reasons.append("impersonal_reported_stance")
    if target_type == "government" and _TARGET_SELF_POSITION.search(current):
        reasons.append("target_self_position")
    if target_type == "government" and _THIRD_PARTY_SUPPORTS_TARGET.search(current):
        reasons.append("third_party_supports_target_policy")
    if target_type == "government" and _THIRD_COUNTRY_SUPPORT.search(current):
        reasons.append("third_party_supports_target_policy")
    if target_type == "government" and _TARGET_SELF_REPORT.search(current):
        reasons.append("target_self_report")
    if (
        target_type == "person"
        and target_name
        and re.search(
            rf"{re.escape(target_name)}\s*(?:정부|정권).{{0,20}}(?:진단|시인|말하|평가|설명|발표)",
            current,
        )
    ):
        reasons.append("target_self_report")
    if target_type == "government" and re.search(
        r"우리\s*정부의.{0,100}설명해\s*왔", current
    ):
        reasons.append("target_self_report")
    if target_type == "person" and _TARGET_BENEFICIARY.search(current):
        reasons.append("target_is_beneficiary_not_object")
    if target_type == "person" and _OTHER_GOVERNMENT_ATTACKS.search(current):
        reasons.append("other_government_owns_stance")
    if (
        target_type == "government"
        and _PAST_REGIME_ONLY.search(current)
        and not _CURRENT_REGIME_MARKER.search(current)
    ):
        reasons.append("past_regime_not_assigned_target")
    if _REPORTED_REFERENCE.search(current):
        reasons.append("reported_reference")
    if (
        target_type == "person"
        and target_name
        and re.search(
            rf"{re.escape(target_name)}.*?보다.{{0,50}}(?:현재|지금|[가-힣]+\s*정부)",
            current,
        )
    ):
        reasons.append("comparison_targets_other_actor")
    if (
        _REBUTTAL_OPENING.search(after)
        and re.search(r"실패|잘못|못\s*한다|무능", current)
    ):
        reasons.append("claim_rebutted_in_following_context")
    return tuple(dict.fromkeys(reasons))


def apply_ambiguity_abstention(
    frame: Sequence[Mapping[str, object]], prediction: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Force model output to neutral whenever a hard ambiguity is present."""

    output = np.asarray(prediction, dtype=str).astype("<U8")
    if len(frame) != len(output):
        raise ValueError("frame and prediction lengths do not match")
    encoded_reasons: list[str] = []
    for index, row in enumerate(frame):
        reasons = ambiguity_abstention_reasons(row)
        encoded_reasons.append("|".join(reasons))
        if reasons:
            output[index] = "neutral"
    return output, encoded_reasons


class KoreanPrecisionStructureFeatures(BaseEstimator, TransformerMixin):
    """Small UTF-8-safe structural feature block for Korean hearing language."""

    _patterns = (
        _QUESTION,
        _QUOTATION,
        _REPORTING,
        _NUMERIC_EVIDENCE,
        _PROCEDURAL,
        _NEGATION,
        _CONTRAST,
        _SUPPORT,
        _CRITICISM,
        _DIRECT_POSITION,
    )

    def fit(self, x: Iterable[object], y: object = None) -> "KoreanPrecisionStructureFeatures":
        return self

    def transform(self, x: Iterable[object]) -> sparse.csr_matrix:
        rows: list[list[float]] = []
        for value in x:
            text = clean_text(value)
            length = max(len(text), 1)
            counts = [len(pattern.findall(text)) for pattern in self._patterns]
            rows.append(
                [
                    min(np.log1p(length) / np.log1p(800.0), 1.0),
                    min(text.count(",") / 10.0, 1.0),
                    min(text.count(".") / 5.0, 1.0),
                    *[min(count / 3.0, 1.0) for count in counts],
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))


def build_precision_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    min_df=2,
                    max_df=0.995,
                    max_features=60_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    token_pattern=r"(?u)\b\w+\b",
                    min_df=2,
                    max_df=0.995,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            ("structure", KoreanPrecisionStructureFeatures()),
        ]
    )


@dataclass(frozen=True)
class PrecisionPolicy:
    direction_threshold: float
    polarity_threshold: float
    risk_surcharge: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ConsensusPolicy:
    direction_threshold: float
    polarity_threshold: float
    sign_margin: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def risk_flags(texts: Sequence[object]) -> np.ndarray:
    flags: list[bool] = []
    for value in texts:
        text = clean_text(value)
        flags.append(
            bool(
                _QUOTATION.search(text)
                or _QUESTION.search(text)
                or _METALINGUISTIC.search(text)
                or (_NEGATION.search(text) and (_SUPPORT.search(text) or _CRITICISM.search(text)))
                or (_SUPPORT.search(text) and _CRITICISM.search(text))
            )
        )
    return np.asarray(flags, dtype=bool)


def apply_precision_policy(
    direction_probability: np.ndarray,
    polarity_probabilities: np.ndarray,
    texts: Sequence[object],
    policy: PrecisionPolicy,
) -> np.ndarray:
    """Emit direction only when both direction and sign clear their gates."""

    p_direction = np.asarray(direction_probability, dtype=float)
    p_polarity = np.asarray(polarity_probabilities, dtype=float)
    if p_polarity.ndim != 2 or p_polarity.shape[1] != 2:
        raise ValueError("polarity probabilities must have negative/positive columns")
    risky = risk_flags(texts)
    required_direction = np.minimum(
        policy.direction_threshold + risky.astype(float) * policy.risk_surcharge,
        0.999,
    )
    polarity_best = p_polarity.max(axis=1)
    eligible = (p_direction >= required_direction) & (
        polarity_best >= policy.polarity_threshold
    )
    prediction = np.full(len(p_direction), "neutral", dtype="<U8")
    sign = np.where(p_polarity[:, 0] >= p_polarity[:, 1], "negative", "positive")
    prediction[eligible] = sign[eligible]
    return prediction


def combine_precision_children(
    first_prediction: Sequence[str],
    second_prediction: Sequence[str],
    first_direction_probability: np.ndarray,
    second_direction_probability: np.ndarray,
    first_polarity_probabilities: np.ndarray,
    second_polarity_probabilities: np.ndarray,
    texts: Sequence[object],
    policy: ConsensusPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Union conservative children and add only non-risk consensus rows."""

    first = np.asarray(first_prediction, dtype=str)
    second = np.asarray(second_prediction, dtype=str)
    first_polarity = np.asarray(first_polarity_probabilities, dtype=float)
    second_polarity = np.asarray(second_polarity_probabilities, dtype=float)
    if first_polarity.shape != second_polarity.shape or first_polarity.shape[1] != 2:
        raise ValueError("child polarity arrays must have negative/positive columns")
    if len(first) != len(second) or len(first) != len(first_polarity):
        raise ValueError("child prediction lengths do not match")

    conflict = (first != "neutral") & (second != "neutral") & (first != second)
    combined = np.where(first != "neutral", first, second).astype("<U8")
    combined[conflict] = "neutral"
    source = np.where(
        conflict,
        "conflict_abstain",
        np.where(
            (first != "neutral") & (second != "neutral"),
            "both_conservative",
            np.where(first != "neutral", "first_conservative", np.where(second != "neutral", "second_conservative", "neutral")),
        ),
    ).astype("<U20")

    first_sign = np.where(first_polarity[:, 0] >= first_polarity[:, 1], "negative", "positive")
    second_sign = np.where(second_polarity[:, 0] >= second_polarity[:, 1], "negative", "positive")
    sign_agreement = first_sign == second_sign
    direction_score = np.sqrt(
        np.asarray(first_direction_probability, dtype=float)
        * np.asarray(second_direction_probability, dtype=float)
    )
    polarity_score = np.minimum(
        first_polarity.max(axis=1),
        second_polarity.max(axis=1),
    )
    sign_margin = np.minimum(
        np.abs(first_polarity[:, 1] - first_polarity[:, 0]),
        np.abs(second_polarity[:, 1] - second_polarity[:, 0]),
    )
    consensus = (
        (combined == "neutral")
        & ~conflict
        & ~risk_flags(texts)
        & sign_agreement
        & (direction_score >= policy.direction_threshold)
        & (polarity_score >= policy.polarity_threshold)
        & (sign_margin >= policy.sign_margin)
    )
    combined[consensus] = first_sign[consensus]
    source[consensus] = "consensus"
    return combined, source


def precision_first_metrics(truth: Sequence[str], prediction: Sequence[str]) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=str)
    pred = np.asarray(prediction, dtype=str)
    neutral = y == "neutral"
    directional = ~neutral
    predicted_directional = pred != "neutral"
    neutral_to_direction = neutral & predicted_directional
    wrong_direction = directional & predicted_directional & (pred != y)
    direction_to_neutral = directional & ~predicted_directional
    correct_direction = directional & (pred == y)
    harmful_error_count = int(neutral_to_direction.sum() + wrong_direction.sum())
    emitted_count = int(predicted_directional.sum())
    if emitted_count == 0:
        harmful_error_upper_95 = 1.0
    elif harmful_error_count >= emitted_count:
        harmful_error_upper_95 = 1.0
    else:
        harmful_error_upper_95 = float(
            beta.ppf(0.95, harmful_error_count + 1, emitted_count - harmful_error_count)
        )
    return {
        "n": int(len(y)),
        "neutral_rows": int(neutral.sum()),
        "directional_rows": int(directional.sum()),
        "predicted_directional_rows": int(predicted_directional.sum()),
        "neutral_to_direction_count": int(neutral_to_direction.sum()),
        "neutral_to_direction_rate": float(
            neutral_to_direction.sum() / max(int(neutral.sum()), 1)
        ),
        "wrong_direction_count": int(wrong_direction.sum()),
        "wrong_direction_rate": float(
            wrong_direction.sum() / max(int(directional.sum()), 1)
        ),
        "direction_to_neutral_count": int(direction_to_neutral.sum()),
        "direction_to_neutral_rate": float(
            direction_to_neutral.sum() / max(int(directional.sum()), 1)
        ),
        "correct_direction_count": int(correct_direction.sum()),
        "correct_direction_coverage": float(
            correct_direction.sum() / max(int(directional.sum()), 1)
        ),
        "directional_precision": float(
            correct_direction.sum() / max(int(predicted_directional.sum()), 1)
        ),
        "harmful_error_count": harmful_error_count,
        "harmful_error_rate_among_emitted": float(
            harmful_error_count / max(emitted_count, 1)
        ),
        "harmful_error_upper_95": harmful_error_upper_95,
    }


def neutral_information_features(text: object) -> dict[str, float | str]:
    """Retain informational value without assigning a political direction."""

    value = clean_text(text)
    numeric = min(len(_NUMERIC_EVIDENCE.findall(value)) / 2.0, 1.0)
    reporting = min(len(_REPORTING.findall(value)) / 2.0, 1.0)
    procedural = min(len(_PROCEDURAL.findall(value)) / 2.0, 1.0)
    analytical = min((len(_CONTRAST.findall(value)) + value.count("때문")) / 2.0, 1.0)
    length = min(np.log1p(len(value)) / np.log1p(500.0), 1.0)
    question_only = bool(_QUESTION.search(value)) and len(value) < 35
    score = float(
        np.clip(
            0.30 * numeric
            + 0.25 * reporting
            + 0.15 * procedural
            + 0.20 * analytical
            + 0.10 * length
            - (0.25 if question_only else 0.0),
            0.0,
            1.0,
        )
    )
    categories = {
        "evidence": numeric,
        "reporting": reporting,
        "procedural": procedural,
        "analysis": analytical,
    }
    category = max(categories, key=categories.get) if max(categories.values()) > 0 else "general"
    return {
        "neutral_information_score": score,
        "neutral_information_category": category,
        "neutral_numeric_evidence": numeric,
        "neutral_reporting_signal": reporting,
        "neutral_procedural_signal": procedural,
        "neutral_analysis_signal": analytical,
    }
