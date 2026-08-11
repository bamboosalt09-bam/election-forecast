"""Representation, ownership, and calibration helpers for stance v3."""

from __future__ import annotations

import re

import numpy as np


ISSUE_CHARACTER_GAIN = 0.24
_SPACE = re.compile(r"\s+")
_RISK = re.compile(
    r"[‘’“”\"']|라고|다고|라는|이라며|말했|주장|보도|전했|회신|"
    r"아니|않|없|못|하지만|그러나|그런데|반면|불구하고"
)
_META_EXAMPLE = re.compile(r"(?:예컨대|예를\s*들어|가령).{0,180}(?:지지|찬성|반대|비판|비방|공격)")
_NEGATED_RECEPTION = re.compile(
    r"(?:비판|비방|공격)(?:한다고|하는\s*것으로)(?:만)?\s*(?:받아들이지|보지|생각하지)"
)
_DENIED_REPORTED_STANCE = re.compile(
    r"(?:지지|찬성|반대|비판|비방)(?:한다고|한다는|했다는).{0,45}"
    r"(?:말한\s*적이\s*없|얘기하지|이야기하지|사실이\s*아니|부인)"
)
_REPORTED_SURVEY = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%|퍼센트|조사|데이터).{0,80}"
    r"(?:지지|찬성|반대).{0,45}(?:나왔|조사|집계|응답|모인)"
)
_STANCE_CAUSAL_MENTION = re.compile(r"(?:지지|찬성|반대|비판|비방)한다고\s*해서")
_HEARSAY_ONLY = re.compile(r"(?:요구|주장|발언).{0,45}(?:이야기를\s*들었습니다|전해졌습니다)")


def _clean(value: object) -> str:
    return _SPACE.sub(" ", "" if value is None else str(value)).strip()


def _mask(text: str, target_name: str, target_alias: str) -> str:
    masked = text
    for value in sorted({target_name, target_alias}, key=len, reverse=True):
        if value:
            masked = masked.replace(value, " [TARGET] ")
    return _clean(masked)


def compose_v3_input(row: dict[str, object], mode: str) -> str:
    """Compose current-only or single-nearest-context input."""
    target_name = _clean(row.get("target_name", ""))
    target_alias = _clean(row.get("target_alias", ""))
    current = _mask(_clean(row.get("text_excerpt", "")), target_name, target_alias)
    before = _mask(_clean(row.get("context_before", "")), target_name, target_alias)
    after = _mask(_clean(row.get("context_after", "")), target_name, target_alias)
    target_type = _clean(row.get("target_type", "none")) or "none"
    base = f"[TARGET_TYPE={target_type}] [CURRENT] {current} [CURRENT_REPEAT] {current}"
    if mode == "current_only":
        return base
    if mode not in {"nearest_context", "risk_aware_nearest"}:
        raise ValueError(f"unsupported v3 representation: {mode}")
    if mode == "risk_aware_nearest" and _RISK.search(current):
        return base
    candidates: list[tuple[float, str, str]] = []
    if before:
        try:
            gap = float(row.get("context_gap_before", "") or 1e9)
        except (TypeError, ValueError):
            gap = 1e9
        candidates.append((gap, "BEFORE", before))
    if after:
        try:
            gap = float(row.get("context_gap_after", "") or 1e9)
        except (TypeError, ValueError):
            gap = 1e9
        candidates.append((gap, "AFTER", after))
    if not candidates:
        return base
    _, side, context = min(candidates, key=lambda item: (item[0], item[1]))
    return f"{base} [NEAREST_{side}] {context}"


def ownership_abstention(text: object, prediction: str) -> tuple[str, str]:
    """Neutralize explicit stance mentions that do not belong to the speaker."""
    value = _clean(text)
    if prediction == "neutral":
        return prediction, "already_neutral"
    patterns = (
        ("metalinguistic_example", _META_EXAMPLE),
        ("negated_reception", _NEGATED_RECEPTION),
        ("denied_reported_stance", _DENIED_REPORTED_STANCE),
        ("reported_survey", _REPORTED_SURVEY),
        ("stance_causal_mention", _STANCE_CAUSAL_MENTION),
        ("hearsay_only", _HEARSAY_ONLY),
    )
    for reason, pattern in patterns:
        if pattern.search(value):
            return "neutral", reason
    return prediction, "retained"


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def directional_abstention(
    probabilities: np.ndarray,
    classes: list[str] | np.ndarray,
    *,
    min_probability: float,
    min_margin: float,
) -> np.ndarray:
    classes_array = np.asarray(classes)
    order = np.argsort(probabilities, axis=1)
    best_indices = order[:, -1]
    second_indices = order[:, -2]
    predictions = classes_array[best_indices].astype("<U8", copy=True)
    best = probabilities[np.arange(len(probabilities)), best_indices]
    margin = best - probabilities[np.arange(len(probabilities)), second_indices]
    abstain = (predictions != "neutral") & (
        (best < min_probability) | (margin < min_margin)
    )
    predictions[abstain] = "neutral"
    return predictions


def classify_issue_character(
    negative_share: float,
    neutral_share: float,
    positive_share: float,
    *,
    confidence_quality: float,
) -> dict[str, float | str]:
    """Score aggregate discourse continuously without assigning a beneficiary."""
    total = float(negative_share + neutral_share + positive_share)
    if total <= 0.0:
        negative_share, neutral_share, positive_share = 0.0, 1.0, 0.0
    else:
        negative_share /= total
        neutral_share /= total
        positive_share /= total
    quality = float(np.clip(confidence_quality, 0.0, 1.0))
    directional_share = negative_share + positive_share
    directional_balance = positive_share - negative_share
    polarization = 2.0 * min(negative_share, positive_share)

    accountability_score = max(negative_share - positive_share, 0.0)
    performance_score = max(positive_share - negative_share, 0.0)
    informational_score = neutral_share
    polarized_score = polarization
    mixed_score = directional_share * (1.0 - abs(directional_balance))

    diagnostic_scores = {
        "informational_context": informational_score,
        "negative_accountability": accountability_score,
        "positive_performance": performance_score,
        "polarized_contest": polarized_score,
        "mixed_evaluation": mixed_score * 0.50,
    }
    name = max(diagnostic_scores, key=diagnostic_scores.get)

    # The category above is diagnostic only. This continuous score changes the
    # issue magnitude used by the forecasting overlay.
    character_score = (
        -0.35 * informational_score
        + 0.80 * accountability_score
        + 0.50 * performance_score
        + 1.00 * polarized_score
        + 0.15 * directional_share
    )
    character_score = float(np.clip(character_score, -0.35, 1.15))
    multiplier = float(
        np.clip(1.0 + ISSUE_CHARACTER_GAIN * quality * character_score, 0.88, 1.24)
    )

    return {
        "issue_character": name,
        "character_score": character_score,
        "character_intensity": float(min(abs(character_score), 1.0)),
        "character_multiplier": multiplier,
        "informational_score": float(informational_score),
        "accountability_score": float(accountability_score),
        "performance_score": float(performance_score),
        "polarized_score": float(polarized_score),
        "mixed_score": float(mixed_score),
        "directional_share": float(directional_share),
        "directional_balance": float(directional_balance),
        "polarization": float(polarization),
    }
