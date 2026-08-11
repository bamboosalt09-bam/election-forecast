"""Metadata-free quantitative sentence stance classifier components."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


LABELS = np.asarray(["negative", "neutral", "positive"])

_LEGACY_PATTERNS = {
    "rebuttal": ("사실이 아니", "근거 없는", "근거없", "왜곡이다", "허위다", "반박한다"),
    "defend": ("변호한다", "옹호한다", "정당하다", "정당하며", "문제가 없다", "책임이 없다", "잘못이 없다"),
    "endorse": ("지지한다", "지지합니다", "적극 지지", "찬성한다", "찬성합니다", "환영한다", "환영합니다", "높이 평가한다"),
    "attack": ("비판한다", "강력히 비판", "규탄한다", "사퇴해야", "퇴진해야", "무능하다", "정책실패", "잘못했다", "부정부패", "불법이다", "은폐했다"),
}

_SPACE = re.compile(r"\s+")
_PATTERNS = {
    "question": re.compile(r"[?？]|(?:습니까|나요|아닌가요|것 아니|겠습니까)"),
    "quotation": re.compile(r"[‘’“”\"']|(?:라고|다고|라는|이라며|말했|밝혔|주장했)"),
    "conditional": re.compile(r"(?:면|다면|경우|가정|전제|라면)"),
    "negation": re.compile(r"(?:아니|않|없|못|말라|금지)"),
    "contrast": re.compile(r"(?:하지만|그러나|그런데|반면|불구하고|다만|오히려)"),
    "first_person": re.compile(r"(?:저는|제가|본 위원|우리 당|저희는|생각합니다|봅니다)"),
    "reporting": re.compile(r"(?:보도|전했|회신|발표|기록|자료|판결|의견|답변)"),
    "support": re.compile(r"(?:지지|찬성|환영|높이 평가|격려|감사|뜻깊|노고)"),
    "defense": re.compile(r"(?:문제가 없|책임이 없|잘못이 없|정당하|옹호|변호)"),
    "criticism": re.compile(
        r"(?:비판|규탄|사퇴|퇴진|무능|실패|잘못|부정부패|불법|은폐|심각|유감|책임)"
    ),
    "scope_reversal": re.compile(
        r"(?:문제가 없다고|책임이 없다고|잘못이 없다고|지지한다고|찬성한다고|비판한다고)"
    ),
}


def normalize_text(value: object) -> str:
    return _SPACE.sub(" ", "" if value is None else str(value)).strip()


class KoreanStanceStructureFeatures(BaseEstimator, TransformerMixin):
    """Extract compact structural counts without using row metadata."""

    def fit(self, x: Iterable[object], y: object = None) -> "KoreanStanceStructureFeatures":
        return self

    def transform(self, x: Iterable[object]) -> sparse.csr_matrix:
        rows: list[list[float]] = []
        for value in x:
            text = normalize_text(value)
            length = max(len(text), 1)
            counts = [len(pattern.findall(text)) for pattern in _PATTERNS.values()]
            rows.append(
                [
                    min(np.log1p(length) / np.log1p(600.0), 1.0),
                    min(text.count(",") / 8.0, 1.0),
                    min(text.count(".") / 4.0, 1.0),
                    *[min(count / 3.0, 1.0) for count in counts],
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))


def build_stance_pipeline(*, c_value: float = 1.0) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_df=0.995,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w+\b",
                    min_df=2,
                    max_df=0.995,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            ("structure", KoreanStanceStructureFeatures()),
        ]
    )
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=3_000,
        solver="lbfgs",
        random_state=20260714,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def apply_directional_abstention(
    probabilities: np.ndarray,
    classes: Sequence[str],
    *,
    min_directional_probability: float,
    min_probability_margin: float,
) -> np.ndarray:
    classes_array = np.asarray(classes)
    order = np.argsort(probabilities, axis=1)
    best_indices = order[:, -1]
    second_indices = order[:, -2]
    predictions = classes_array[best_indices].copy()
    best = probabilities[np.arange(len(probabilities)), best_indices]
    second = probabilities[np.arange(len(probabilities)), second_indices]
    abstain = (predictions != "neutral") & (
        (best < min_directional_probability) | ((best - second) < min_probability_margin)
    )
    predictions[abstain] = "neutral"
    return predictions


def label_to_polarity(label: str) -> int:
    return {"negative": -1, "neutral": 0, "positive": 1}[label]


def legacy_rule_label(text: object) -> str:
    compact = normalize_text(text).replace(" ", "")
    active = {
        label
        for label, cues in _LEGACY_PATTERNS.items()
        if any(cue.replace(" ", "") in compact for cue in cues)
    }
    if not active or "rebuttal" in active:
        return "neutral"
    if "attack" in active and ({"defend", "endorse"} & active):
        return "neutral"
    if "attack" in active:
        return "negative"
    if "defend" in active or "endorse" in active:
        return "positive"
    return "neutral"


def apply_rule_correction(
    probabilities: np.ndarray,
    classes: Sequence[str],
    legacy_predictions: Sequence[str],
    *,
    min_override_probability: float,
    min_probability_margin: float,
    allow_neutral_source: bool = False,
) -> np.ndarray:
    classes_array = np.asarray(classes)
    legacy = np.asarray(legacy_predictions).astype("<U8", copy=True)
    order = np.argsort(probabilities, axis=1)
    best_indices = order[:, -1]
    second_indices = order[:, -2]
    model_predictions = classes_array[best_indices]
    best = probabilities[np.arange(len(probabilities)), best_indices]
    second = probabilities[np.arange(len(probabilities)), second_indices]
    override = (
        (model_predictions != legacy)
        & (best >= min_override_probability)
        & ((best - second) >= min_probability_margin)
    )
    if not allow_neutral_source:
        override &= legacy != "neutral"
    legacy[override] = model_predictions[override]
    return legacy
