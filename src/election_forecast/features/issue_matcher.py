"""Issue keyword and phrase matching utilities.

The first assembly-speech prototype used plain substring checks. This module
keeps that behavior for single terms and adds two phrase behaviors:

1. tight phrase matching: tokens may be separated by whitespace, punctuation,
   or short Korean case particles.
2. sentence proximity matching: phrase tokens may be separated by other words
   when they appear in order within the same sentence and within a bounded
   character distance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping


DEFAULT_MAX_PHRASE_GAP_CHARS = 40
SINGLE_TERM_WEIGHT = 0.35
PROXIMITY_PHRASE_WEIGHT = 0.75
TIGHT_PHRASE_WEIGHT = 1.0

_SPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"[\n\r.!?\u3002\uff01\uff1f;；]+")
_KOREAN_PARTICLE = (
    r"(?:"
    r"\uc740|\ub294|\uc774|\uac00|\uc744|\ub97c|\uc758|\uc5d0|\uc5d0\uc11c|"
    r"\ub85c|\uc73c\ub85c|\uc640|\uacfc|\ub3c4|\ub9cc|\ubd80\ud130|\uae4c\uc9c0"
    r")?"
)
_TIGHT_PHRASE_GAP = rf"{_KOREAN_PARTICLE}[\s\W_]*"
_PROXIMITY_BLOCKING_TOKENS = frozenset(
    {
        "정책",
        "지원",
        "발전",
        "안정",
        "개혁",
        "혁신",
        "경쟁",
        "협력",
        "관계",
        "비전",
        "기회",
        "평가",
        "능력",
        "경험",
        "리더십",
        "통합",
        "공정",
    }
)


@dataclass(frozen=True)
class IssueTerm:
    """A compiled issue term."""

    issue_name: str
    raw_term: str
    tokens: tuple[str, ...]
    pattern: re.Pattern[str]
    is_phrase: bool


@dataclass(frozen=True)
class IssueContextRule:
    """A context rule applied after direct issue matching.

    If ``source_issue`` is matched and one of ``context_terms`` appears in the
    same text, the source issue can be multiplied and an optional target issue
    can be emitted. This keeps large background issues such as housing pressure
    connected to regime-responsibility language without turning those context
    words into standalone issue matches.
    """

    source_issue: str
    context_terms: tuple[str, ...]
    source_multiplier: float = 1.0
    target_issue: str | None = None
    target_weight: float = 0.0


@dataclass(frozen=True)
class _IssueMatch:
    issue_name: str
    raw_term: str
    start: int
    end: int
    is_phrase: bool
    weight: float


def normalize_text(value: object) -> str:
    """Normalize text for deterministic matching without language-specific NLP."""

    text = "" if value is None else str(value)
    return _SPACE.sub(" ", text.casefold()).strip()


def term_tokens(term: str) -> list[str]:
    """Split a term into phrase tokens while preserving compact Korean words."""

    normalized = normalize_text(term)
    return [token for token in normalized.split(" ") if token]


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like spans for proximity matching."""

    return [span.strip() for span in _SENTENCE_SPLIT.split(text) if span.strip()]


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Return sentence-like spans with offsets in the normalized text."""

    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        end = match.start()
        span = text[start:end].strip()
        if span:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip())
            right_trim = len(text[start:end].rstrip())
            spans.append((start + left_trim, start + right_trim, span))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        left_trim = len(text[start:]) - len(text[start:].lstrip())
        right_trim = len(text[start:].rstrip())
        spans.append((start + left_trim, start + right_trim, tail))
    return spans


def compile_issue_terms(keyword_map: Mapping[str, Iterable[str]]) -> list[IssueTerm]:
    """Compile issue keywords and phrases into regex matchers.

    A one-token term is matched as a literal substring. A multi-token term is
    treated as a phrase. Tight phrase matching catches forms such as
    ``youth jobs`` and ``youth-jobs``. Proximity matching, applied later, catches
    ordered sentence-level forms such as ``youth unemployment and job support``.
    """

    compiled: list[IssueTerm] = []
    for issue_name, terms in keyword_map.items():
        for raw in terms:
            raw_term = str(raw or "").strip()
            tokens = tuple(term_tokens(raw_term))
            if not tokens:
                continue
            if len(tokens) == 1:
                pattern_text = re.escape(tokens[0])
                is_phrase = False
            else:
                pattern_text = _TIGHT_PHRASE_GAP.join(re.escape(token) for token in tokens)
                is_phrase = True
            compiled.append(
                IssueTerm(
                    issue_name=str(issue_name),
                    raw_term=raw_term,
                    tokens=tokens,
                    pattern=re.compile(pattern_text, flags=re.IGNORECASE),
                    is_phrase=is_phrase,
                )
            )
    return compiled


@lru_cache(maxsize=512)
def _compile_cached(items: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[IssueTerm, ...]:
    return tuple(compile_issue_terms({issue: terms for issue, terms in items}))


def _cache_key(keyword_map: Mapping[str, Iterable[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        sorted(
            (str(issue), tuple(str(term) for term in terms if str(term or "").strip()))
            for issue, terms in keyword_map.items()
        )
    )


def _phrase_tokens_in_span(
    span: str,
    tokens: tuple[str, ...],
    max_gap_chars: int = DEFAULT_MAX_PHRASE_GAP_CHARS,
) -> tuple[int, int] | None:
    """Return phrase span when tokens occur in order within a bounded sentence span."""

    search_from = 0
    previous_end: int | None = None
    first_start: int | None = None
    for token in tokens:
        match = re.search(re.escape(token), span[search_from:], flags=re.IGNORECASE)
        if not match:
            return None
        start = search_from + match.start()
        end = search_from + match.end()
        if previous_end is not None and start - previous_end > max_gap_chars:
            return None
        if first_start is None:
            first_start = start
        previous_end = end
        search_from = end
    if first_start is None or previous_end is None:
        return None
    return first_start, previous_end


def _proximity_phrase_span(
    text: str,
    tokens: tuple[str, ...],
    max_gap_chars: int = DEFAULT_MAX_PHRASE_GAP_CHARS,
) -> tuple[int, int] | None:
    """Return phrase-token span in order within the same sentence-like span."""

    if _requires_tight_phrase(tokens):
        return None
    for start, _, sentence in _sentence_spans(text):
        span = _phrase_tokens_in_span(sentence, tokens, max_gap_chars)
        if span is not None:
            return start + span[0], start + span[1]
    return None


def _requires_tight_phrase(tokens: tuple[str, ...]) -> bool:
    """Avoid proximity matches for broad rhetorical phrase endings.

    Terms such as ``청년 정책`` or ``미래 비전`` are useful when they appear as
    exact phrases, but proximity matching can over-connect unrelated words in a
    long sentence, for example ``청년 지원 정책`` -> ``청년 정책``. Concrete
    issue phrases such as ``청년 실업`` still keep proximity matching.
    """

    return any(token in _PROXIMITY_BLOCKING_TOKENS for token in tokens)


def _overlaps_used(candidate: _IssueMatch, used_spans: list[tuple[int, int]]) -> bool:
    return any(candidate.start < end and candidate.end > start for start, end in used_spans)


def _find_candidates(
    normalized: str,
    keyword_map: Mapping[str, Iterable[str]],
    max_gap_chars: int,
    term_weights: Mapping[tuple[str, str], float] | None = None,
) -> list[_IssueMatch]:
    """Find all candidate matches before resolving overlap."""

    candidates: list[_IssueMatch] = []
    term_weights = term_weights or {}
    for term in _compile_cached(_cache_key(keyword_map)):
        match = term.pattern.search(normalized)
        if match:
            start, end = match.span()
            weight = TIGHT_PHRASE_WEIGHT if term.is_phrase else SINGLE_TERM_WEIGHT
        elif term.is_phrase:
            span = _proximity_phrase_span(normalized, term.tokens, max_gap_chars)
            if span is None:
                continue
            start, end = span
            weight = PROXIMITY_PHRASE_WEIGHT
        else:
            continue
        weight = term_weights.get((term.issue_name, term.raw_term), weight)
        candidates.append(
            _IssueMatch(
                issue_name=term.issue_name,
                raw_term=term.raw_term,
                start=start,
                end=end,
                is_phrase=term.is_phrase,
                weight=weight,
            )
        )
    return candidates


def match_issue_terms(
    text: object,
    keyword_map: Mapping[str, Iterable[str]],
    max_gap_chars: int = DEFAULT_MAX_PHRASE_GAP_CHARS,
) -> dict[str, list[str]]:
    """Return matched raw terms grouped by issue."""

    normalized = normalize_text(text)
    if not normalized:
        return {}
    matches: dict[str, list[str]] = {}
    candidates = _find_candidates(normalized, keyword_map, max_gap_chars)
    candidates.sort(key=lambda item: (not item.is_phrase, item.end - item.start, item.start, item.raw_term))
    used_spans: list[tuple[int, int]] = []
    for candidate in candidates:
        if _overlaps_used(candidate, used_spans):
            continue
        used_spans.append((candidate.start, candidate.end))
        matches.setdefault(candidate.issue_name, []).append(candidate.raw_term)
    return matches


def match_issue_weights(
    text: object,
    keyword_map: Mapping[str, Iterable[str]],
    max_gap_chars: int = DEFAULT_MAX_PHRASE_GAP_CHARS,
    term_weights: Mapping[tuple[str, str], float] | None = None,
    issue_boosts: Mapping[str, float] | None = None,
    context_rules: Iterable[IssueContextRule] | None = None,
) -> dict[str, float]:
    """Return issue-level weights from matched terms.

    Single-token terms are retained as weak evidence. Exact/tight phrases carry
    full evidence, and sentence-proximity phrase matches carry intermediate
    evidence. The maximum matched term weight per issue is used so repeated
    broad words in one speech do not inflate that speech's issue count.
    """

    normalized = normalize_text(text)
    if not normalized:
        return {}
    weights: dict[str, float] = {}
    candidates = _find_candidates(normalized, keyword_map, max_gap_chars, term_weights=term_weights)
    candidates.sort(key=lambda item: (not item.is_phrase, item.end - item.start, item.start, item.raw_term))
    used_spans: list[tuple[int, int]] = []
    for candidate in candidates:
        if _overlaps_used(candidate, used_spans):
            continue
        used_spans.append((candidate.start, candidate.end))
        weights[candidate.issue_name] = max(weights.get(candidate.issue_name, 0.0), candidate.weight)
    if not weights:
        return weights
    for rule in context_rules or ():
        if rule.source_issue not in weights:
            continue
        if not any(normalize_text(term) in normalized for term in rule.context_terms):
            continue
        weights[rule.source_issue] *= rule.source_multiplier
        if rule.target_issue and rule.target_weight > 0:
            weights[rule.target_issue] = max(
                weights.get(rule.target_issue, 0.0),
                weights[rule.source_issue] * rule.target_weight,
            )
    for issue_name, multiplier in (issue_boosts or {}).items():
        if issue_name in weights:
            weights[issue_name] *= float(multiplier)
    return weights


def matched_issues(
    text: object,
    keyword_map: Mapping[str, Iterable[str]],
    max_gap_chars: int = DEFAULT_MAX_PHRASE_GAP_CHARS,
) -> list[str]:
    """Return issue names whose keyword or phrase terms appear in text."""

    return list(match_issue_terms(text, keyword_map, max_gap_chars).keys())
