from __future__ import annotations

from election_forecast.features.issue_matcher import (
    IssueContextRule,
    matched_issues,
    match_issue_terms,
    match_issue_weights,
)
from news_collector.sources.assembly_records import _matched_issues


def test_phrase_match_accepts_whitespace_and_punctuation() -> None:
    keyword_map = {"housing": ["housing supply"], "jobs": ["youth jobs"]}

    assert matched_issues("The housing, supply plan was discussed.", keyword_map) == ["housing"]


def test_phrase_match_accepts_korean_particle_between_terms() -> None:
    keyword_map = {"jobs_labor": ["\uccad\ub144 \uc2e4\uc5c5"]}

    out = match_issue_terms("\uccad\ub144\uc758 \uc2e4\uc5c5 \ub300\ucc45\uc744 \ub17c\uc758", keyword_map)

    assert out == {"jobs_labor": ["\uccad\ub144 \uc2e4\uc5c5"]}


def test_phrase_only_does_not_match_single_component() -> None:
    keyword_map = {"housing": ["housing supply"]}

    assert matched_issues("housing prices were discussed", keyword_map) == []


def test_phrase_match_accepts_ordered_tokens_separated_in_sentence() -> None:
    keyword_map = {"jobs_labor": ["youth unemployment"]}

    text = "The youth support hearing focused on severe unemployment this year."

    assert matched_issues(text, keyword_map) == ["jobs_labor"]


def test_rhetorical_phrase_requires_tight_match_for_proximity() -> None:
    keyword_map = {"gender_generation": ["\uccad\ub144 \uc815\ucc45"]}

    loose_text = (
        "\uccad\ub144 \uc9c0\uc6d0 \uc815\ucc45\uacfc "
        "\uc77c\ubc18 \uc0b0\uc5c5 \uc815\ucc45\uc744 \uac19\uc774 \ub17c\uc758\ud588\ub2e4."
    )
    tight_text = "\uccad\ub144 \uc815\ucc45\uc744 \ub17c\uc758\ud588\ub2e4."

    assert matched_issues(loose_text, keyword_map) == []
    assert matched_issues(tight_text, keyword_map) == ["gender_generation"]


def test_concrete_phrase_still_allows_sentence_proximity() -> None:
    keyword_map = {"jobs_labor": ["\uccad\ub144 \uc2e4\uc5c5"]}

    text = "\uccad\ub144 \uc9c0\uc6d0\uacfc \uc2ec\uac01\ud55c \uc2e4\uc5c5 \ubb38\uc81c\ub97c \ub17c\uc758\ud588\ub2e4."

    assert matched_issues(text, keyword_map) == ["jobs_labor"]


def test_phrase_match_does_not_cross_sentence_boundary() -> None:
    keyword_map = {"jobs_labor": ["youth unemployment"]}

    text = "Youth support was discussed. Unemployment also came up later."

    assert matched_issues(text, keyword_map) == []


def test_phrase_match_respects_max_gap() -> None:
    keyword_map = {"jobs_labor": ["youth unemployment"]}

    text = "Youth " + ("very " * 20) + "unemployment"

    assert matched_issues(text, keyword_map, max_gap_chars=20) == []


def test_phrase_match_consumes_overlapping_single_term() -> None:
    keyword_map = {
        "jobs_labor": ["youth unemployment"],
        "generic": ["unemployment"],
    }

    out = match_issue_terms("Youth unemployment was discussed.", keyword_map)

    assert out == {"jobs_labor": ["youth unemployment"]}


def test_shorter_overlapping_phrase_wins() -> None:
    keyword_map = {
        "broad": ["youth unemployment"],
        "narrow": ["unemployment problem"],
    }

    text = "Youth policy and unemployment problem were discussed."

    assert match_issue_terms(text, keyword_map) == {"narrow": ["unemployment problem"]}


def test_assembly_records_uses_phrase_matcher() -> None:
    keyword_map = {"regional_dev": ["regional development"]}

    assert _matched_issues("regional-development pledge", keyword_map) == ["regional_dev"]


def test_issue_boosts_multiply_matched_issue_weight() -> None:
    keyword_map = {"housing": ["\ubd80\ub3d9\uc0b0"]}

    weights = match_issue_weights(
        "\ubd80\ub3d9\uc0b0 \ubb38\uc81c\ub97c \ub17c\uc758\ud588\ub2e4.",
        keyword_map,
        issue_boosts={"housing": 1.2},
    )

    assert weights["housing"] == 0.35 * 1.2


def test_context_rule_links_housing_to_regime_change_only_when_context_appears() -> None:
    keyword_map = {"housing": ["\ubd80\ub3d9\uc0b0"]}
    rule = IssueContextRule(
        source_issue="housing",
        context_terms=("\uc815\uad8c", "\uc815\ubd80"),
        source_multiplier=1.1,
        target_issue="regime_change",
        target_weight=0.5,
    )

    plain = match_issue_weights("\ubd80\ub3d9\uc0b0 \uacf5\uae09 \ub300\ucc45", keyword_map, context_rules=[rule])
    contextual = match_issue_weights(
        "\uc815\ubd80\uc758 \ubd80\ub3d9\uc0b0 \uc2e4\uc815\uacfc \uc815\uad8c \ucc45\uc784",
        keyword_map,
        context_rules=[rule],
    )

    assert "regime_change" not in plain
    assert contextual["housing"] == 0.35 * 1.1
    assert contextual["regime_change"] == contextual["housing"] * 0.5
