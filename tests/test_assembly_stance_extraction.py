from __future__ import annotations

from scripts.extract_assembly_stance_rows import classify_stance, resolve_target, split_sentences


def test_stance_labels_are_conservative_for_attack_defense_and_rebuttal() -> None:
    assert classify_stance("정부의 무능과 실패를 강력히 비판합니다.")[0:2] == ("attack", -1)
    assert classify_stance("그 의혹은 사실이 아니며 근거 없는 왜곡입니다.")[0:2] == ("rebuttal", 0)
    assert classify_stance("그 정책은 정당하며 적극 지지합니다.")[0:2] == ("defend", 1)
    assert classify_stance("경제 문제를 논의하겠습니다.")[0:2] == ("neutral", 0)


def test_sentence_split_and_alias_resolution_preserve_explicit_target() -> None:
    sentences = split_sentences("이명박 후보의 의혹을 비판합니다. 경제 문제도 논의합니다.")
    assert len(sentences) == 2
    aliases = [
        type("Alias", (), {"alias": "이명박", "canonical_name": "이명박", "entity_type": "person", "model_eligible": False})(),
    ]
    target = resolve_target(sentences[0], aliases)
    assert target is not None
    assert target.canonical_name == "이명박"
