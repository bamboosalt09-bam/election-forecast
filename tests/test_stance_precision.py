import numpy as np
import pandas as pd
import pytest

from election_forecast.stance_precision import (
    PrecisionPolicy,
    ConsensusPolicy,
    apply_precision_policy,
    ambiguity_abstention_reasons,
    apply_ambiguity_abstention,
    compose_embedding_input,
    combine_precision_children,
    compose_precision_input,
    neutral_information_features,
    precision_first_metrics,
    stance_adoption_assessment,
)
from election_forecast.stance_context_v15 import (
    apply_contextual_ownership_gate_v15,
    target_owned_report_reason,
)
from election_forecast.stance_context_v16 import (
    apply_contextual_attribution_gate_v16,
)
from election_forecast.stance_context_v17 import (
    apply_contextual_speaker_scope_gate_v17,
)
from scripts.build_stance_context_consensus import majority_consensus
from scripts.train_stance_precision_first import _load_gold
from scripts.train_stance_precision_augmented import _load_manual
from scripts.apply_stance_precision_ensemble import validate_shadow_corpus


def test_compose_precision_input_masks_target_in_current_and_context() -> None:
    row = {
        "target_type": "person",
        "target_name": "홍길동",
        "target_alias": "길동 후보",
        "text_excerpt": "홍길동 후보의 정책을 검토했습니다.",
        "context_before": "길동 후보가 발표했습니다.",
        "context_after": "홍길동에게 자료 제출을 요구합니다.",
        "agenda": "정책 검증",
    }

    value = compose_precision_input(row, "current_context")

    assert "홍길동" not in value
    assert "길동 후보" not in value
    assert value.count("[TARGET]") >= 3


def test_context_majority_consensus_requires_two_matching_directions() -> None:
    predictions = np.asarray(
        [
            ["negative", "negative", "neutral"],
            ["positive", "neutral", "positive"],
            ["negative", "positive", "neutral"],
            ["neutral", "neutral", "negative"],
        ]
    )
    assert majority_consensus(predictions).tolist() == [
        "negative",
        "positive",
        "neutral",
        "neutral",
    ]


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "text_excerpt": "전문가들이 정부 정책 실패라고 평가했습니다.",
            },
            "external_or_reported_stance",
        ),
        (
            {
                "target_type": "person",
                "target_name": "홍길동",
                "text_excerpt": "홍길동 후보의 정책 실패라고 생각하십니까?",
            },
            "question",
        ),
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "text_excerpt": "부정부패가 심각합니다.",
            },
            "target_not_explicit",
        ),
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "text_excerpt": "체코 정부는 우리 정부의 정책을 적극 지지하고 있습니다.",
            },
            "third_party_supports_target_policy",
        ),
        (
            {
                "target_type": "person",
                "target_name": "박근혜",
                "text_excerpt": "박근혜정부보다도 현재 문재인 정부의 정책이 원인입니다.",
            },
            "comparison_targets_other_actor",
        ),
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "text_excerpt": "정부는 경제가 어렵다고 진단했습니다.",
            },
            "target_self_report",
        ),
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "text_excerpt": "네덜란드는 우리 정부의 노력을 지지하고 있습니다.",
            },
            "third_party_supports_target_policy",
        ),
        (
            {
                "target_type": "person",
                "target_name": "이명박",
                "text_excerpt": "측근의 발언을 기사에서 소개하고 있어요.",
            },
            "reported_reference",
        ),
        (
            {
                "target_type": "person",
                "target_name": "이명박",
                "text_excerpt": "지금 정부가 이명박 정부를 공격합니다.",
            },
            "other_government_owns_stance",
        ),
        (
            {
                "target_type": "person",
                "target_name": "문재인",
                "text_excerpt": "문재인 정부는 경제가 어렵다고 진단했습니다.",
            },
            "target_self_report",
        ),
    ],
)
def test_ambiguity_reasons_force_risky_stances_to_neutral(row, reason) -> None:
    reasons = ambiguity_abstention_reasons(row)
    prediction, encoded = apply_ambiguity_abstention([row], ["negative"])
    assert reason in reasons
    assert prediction.tolist() == ["neutral"]
    assert reason in encoded[0]


def test_ambiguity_gate_preserves_explicit_direct_criticism() -> None:
    row = {
        "target_type": "government",
        "target_name": "정부",
        "text_excerpt": "정부의 경제정책 실패를 분명히 비판합니다.",
    }
    prediction, reasons = apply_ambiguity_abstention([row], ["negative"])
    assert prediction.tolist() == ["negative"]
    assert reasons == [""]


def test_v15_abstains_on_long_target_owned_report() -> None:
    row = {
        "target_type": "person",
        "target_name": "문재인",
        "text_excerpt": (
            "문재인 정부는 우리 경제와 사회에 저성장이 고착화되고 양극화가 "
            "심화되어 공동체가 무너질 수도 있다고 진단했습니다."
        ),
    }

    prediction, reasons = apply_contextual_ownership_gate_v15([row], ["negative"])

    assert target_owned_report_reason(row) == "target_subject_owns_reported_stance"
    assert prediction.tolist() == ["neutral"]
    assert "target_subject_owns_reported_stance" in reasons[0]


def test_v15_preserves_speaker_evaluation_of_target_policy() -> None:
    row = {
        "target_type": "person",
        "target_name": "문재인",
        "text_excerpt": "문재인 정부의 부동산 정책은 명백하게 실패했습니다.",
    }

    prediction, reasons = apply_contextual_ownership_gate_v15([row], ["negative"])

    assert target_owned_report_reason(row) == ""
    assert prediction.tolist() == ["negative"]
    assert reasons == [""]


def test_v15_abstains_on_party_owned_announcement() -> None:
    row = {
        "target_type": "party",
        "target_name": "한나라당",
        "text_excerpt": "한나라당은 오랜 논의 끝에 해당 정책을 지지한다고 발표했습니다.",
    }

    prediction, reasons = apply_contextual_ownership_gate_v15([row], ["positive"])

    assert prediction.tolist() == ["neutral"]
    assert "target_subject_owns_reported_stance" in reasons[0]


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (
            {
                "target_type": "person",
                "target_name": "박근혜",
                "text_excerpt": "야당에서는 박근혜정부의 경제정책이 실패했다고 비판합니다.",
            },
            "external_actor_owns_direction",
        ),
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "text_excerpt": "지난 참여정부 때의 부동산정책 실패가 원인입니다.",
            },
            "historical_government_not_current_target",
        ),
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "text_excerpt": "미국 행정부도 해당 방안을 지지하고 있습니다.",
            },
            "foreign_government_not_current_target",
        ),
        (
            {
                "target_type": "party",
                "target_name": "민주노동당",
                "text_excerpt": "민주노동당은 해당 방북을 적극 지지할 것입니다.",
            },
            "target_subject_owns_self_position",
        ),
    ],
)
def test_v16_abstains_on_attribution_failures(row, reason) -> None:
    prediction, reasons = apply_contextual_attribution_gate_v16([row], ["negative"])

    assert prediction.tolist() == ["neutral"]
    assert reason in reasons[0]


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "speaker": "정형근 위원",
                "text_excerpt": "이것이 과연 정부가 할 조치였다고 판단되는지 답변해 주시기 바랍니다.",
            },
            "answer_request_not_owned_stance",
        ),
        (
            {
                "target_type": "person",
                "target_name": "문재인",
                "speaker": "지상욱 위원",
                "text_excerpt": "교수님께서 문재인 정부의 정책은 실패했다고 판단하셨습니다.",
            },
            "reported_expert_stance",
        ),
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "speaker": "외교부장관 강경화",
                "text_excerpt": "우리 정부의 정책은 큰 성과를 거두었습니다.",
            },
            "government_representative_self_position",
        ),
        (
            {
                "target_type": "government",
                "target_name": "정부",
                "speaker": "홍길동 위원",
                "text_excerpt": "미국 신 정부 출범에 맞춰 대응해야 합니다.",
            },
            "foreign_or_future_government_scope",
        ),
    ],
)
def test_v17_abstains_on_speaker_and_scope_failures(row, reason) -> None:
    prediction, reasons = apply_contextual_speaker_scope_gate_v17([row], ["negative"])

    assert prediction.tolist() == ["neutral"]
    assert reason in reasons[0]


def test_risk_aware_context_omits_context_for_reported_speech() -> None:
    row = {
        "target_type": "party",
        "text_excerpt": "그 당이 찬성한다고 말했다는 보도입니까?",
        "context_before": "문맥에 강한 비판 표현이 있습니다.",
        "context_after": "문맥에 강한 지지 표현도 있습니다.",
        "agenda": "현안 질의",
    }

    value = compose_precision_input(row, "risk_aware_context")

    assert "강한 비판" not in value
    assert "강한 지지" not in value


def test_embedding_input_masks_target_and_drops_risky_context() -> None:
    row = {
        "target_name": "홍길동",
        "target_alias": "길동 후보",
        "text_excerpt": "홍길동이 찬성한다고 보도했습니까?",
        "context_before": "길동 후보를 강하게 비판합니다.",
        "context_after": "홍길동을 적극 지지합니다.",
    }

    value = compose_embedding_input(row, "risk_aware_context")

    assert "홍길동" not in value
    assert "길동 후보" not in value
    assert "강하게 비판" not in value
    assert "적극 지지" not in value


def test_precision_policy_abstains_when_either_head_is_uncertain() -> None:
    direction = np.asarray([0.94, 0.65, 0.94])
    polarity = np.asarray([[0.93, 0.07], [0.96, 0.04], [0.55, 0.45]])
    policy = PrecisionPolicy(direction_threshold=0.90, polarity_threshold=0.90)

    prediction = apply_precision_policy(direction, polarity, ["비판", "비판", "비판"], policy)

    assert prediction.tolist() == ["negative", "neutral", "neutral"]


def test_risk_surcharge_forces_question_to_abstain() -> None:
    direction = np.asarray([0.91, 0.91])
    polarity = np.asarray([[0.95, 0.05], [0.95, 0.05]])
    policy = PrecisionPolicy(
        direction_threshold=0.90,
        polarity_threshold=0.90,
        risk_surcharge=0.05,
    )

    prediction = apply_precision_policy(
        direction,
        polarity,
        ["잘못된 정책입니다.", "잘못된 정책입니까?"],
        policy,
    )

    assert prediction.tolist() == ["negative", "neutral"]


def test_risk_surcharge_forces_metalinguistic_example_to_abstain() -> None:
    direction = np.asarray([0.91])
    polarity = np.asarray([[0.05, 0.95]])
    policy = PrecisionPolicy(
        direction_threshold=0.90,
        polarity_threshold=0.90,
        risk_surcharge=0.05,
    )

    prediction = apply_precision_policy(
        direction,
        polarity,
        ["예컨대 갑 후보를 지지한다, 을 후보를 비판한다고 표현할 수 있습니다."],
        policy,
    )

    assert prediction.tolist() == ["neutral"]


def test_precision_metrics_separate_harmful_errors_from_abstention() -> None:
    truth = ["neutral", "negative", "positive", "negative"]
    prediction = ["positive", "positive", "neutral", "negative"]

    metrics = precision_first_metrics(truth, prediction)

    assert metrics["neutral_to_direction_count"] == 1
    assert metrics["wrong_direction_count"] == 1
    assert metrics["direction_to_neutral_count"] == 1
    assert metrics["correct_direction_count"] == 1
    assert metrics["harmful_error_count"] == 2
    assert metrics["harmful_error_rate_among_emitted"] == pytest.approx(2 / 3)
    assert metrics["harmful_error_upper_95"] > metrics["harmful_error_rate_among_emitted"]


def test_consensus_addition_abstains_on_risky_metalinguistic_text() -> None:
    combined, source = combine_precision_children(
        ["neutral"],
        ["neutral"],
        np.asarray([0.90]),
        np.asarray([0.90]),
        np.asarray([[0.05, 0.95]]),
        np.asarray([[0.10, 0.90]]),
        ["예컨대 갑 후보를 지지한다는 표현입니다."],
        ConsensusPolicy(0.30, 0.75, 0.0),
    )

    assert combined.tolist() == ["neutral"]
    assert source.tolist() == ["neutral"]


def test_conservative_child_conflict_forces_abstention() -> None:
    combined, source = combine_precision_children(
        ["negative"],
        ["positive"],
        np.asarray([0.99]),
        np.asarray([0.99]),
        np.asarray([[0.99, 0.01]]),
        np.asarray([[0.01, 0.99]]),
        ["직접 평가 문장"],
        ConsensusPolicy(0.30, 0.75, 0.0),
    )

    assert combined.tolist() == ["neutral"]
    assert source.tolist() == ["conflict_abstain"]


def test_neutral_information_retains_evidence_without_direction() -> None:
    evidence = neutral_information_features(
        "조사 결과 2021년 기준 37.5%로 집계되었고 자료로 확인했습니다."
    )
    short_question = neutral_information_features("확인했습니까?")

    assert evidence["neutral_information_category"] == "evidence"
    assert evidence["neutral_information_score"] > short_question["neutral_information_score"]
    assert evidence["neutral_numeric_evidence"] > 0


def test_gold_loader_rejects_any_forbidden_election(tmp_path) -> None:
    rows = []
    for split in ("train", "holdout"):
        for label in ("negative", "neutral", "positive"):
            rows.append(
                {
                    "election_id": "pres_2002",
                    "split": split,
                    "review_label": label,
                }
            )
    rows.append(
        {
            "election_id": "pres_2099",
            "split": "holdout",
            "review_label": "neutral",
        }
    )
    path = tmp_path / "gold.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="forbidden elections"):
        _load_gold(path)


def _manual_rows() -> list[dict[str, str]]:
    return [
        {
            "election_id": "pres_2002",
            "review_label": label,
            "review_target_correct": "true",
            "text_sha256": char * 64,
        }
        for label, char in (("negative", "a"), ("neutral", "b"), ("positive", "c"))
    ]


def test_manual_loader_rejects_forbidden_election(tmp_path) -> None:
    rows = _manual_rows()
    rows[0]["election_id"] = "pres_2025"
    path = tmp_path / "manual.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="forbidden elections"):
        _load_manual(path, pd.DataFrame({"text_sha256": []}))


def test_manual_loader_rejects_false_target_review(tmp_path) -> None:
    rows = _manual_rows()
    rows[0]["review_target_correct"] = "false"
    path = tmp_path / "manual.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="target-invalid"):
        _load_manual(path, pd.DataFrame({"text_sha256": []}))


def test_manual_loader_rejects_original_gold_overlap(tmp_path) -> None:
    rows = _manual_rows()
    path = tmp_path / "manual.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    original = pd.DataFrame({"text_sha256": ["a" * 64]})

    with pytest.raises(ValueError, match="overlaps original gold"):
        _load_manual(path, original)


def _shadow_corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "election_id": ["pres_2002"],
            "text_excerpt": ["검증 문장"],
            "text_sha256": ["d" * 64],
            "target_type": ["person"],
            "target_name": ["후보"],
            "meeting_date": ["2002-12-18"],
        }
    )


def test_shadow_application_rejects_post_2022_rows() -> None:
    frame = _shadow_corpus()
    frame.loc[0, "election_id"] = "pres_2025"

    with pytest.raises(ValueError, match="forbidden elections"):
        validate_shadow_corpus(frame)


def test_shadow_application_rejects_outcome_columns() -> None:
    frame = _shadow_corpus()
    frame["actual_vote_share"] = 0.5

    with pytest.raises(ValueError, match="outcome-like columns"):
        validate_shadow_corpus(frame)


def test_shadow_application_rejects_post_cutoff_meeting() -> None:
    frame = _shadow_corpus()
    frame.loc[0, "meeting_date"] = "2002-12-20"

    with pytest.raises(ValueError, match="post-cutoff"):
        validate_shadow_corpus(frame)


def test_adoption_gate_does_not_use_coverage_threshold() -> None:
    metrics = {
        "predicted_directional_rows": 59,
        "harmful_error_count": 0,
        "harmful_error_upper_95": 0.0495,
        "correct_direction_coverage": 0.01,
    }

    result = stance_adoption_assessment(
        metrics,
        independent_audit=True,
        target_attribution_audited=True,
        point_in_time_audited=True,
        rolling_non_degradation=True,
    )

    assert result["classifier_quality_gate_passed"] is True
    assert result["quality_gate"]["coverage_threshold"] is None


def test_adoption_gate_rejects_reused_holdout() -> None:
    metrics = {
        "predicted_directional_rows": 100,
        "harmful_error_count": 0,
        "harmful_error_upper_95": 0.03,
    }

    result = stance_adoption_assessment(
        metrics,
        independent_audit=False,
        target_attribution_audited=True,
        point_in_time_audited=True,
        rolling_non_degradation=True,
    )

    assert result["classifier_quality_gate_passed"] is False
    assert result["checks"]["independent_audit"] is False


def test_v18_abstains_for_named_noncurrent_government() -> None:
    from election_forecast.stance_context_v18 import contextual_assertion_reasons_v18

    row = {
        "target_type": "government",
        "target_name": "정부",
        "meeting_date": "2014-10-27",
        "text_excerpt": "이게 이명박 정부의 대표적인 교육 실패 정책입니다.",
        "speaker": "홍길동 위원",
    }

    assert "named_noncurrent_government" in contextual_assertion_reasons_v18(row)


def test_v18_keeps_named_current_government() -> None:
    from election_forecast.stance_context_v18 import contextual_assertion_reasons_v18

    row = {
        "target_type": "government",
        "target_name": "정부",
        "meeting_date": "2011-02-28",
        "text_excerpt": "이명박 정부는 물가 관리에 실패했습니다.",
        "speaker": "홍길동 위원",
    }

    assert "named_noncurrent_government" not in contextual_assertion_reasons_v18(row)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "우리 당과 정부는 평화체제로 바꾸는 과제를 추진하도록 하겠습니다.",
            "governing_party_self_commitment",
        ),
        (
            "이 잘못이 복지부의 것인가 청와대의 것인가가 문제입니다.",
            "unresolved_attribution_question",
        ),
        (
            "경제가 어려워질수록 정부와 대통령의 지지도는 떨어집니다.",
            "analytical_support_projection",
        ),
        (
            "제대로 경영하지 않으면 정말로 나쁜 정부라고 생각합니다.",
            "conditional_evaluation",
        ),
        (
            "박원순 서울시장도 그렇고 이해찬 의원도 그렇고 지금의 고용 악화가 이전 정부 탓이다.",
            "reported_political_actors",
        ),
    ],
)
def test_v18_abstains_for_nonassertive_or_reported_frames(text: str, reason: str) -> None:
    from election_forecast.stance_context_v18 import contextual_assertion_reasons_v18

    row = {
        "target_type": "government",
        "target_name": "정부",
        "meeting_date": "2019-01-01",
        "text_excerpt": text,
        "speaker": "홍길동 위원",
    }

    assert reason in contextual_assertion_reasons_v18(row)


@pytest.mark.parametrize(
    ("text", "before", "reason"),
    [
        (
            "유동화에 실패하면 지방정부 재정에 직접적인 타격을 줄 수 있습니다.",
            "",
            "local_government_not_national_target",
        ),
        (
            "최근 갤럽조사에 따르면 국민 87%가 정부 정책이 실패했다고 합니다.",
            "",
            "survey_owns_direction",
        ),
        (
            "기업하는 사람들이 정부의 정책을 불신하는 것이 문제입니다.",
            "",
            "collective_actor_owns_direction",
        ),
        (
            "인도는 우리 정부의 비핵화 정책을 적극적으로 지지하고 있습니다.",
            "",
            "government_policy_is_support_object",
        ),
        (
            "보금자리주택사업은 정부가 내놓은 잘못된 정책입니다.",
            "현 정부 역시 이 문제를 해결하기 위해 공급을 조정했습니다.",
            "unnamed_prior_policy_not_current_government",
        ),
    ],
)
def test_v19_abstains_for_external_owner_or_wrong_government_scope(
    text: str, before: str, reason: str
) -> None:
    from election_forecast.stance_context_v19 import contextual_owner_resolution_reasons_v19

    row = {
        "target_type": "government",
        "target_name": "정부",
        "meeting_date": "2019-01-01",
        "text_excerpt": text,
        "context_before": before,
        "speaker": "홍길동 위원",
    }

    assert reason in contextual_owner_resolution_reasons_v19(row)


def test_v19_keeps_explicit_first_person_government_support() -> None:
    from election_forecast.stance_context_v19 import contextual_owner_resolution_reasons_v19

    row = {
        "target_type": "government",
        "target_name": "정부",
        "meeting_date": "2019-01-01",
        "text_excerpt": "저는 우리 정부의 평화 정책을 적극 지지합니다.",
        "context_before": "",
        "speaker": "홍길동 위원",
    }

    assert "government_policy_is_support_object" not in contextual_owner_resolution_reasons_v19(row)


@pytest.mark.parametrize(
    ("speaker", "target_type", "text", "before", "reason"),
    [
        (
            "국가안보실장 홍길동",
            "government",
            "정부의 노력은 결코 멈출 수 없습니다.",
            "",
            "government_official_self_position_v20",
        ),
        (
            "홍길동 위원",
            "government",
            "역대 정부의 무능과 무기력이 문제입니다.",
            "",
            "historical_government_scope_v20",
        ),
        (
            "홍길동 위원",
            "government",
            "만일 정부가 일관성을 잃는다면 정책 실패의 원인이 됩니다.",
            "",
            "conditional_direction_v20",
        ),
        (
            "홍길동 위원",
            "government",
            "그 원인으로 청와대에 대한 불신이 거론되고 있습니다.",
            "외국 보고서는 정부의 실패를 지적하고 있습니다.",
            "continued_external_report_v20",
        ),
        (
            "홍길동 위원",
            "government",
            "뚜렷한 대책도 없이 낙관하면 정부 신뢰를 떨어뜨리고 불확실성을 증폭시킵니다.",
            "",
            "analytical_causal_projection_v20",
        ),
    ],
)
def test_v20_abstains_for_role_report_or_scope(
    speaker: str, target_type: str, text: str, before: str, reason: str
) -> None:
    from election_forecast.stance_context_v20 import contextual_strict_owner_reasons_v20

    row = {
        "target_type": target_type,
        "target_name": "정부",
        "meeting_date": "2019-01-01",
        "text_excerpt": text,
        "context_before": before,
        "speaker": speaker,
    }

    assert reason in contextual_strict_owner_reasons_v20(row)


def test_v20_requires_explicit_owner_for_positive_emission() -> None:
    from election_forecast.stance_context_v20 import apply_contextual_strict_owner_gate_v20

    rows = [
        {
            "target_type": "person",
            "target_name": "후보",
            "meeting_date": "2019-01-01",
            "text_excerpt": "후보의 정책은 큰 성과입니다.",
            "speaker": "홍길동 위원",
        },
        {
            "target_type": "person",
            "target_name": "후보",
            "meeting_date": "2019-01-01",
            "text_excerpt": "저는 후보의 정책을 높이 평가하고 지지합니다.",
            "speaker": "홍길동 위원",
        },
    ]

    prediction, _ = apply_contextual_strict_owner_gate_v20(rows, ["positive", "positive"])

    assert prediction.tolist() == ["neutral", "positive"]


@pytest.mark.parametrize(
    ("speaker", "target_type", "target_name", "text", "before", "reason"),
    [
        (
            "이용주 의원",
            "government",
            "정부",
            "이처럼 무능한 정부를 우리 국민들께서는 심판하였습니다.",
            "무능과 비리의 이명박ㆍ박근혜 정부의 9년입니다.",
            "demonstrative_inherits_historical_government_v21",
        ),
        (
            "노철래 위원",
            "government",
            "정부",
            "물가를 잡지 못하면 서민경제가 파탄나고 실패한 정부로 끝날 수 있습니다.",
            "",
            "conditional_failure_projection_v21",
        ),
        (
            "수석전문위원 국경복",
            "government",
            "정부",
            "경제정책을 총괄하기에는 크게 부족하다는 의견도 있었습니다.",
            "",
            "committee_staff_reports_others_v21",
        ),
        (
            "박범계 위원",
            "person",
            "문재인",
            "문재인 대통령께서 말씀하신 공공기관 채용 비리는 엄단해야 마땅하다고 생각합니다.",
            "",
            "target_person_is_quote_owner_v21",
        ),
        (
            "김경재 위원",
            "government",
            "정부",
            "정부의 요구를 국민들이 순순히 받아들이지는 않을 것이라는 것이 일반적인 관측입니다.",
            "",
            "reported_public_observation_v21",
        ),
        (
            "이한구 의원",
            "party",
            "한나라당",
            "한나라당은 위기 확산을 예측했기 때문에 추가 대책을 강조했었지만 정부는 오락가락하고 거짓말만 되풀이했습니다.",
            "",
            "party_mentioned_but_government_evaluated_v21",
        ),
        (
            "조한천 의원",
            "party",
            "한나라당",
            "경제위기는 과거정부의 실정에서 비롯되었고 지금의 한나라당도 부인할 수 없는 사실입니다.",
            "",
            "party_only_concedes_prior_government_blame_v21",
        ),
        (
            "고승덕 위원",
            "person",
            "이명박",
            "이명박 정부는 부동산 침체의 폭탄을 떠안은 책임밖에 없는 것이고 모든 거품은 참여정부와 그전의 정부에서 만들어졌습니다.",
            "",
            "target_defended_by_prior_government_blame_v21",
        ),
    ],
)
def test_v21_resolves_owner_scope_and_target_before_polarity(
    speaker: str,
    target_type: str,
    target_name: str,
    text: str,
    before: str,
    reason: str,
) -> None:
    from election_forecast.stance_context_v21 import resolve_discourse_target_v21

    row = {
        "speaker": speaker,
        "target_type": target_type,
        "target_name": target_name,
        "text_excerpt": text,
        "context_before": before,
    }

    assert reason in resolve_discourse_target_v21(row)["abstention_reasons"]


def test_v21_recovers_explicit_first_person_support_after_target_mention() -> None:
    from election_forecast.stance_context_v21 import apply_discourse_target_gate_v21

    row = {
        "speaker": "김태흠 위원",
        "target_type": "government",
        "target_name": "정부",
        "text_excerpt": "지금 현 정부의 지방분권과 국가균형 발전에 저는 100% 동의하는 사람이에요.",
        "context_before": "",
    }

    prediction, reasons, _ = apply_discourse_target_gate_v21([row], ["positive"])

    assert prediction.tolist() == ["positive"]
    assert reasons == [""]


def test_v21_does_not_recover_positive_without_speaker_ownership() -> None:
    from election_forecast.stance_context_v21 import apply_discourse_target_gate_v21

    row = {
        "speaker": "대통령비서실장 임종석",
        "target_type": "person",
        "target_name": "문재인",
        "text_excerpt": "문재인 대통령을 100% 신뢰한다는 말과 평창올림픽에 대한 지지가 있었습니다.",
        "context_before": "",
    }

    prediction, reasons, _ = apply_discourse_target_gate_v21([row], ["positive"])

    assert prediction.tolist() == ["neutral"]
    assert "positive_owner_not_explicit_v20" in reasons[0]


def test_v21_keeps_direct_speaker_owned_negative() -> None:
    from election_forecast.stance_context_v21 import apply_discourse_target_gate_v21

    row = {
        "speaker": "김용환 위원",
        "target_type": "government",
        "target_name": "정부",
        "text_excerpt": "리더십의 부재와 정부의 무능 때문입니다.",
        "context_before": "",
    }

    prediction, reasons, _ = apply_discourse_target_gate_v21([row], ["negative"])

    assert prediction.tolist() == ["negative"]
    assert reasons == [""]


@pytest.mark.parametrize(
    ("target_type", "target_name", "text", "reason"),
    [
        (
            "party",
            "자유한국당",
            "자유한국당은 실패한 정책의 책임을 물어 관련 장관의 경질을 요구했습니다.",
            "party_owns_criticism_not_negative_target_v22",
        ),
        (
            "government",
            "정부",
            "이는 공공기관으로서 정부의 국가균형발전 정책을 저해하고 있는 겁니다.",
            "government_policy_is_beneficiary_v22",
        ),
        (
            "government",
            "정부",
            "막대한 토지 미매각으로 사업시행자인 토지공사의 사업성이 악화되었습니다.",
            "central_government_referent_absent_v22",
        ),
        (
            "government",
            "정부",
            "정부투자기관과 지방자치단체 조달 담당자의 전문성 부족으로 예산이 낭비됩니다.",
            "affiliated_or_local_body_not_central_government_v22",
        ),
        (
            "government",
            "정부",
            "물가가 급등했거나 민영화 준비가 안 되어 있거나 정부에서 관리를 못한 경우 등은 실패하는 요인입니다.",
            "generic_hypothetical_failure_v22",
        ),
        (
            "government",
            "정부",
            "노동계가 정부의 정책을 극도로 불신하고 있다는 것입니다.",
            "collective_report_owns_direction_v22",
        ),
        (
            "government",
            "정부",
            "지역편중 인사와 낙하산 공화국이라고 국민들은 말하고 있습니다.",
            "collective_report_owns_direction_v22",
        ),
        (
            "government",
            "정부",
            "준정부기관과 공기업이 파산할 경우 국가가 모든 부채를 책임지면 도덕적 해이가 생길 수 있습니다.",
            "central_government_referent_absent_v22",
        ),
    ],
)
def test_v22_removes_non_target_or_non_asserted_negative(
    target_type: str, target_name: str, text: str, reason: str
) -> None:
    from election_forecast.stance_context_v22 import grammatical_target_reasons_v22

    row = {
        "speaker": "테스트 위원",
        "target_type": target_type,
        "target_name": target_name,
        "text_excerpt": text,
        "context_before": "",
    }

    assert reason in grammatical_target_reasons_v22(row)


def test_v22_preserves_explicit_current_government_failure() -> None:
    from election_forecast.stance_context_v22 import apply_grammatical_target_gate_v22

    row = {
        "speaker": "테스트 위원",
        "target_type": "government",
        "target_name": "정부",
        "text_excerpt": "현 정부의 부동산 정책은 명백하게 실패했습니다.",
        "context_before": "",
    }

    prediction, reasons, _ = apply_grammatical_target_gate_v22([row], ["negative"])

    assert prediction.tolist() == ["negative"]
    assert reasons == [""]


@pytest.mark.parametrize(
    ("meeting_date", "text", "reason"),
    [
        (
            "2020-01-01",
            "이 주장은 정부를 비난하는 소재로 삼아 국민의 불안을 자극하는 것입니다.",
            "government_is_criticism_object_v23s",
        ),
        (
            "2010-01-01",
            "국민은 정부를 믿지 못하고 지역 간 갈등은 깊어지고 있습니다.",
            "singular_public_owns_direction_v23s",
        ),
        (
            "2014-01-01",
            "많은 위원님들이 박근혜정부의 지지율 급락에 대해서 걱정하고 계십니다.",
            "committee_collective_owns_direction_v23s",
        ),
        (
            "2006-01-01",
            "정부가 만약 그런 사업으로 재정을 운용한다면 지출을 줄이는 방향입니다.",
            "neutral_government_hypothesis_v23s",
        ),
        (
            "2006-01-01",
            "YS 정부가 경제정책에 실패해서 많은 사람이 처벌받았습니다.",
            "only_historical_named_government_v23s",
        ),
    ],
)
def test_v23s_removes_pragmatic_role_and_historical_scope_errors(
    meeting_date: str, text: str, reason: str
) -> None:
    from election_forecast.stance_context_v23s import pragmatic_role_reasons_v23s

    row = {
        "meeting_date": meeting_date,
        "speaker": "테스트 위원",
        "target_type": "government",
        "target_name": "정부",
        "text_excerpt": text,
        "context_before": "",
    }

    assert reason in pragmatic_role_reasons_v23s(row)


def test_v23s_does_not_treat_current_named_government_as_historical() -> None:
    from election_forecast.stance_context_v23s import apply_pragmatic_role_gate_v23s

    row = {
        "meeting_date": "2015-01-01",
        "speaker": "테스트 위원",
        "target_type": "government",
        "target_name": "정부",
        "text_excerpt": "박근혜 정부의 경제정책은 명백하게 실패했습니다.",
        "context_before": "",
    }

    prediction, reasons, _ = apply_pragmatic_role_gate_v23s([row], ["negative"])

    assert prediction.tolist() == ["negative"]
    assert reasons == [""]


def test_v24s_removes_all_independently_observed_v15_harmful_errors() -> None:
    from pathlib import Path

    import pandas as pd

    from election_forecast.stance_context_v24s import apply_lexical_role_gate_v24s
    from election_forecast.stance_precision import precision_first_metrics

    root = Path(__file__).resolve().parents[1]
    audit = pd.read_csv(
        root / "data" / "shadow" / "stance_locked_audit_v15.csv",
        encoding="utf-8-sig",
    ).fillna("")
    labels = pd.read_csv(
        root / "data" / "shadow" / "stance_locked_audit_v15_labels.csv",
        encoding="utf-8-sig",
    ).fillna("")
    evaluated = audit.merge(labels, on="text_sha256", validate="one_to_one")
    prediction, _, _ = apply_lexical_role_gate_v24s(
        evaluated.to_dict(orient="records"), evaluated["context_prediction"]
    )
    truth = evaluated["audit_locked_label"].where(
        evaluated["audit_target_correct"].astype(str).str.lower().eq("true"),
        "neutral",
    )
    metrics = precision_first_metrics(truth, prediction)

    assert metrics["harmful_error_count"] == 0
    assert metrics["correct_direction_count"] == 56


def test_v24s_removes_false_government_substring_in_cheonjeongbuji() -> None:
    from election_forecast.stance_context_v24s import apply_lexical_role_gate_v24s

    row = {
        "meeting_date": "2003-10-29",
        "speaker": "테스트 의원",
        "target_type": "government",
        "target_name": "정부",
        "text_excerpt": "부동산가격은 천정부지로 솟고 빈익빈 부익부가 심화되고 있습니다.",
        "context_before": "대기업은 투자를 줄이고 중소기업은 자금난을 겪고 있습니다.",
    }

    prediction, reasons, _ = apply_lexical_role_gate_v24s([row], ["negative"])

    assert prediction.tolist() == ["neutral"]
    assert "government_only_inside_false_compound_v24s" in reasons[0]


def test_v25s_removes_all_independently_observed_v16_harmful_errors() -> None:
    from pathlib import Path

    import pandas as pd

    from election_forecast.stance_context_v25s import apply_semantic_role_gate_v25s
    from election_forecast.stance_precision import precision_first_metrics

    root = Path(__file__).resolve().parents[1]
    audit = pd.read_csv(
        root / "data" / "shadow" / "stance_locked_audit_v16.csv",
        encoding="utf-8-sig",
    ).fillna("")
    labels = pd.read_csv(
        root / "data" / "shadow" / "stance_locked_audit_v16_labels.csv",
        encoding="utf-8-sig",
    ).fillna("")
    evaluated = audit.merge(labels, on="text_sha256", validate="one_to_one")
    prediction, _, _ = apply_semantic_role_gate_v25s(
        evaluated.to_dict(orient="records"), evaluated["context_prediction"]
    )
    truth = evaluated["audit_locked_label"].where(
        evaluated["audit_target_correct"].astype(str).str.lower().eq("true"),
        "neutral",
    )
    metrics = precision_first_metrics(truth, prediction)

    assert metrics["harmful_error_count"] == 0
    assert metrics["correct_direction_count"] == 85
