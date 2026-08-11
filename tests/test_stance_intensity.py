from election_forecast.stance_intensity import neutral_information, stance_intensity


def test_stance_intensity_uses_directional_contrast() -> None:
    result = stance_intensity(0.10, 0.20, 0.70, "매우 강력히 지지합니다.")
    assert result.positive_strength > 0.60
    assert result.negative_strength == 0.0
    assert result.directional_score > 0.0


def test_neutral_information_distinguishes_substance_from_procedure() -> None:
    substantive = neutral_information(
        0.80,
        "통계청 자료에 따르면 주택가격이 12% 증가한 원인과 영향에 대한 대책이 필요합니다.",
        issue_name="housing",
    )
    procedural = neutral_information(
        0.80,
        "다음 안건을 상정합니다.",
        issue_name="housing",
    )
    assert substantive.neutral_information_score > procedural.neutral_information_score
    assert substantive.label in {"medium", "high"}
    assert procedural.label == "none"


def test_neutral_information_does_not_create_direction() -> None:
    result = stance_intensity(0.20, 0.60, 0.20, "원인과 현황을 분석합니다.")
    assert result.directional_score == 0.0
    assert result.positive_strength == 0.0
    assert result.negative_strength == 0.0
