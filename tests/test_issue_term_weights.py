from __future__ import annotations

from pathlib import Path

from election_forecast.features.issue_matcher import match_issue_weights
from news_collector.sources.issue_term_weights import load_campaign_issue_terms, merge_issue_terms


def test_load_campaign_issue_terms_filters_by_election_window(tmp_path: Path) -> None:
    path = tmp_path / "campaign_issue_terms.csv"
    path.write_text(
        "\n".join(
            [
                "issue_name,term,term_type,weight,start_election,end_election,notes",
                "housing,전세 사기,campaign_phrase,1.4,pres_2022,pres_2022,",
                "housing,종부세,campaign_term,1.2,pres_2007,,",
            ]
        ),
        encoding="utf-8",
    )

    terms, weights = load_campaign_issue_terms(path, ["pres_2017", "pres_2022"])

    assert "전세 사기" not in terms["pres_2017"].get("housing", [])
    assert terms["pres_2022"]["housing"] == ["전세 사기", "종부세"]
    assert weights["pres_2022"][("housing", "전세 사기")] == 1.4


def test_campaign_weight_overrides_default_term_weight() -> None:
    base = {"housing": ["전세 사기"]}
    extra = {"housing": ["전세 사기"]}
    merged = merge_issue_terms(base, extra)

    weights = match_issue_weights(
        "전세 사기 대책을 논의했다.",
        merged,
        term_weights={("housing", "전세 사기"): 1.4},
    )

    assert weights == {"housing": 1.4}


def test_mega_issue_terms_are_election_scoped() -> None:
    terms, weights = load_campaign_issue_terms(
        "presidential_issue_engine/fixed_dataset/mega_issue_terms.csv",
        ["pres_2017", "pres_2022"],
    )

    assert "\ud0c4\ud575" in terms["pres_2017"]["regime_change"]
    assert "\ub300\uc7a5\ub3d9" in terms["pres_2022"]["corruption_integrity"]
    assert ("external_shock", "\uacc4\uc5c4") not in weights["pres_2022"]
