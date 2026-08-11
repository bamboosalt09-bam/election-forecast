import pandas as pd

from election_forecast.stance_explanatory_overlay import compile_explanatory_overlay


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"election_id": "pres_test", "slot": "A", "candidate_name": "가", "party_name": "가당", "candidate_bloc": "challenger"},
            {"election_id": "pres_test", "slot": "B", "candidate_name": "나", "party_name": "나당", "candidate_bloc": "incumbent"},
        ]
    )


def _row(hash_value: str, label: str, *, issue: str, target: str = "") -> dict[str, object]:
    return {
        "election_id": "pres_test",
        "meeting_date": "2020-01-01",
        "issue_name": issue,
        "speaker": "의원",
        "committee": "위원회",
        "target_type": "person" if target else "none",
        "target_name": target,
        "text_sha256": hash_value,
        "context_confidence": 0.8,
        "ambiguity_gated_prediction": label,
    }


def _responsibility() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "slot": "B",
                "responsibility_score": 1.0,
                "available_date": "2019-12-31",
            },
            {
                "election_id": "pres_test",
                "slot": "A",
                "responsibility_score": -1.0,
                "available_date": "2019-12-31",
            },
        ]
    )


def test_neutral_rows_are_retained_as_unsigned_issue_information() -> None:
    frame = pd.DataFrame(
        [
            _row("1", "neutral", issue="economy"),
            _row("2", "neutral", issue="economy"),
            _row("3", "negative", issue="integrity", target="가"),
        ]
    )
    out = compile_explanatory_overlay(frame, _candidates())
    economy = out.loc[out["issue_name"].eq("economy")]
    assert set(out["slot"]) == {"A", "B"}
    assert set(economy["issue_evidence_count"]) == {2}
    assert set(economy["issue_directional_count"]) == {0}
    assert set(economy["informational_score"]) == {1.0}


def test_directional_target_changes_only_bounded_issue_and_link_multipliers() -> None:
    frame = pd.DataFrame(
        [
            _row("1", "neutral", issue="economy"),
            _row("2", "negative", issue="integrity", target="가"),
            _row("3", "negative", issue="integrity", target="가"),
        ]
    )
    out = compile_explanatory_overlay(frame, _candidates(), character_gain=0.04, link_gain=0.01)
    assert out["salience_multiplier"].between(0.95, 1.05).all()
    assert out["link_multiplier"].between(0.98, 1.02).all()
    linked = out.loc[(out["issue_name"].eq("integrity")) & (out["slot"].eq("A"))].iloc[0]
    unlinked = out.loc[(out["issue_name"].eq("integrity")) & (out["slot"].eq("B"))].iloc[0]
    assert linked["link_evidence_count"] == 2
    assert linked["link_multiplier"] > unlinked["link_multiplier"]
    assert linked["target_directional_balance"] == -1.0
    assert linked["target_attribution_confidence"] > 0.0
    assert unlinked["target_directional_balance"] == 0.0


def test_government_target_maps_to_incumbent_responsibility_slot() -> None:
    row = _row("gov", "negative", issue="regime")
    row["target_type"] = "government"
    row["target_name"] = "government"
    out = compile_explanatory_overlay(
        pd.DataFrame([row]),
        _candidates(),
        government_responsibility=_responsibility(),
    )
    incumbent = out.loc[out["slot"].eq("B")].iloc[0]
    challenger = out.loc[out["slot"].eq("A")].iloc[0]
    assert incumbent["target_directional_balance"] == -1.0
    assert "government" in incumbent["target_source_types"]
    assert challenger["target_directional_balance"] == 0.0


def test_incumbent_person_alias_maps_to_responsible_camp() -> None:
    row = _row("incumbent", "negative", issue="integrity", target="former_president")
    incumbent_targets = pd.DataFrame(
        [
            {
                "election_id": "pres_test",
                "target_name": "former_president",
                "available_date": "2019-12-31",
            }
        ]
    )
    out = compile_explanatory_overlay(
        pd.DataFrame([row]),
        _candidates(),
        government_responsibility=_responsibility(),
        incumbent_targets=incumbent_targets,
    )
    incumbent = out.loc[out["slot"].eq("B")].iloc[0]
    assert incumbent["target_directional_balance"] == -1.0


def test_duplicate_text_hash_does_not_double_count_global_issue_mass() -> None:
    row = _row("same", "negative", issue="integrity", target="가")
    out = compile_explanatory_overlay(pd.DataFrame([row, row]), _candidates())
    assert set(out["issue_evidence_count"]) == {1}
