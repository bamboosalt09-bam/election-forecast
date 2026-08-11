from __future__ import annotations

import pandas as pd
import pytest

from presidential_issue_engine import issue_vote_engine


def test_classified_institutional_crisis_has_larger_shock_than_policy_frame(tmp_path, monkeypatch) -> None:
    taxonomy = tmp_path / "mega_issue_taxonomy.csv"
    taxonomy.write_text(
        "\n".join(
            [
                "election_id,mega_event,shock_type,severity,national_scope,persistence,polarization,target_specificity,available_date,confidence,notes",
                "pres_2022,crisis,institutional_crisis,1,1,1,1,1,2022-03-01,1,test",
                "pres_2022,policy,distributional_policy,0.4,0.4,0.4,0.4,0.4,2022-03-01,1,test",
            ]
        ),
        encoding="utf-8",
    )
    intensity = tmp_path / "mega_issue_intensity.csv"
    intensity.write_text(
        "election_id,mega_issue_intensity,available_date,notes\npres_2022,1.0,2022-03-01,test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "MEGA_ISSUE_TAXONOMY", str(taxonomy))
    monkeypatch.setattr(issue_vote_engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(intensity))

    out = issue_vote_engine._mega_event_shock_profiles().set_index("mega_event")

    assert out.loc["crisis", "event_shock_intensity"] > out.loc["policy", "event_shock_intensity"]


def test_signed_mega_attribution_separates_positive_and_negative_slots(tmp_path, monkeypatch) -> None:
    axis = tmp_path / "mega_issue_axis.csv"
    axis.write_text(
        "\n".join(
            [
                "election_id,mega_event,primary_issue,secondary_issue,axis_weight,regime_axis_weight,available_date,activation_method,notes",
                "pres_2022,event,regime_change,,1.0,0.0,2022-03-01,manual,test",
            ]
        ),
        encoding="utf-8",
    )
    attribution = tmp_path / "mega_issue_attribution.csv"
    attribution.write_text(
        "\n".join(
            [
                "election_id,mega_event,issue_name,target_type,target,polarity,weight,available_date,confidence,notes",
                "pres_2022,event,regime_change,candidate_slot,A,1,1,2022-03-01,1,test",
                "pres_2022,event,regime_change,candidate_slot,B,-1,1,2022-03-01,1,test",
            ]
        ),
        encoding="utf-8",
    )
    taxonomy = tmp_path / "mega_issue_taxonomy.csv"
    taxonomy.write_text(
        "\n".join(
            [
                "election_id,mega_event,shock_type,severity,national_scope,persistence,polarization,target_specificity,available_date,confidence,notes",
                "pres_2022,event,institutional_crisis,1,1,1,1,1,2022-03-01,1,test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_AXIS", str(axis))
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_ATTRIBUTION", str(attribution))
    monkeypatch.setattr(issue_vote_engine, "MEGA_ISSUE_TAXONOMY", str(taxonomy))
    monkeypatch.setattr(issue_vote_engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(tmp_path / "missing.csv"))

    effects = issue_vote_engine._mega_signed_attribution_effects().set_index("slot")
    frame = pd.DataFrame(
        [
            {"election_id": "pres_2022", "slot": "A", "issue_name": "regime_change", "salience": 0.5},
            {"election_id": "pres_2022", "slot": "B", "issue_name": "regime_change", "salience": 0.5},
            {"election_id": "pres_2022", "slot": "C", "issue_name": "regime_change", "salience": 0.5},
        ]
    )
    attached = issue_vote_engine._attach_mega_signed_attribution(frame)

    assert effects.loc["A", "mega_signed_attribution_multiplier"] > 1.0
    assert effects.loc["B", "mega_signed_attribution_multiplier"] < 1.0
    assert attached.loc[attached["slot"].eq("C"), "mega_signed_attribution_multiplier"].iloc[0] == pytest.approx(1.0)


def test_signed_attribution_does_not_route_unknown_issue_to_axis_secondary(tmp_path, monkeypatch) -> None:
    axis = tmp_path / "mega_issue_axis.csv"
    axis.write_text(
        "election_id,mega_event,primary_issue,secondary_issue,axis_weight,regime_axis_weight,available_date,activation_method,notes\n"
        "pres_2022,event,regime_change,housing,1.5,1.2,2022-03-01,manual,test\n",
        encoding="utf-8",
    )
    attribution = tmp_path / "mega_issue_attribution.csv"
    attribution.write_text(
        "election_id,mega_event,issue_name,target_type,target,polarity,weight,available_date,confidence,notes\n"
        "pres_2022,event,unknown_issue,candidate_slot,A,1,1,2022-03-01,1,test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_AXIS", str(axis))
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_ATTRIBUTION", str(attribution))
    monkeypatch.setattr(issue_vote_engine, "MEGA_ISSUE_TAXONOMY", str(tmp_path / "missing.csv"))
    monkeypatch.setattr(issue_vote_engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(tmp_path / "missing_intensity.csv"))

    assert issue_vote_engine._mega_signed_attribution_effects().empty


def test_enhanced_fit_does_not_spread_explicit_attribution_to_secondary_issue(tmp_path, monkeypatch) -> None:
    axis = tmp_path / "mega_issue_axis.csv"
    axis.write_text(
        "election_id,mega_event,primary_issue,secondary_issue,axis_weight,regime_axis_weight,available_date,activation_method,notes\n"
        "pres_2022,event,corruption_integrity,regime_change,1.5,1.2,2022-03-01,manual,test\n",
        encoding="utf-8",
    )
    attribution = tmp_path / "mega_issue_attribution.csv"
    attribution.write_text(
        "election_id,mega_event,issue_name,target_type,target,polarity,weight,available_date,confidence,notes\n"
        "pres_2022,event,corruption_integrity,candidate_slot,A,-1,1,2022-03-01,1,test\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_AXIS", str(axis))
    monkeypatch.setattr(issue_vote_engine, "AUTO_MEGA_ISSUE_ATTRIBUTION", str(attribution))
    monkeypatch.setattr(issue_vote_engine, "AUTO_CANDIDATE_ISSUE_PROFILE", str(tmp_path / "missing_profile.csv"))
    monkeypatch.setattr(issue_vote_engine, "MEGA_ISSUE_TAXONOMY", str(tmp_path / "missing.csv"))
    monkeypatch.setattr(issue_vote_engine, "ENHANCED_MEGA_ISSUE_INTENSITY", str(tmp_path / "missing_intensity.csv"))
    slots = pd.DataFrame(
        [{"election_id": "pres_2022", "slot": "A"}, {"election_id": "pres_2022", "slot": "B"}]
    )

    out = issue_vote_engine._manual_signed_issue_fit(slots)

    assert set(out["issue_name"]) == {"corruption_integrity"}
    assert out.loc[out["slot"].eq("A"), "manual_signed_fit"].iloc[0] == pytest.approx(-1.0)
