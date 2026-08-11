from __future__ import annotations

from presidential_issue_engine.build_issue_epoch_importance_from_salience import build_issue_epoch_importance


def test_build_issue_epoch_importance_uses_issue_share_and_momentum(tmp_path) -> None:
    salience = tmp_path / "salience.csv"
    salience.write_text(
        "\n".join(
            [
                "election_id,issue_name,period,salience_score,available_date",
                "pres_2007,economy_growth,2007-10-01,0.8,2007-10-07",
                "pres_2007,economy_growth,2007-11-20,0.8,2007-11-26",
                "pres_2007,security_nk,2007-10-01,0.1,2007-10-07",
                "pres_2007,security_nk,2007-11-20,0.1,2007-11-26",
                "pres_2007,late_issue,2007-10-01,0.1,2007-10-07",
                "pres_2007,late_issue,2007-11-20,0.6,2007-11-26",
            ]
        ),
        encoding="utf-8",
    )

    out, diagnostics = build_issue_epoch_importance(salience, strength=0.2)
    values = dict(zip(out["issue_name"], out["importance_multiplier"]))
    momentum = dict(zip(diagnostics["issue_name"], diagnostics["late_momentum"]))

    assert values["economy_growth"] > values["security_nk"]
    assert values["late_issue"] > values["security_nk"]
    assert momentum["late_issue"] > 0
    assert set(out.columns) == {
        "election_id",
        "issue_name",
        "importance_multiplier",
        "available_date",
        "confidence",
        "notes",
    }


def test_build_issue_epoch_importance_boosts_scoped_mega_terms(tmp_path) -> None:
    salience = tmp_path / "salience.csv"
    mega_terms = tmp_path / "mega_issue_terms.csv"
    salience.write_text(
        "\n".join(
            [
                "election_id,issue_name,period,salience_score,available_date",
                "pres_2017,regime_change,2017-04-01,0.05,2017-04-07",
                "pres_2017,economy_growth,2017-04-01,0.50,2017-04-07",
                "pres_2022,regime_change,2022-02-01,0.05,2022-02-07",
                "pres_2022,economy_growth,2022-02-01,0.50,2022-02-07",
            ]
        ),
        encoding="utf-8",
    )
    mega_terms.write_text(
        "\n".join(
            [
                "issue_name,term,term_type,weight,start_election,end_election,notes",
                "regime_change,탄핵,mega_issue,1.4,pres_2017,pres_2017,impeachment",
            ]
        ),
        encoding="utf-8",
    )

    out, diagnostics = build_issue_epoch_importance(
        salience,
        mega_terms_path=mega_terms,
        strength=0.0,
        mega_term_strength=0.75,
    )
    values = dict(zip(zip(out["election_id"], out["issue_name"]), out["importance_multiplier"]))
    diag = diagnostics.set_index(["election_id", "issue_name"])

    assert values[("pres_2017", "regime_change")] > values[("pres_2022", "regime_change")]
    assert diag.loc[("pres_2017", "regime_change"), "mega_term_weight"] == 1.4
    assert diag.loc[("pres_2022", "regime_change"), "mega_term_weight"] == 1.0
