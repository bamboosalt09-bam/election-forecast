from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import assembly_bloc_issue_posture as posture  # noqa: E402


def test_assembly_bloc_issue_posture_uses_speaker_history_weights(tmp_path, monkeypatch) -> None:
    matches = tmp_path / "matches.csv"
    matches.write_text(
        "\n".join(
            [
                "election_id,assembly_daesu,meeting_date,speaker,issue_name,issue_weight,matched_term_count",
                "pres_2017,20,2017-04-01,홍길동 의원,economy_growth,1.0,4",
                "pres_2017,20,2017-04-01,홍길동 의원,corruption_integrity,0.5,1",
                "pres_2017,20,2017-04-01,김갑수 의원,economy_growth,0.5,1",
            ]
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "profile.csv"
    profile.write_text(
        "\n".join(
            [
                "election_id,assembly_daesu,speaker_clean,speaker_bloc,mandate_type,seniority_weight,role_weight,meeting_weight,mapping_confidence",
                "pres_2017,20,홍길동,더불어민주당,district,1.2,1.1,1.0,0.9",
                "pres_2017,20,김갑수,국민의힘,proportional,1.0,1.0,1.0,0.8",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(posture, "MATCHES", matches)
    monkeypatch.setattr(posture, "SPEAKER_PROFILE", profile)

    issues, diag = posture.load_assembly_bloc_issue_posture()
    dem_vector = posture.vector_for(issues, "pres_2022", "더불어민주당")

    assert {"election_id", "bloc", "issue_name", "bloc_issue_weight"}.issubset(issues.columns)
    assert dem_vector["economy_growth"] > dem_vector["corruption_integrity"]
    assert diag.loc[diag["bloc"].eq("더불어민주당"), "district_share"].iloc[0] == pytest.approx(1.0)
    assert posture.diagnostics_for(diag, "pres_2022", "더불어민주당")["speaker_coverage"] > 0


def test_prior_fallback_never_uses_future_elections() -> None:
    issues = pd.DataFrame(
        [
            {
                "election_id": "pres_2007",
                "bloc": "bloc_a",
                "issue_name": "economy_growth",
                "bloc_issue_weight": 0.3,
            },
            {
                "election_id": "pres_2022",
                "bloc": "bloc_a",
                "issue_name": "economy_growth",
                "bloc_issue_weight": 0.9,
            },
        ]
    )

    vector = posture.vector_for(issues, "pres_2012", "bloc_a")

    assert vector["economy_growth"] == pytest.approx(0.3)
