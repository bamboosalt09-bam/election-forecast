from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from presidential_issue_engine.election_derived_third_candidate_profile_v3 import (
    build_election_derived_third_profile_v3,
)


ROOT = Path(__file__).resolve().parents[1]


def _real_inputs() -> tuple[pd.DataFrame, ...]:
    return (
        pd.read_csv(
            ROOT
            / "outputs"
            / "speech_derived_candidate_context_v2"
            / "auto_candidate_role"
            / "third_candidate_profile.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT / "data" / "raw" / "candidate_party_speech_context.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT / "data" / "raw" / "official_sources" / "nec_candidate_history.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT
            / "presidential_issue_engine"
            / "fixed_dataset"
            / "presidential_results_standardized.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT
            / "presidential_issue_engine"
            / "fixed_dataset"
            / "bloc_history_results.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT / "data" / "raw" / "candidate_political_landscape.csv",
            encoding="utf-8-sig",
        ),
        pd.read_csv(
            ROOT
            / "outputs"
            / "footprint_candidate_base_v9"
            / "candidate_regional_base.csv",
            encoding="utf-8-sig",
        ),
    )


def test_target_presidential_outcomes_cannot_change_v3_profile() -> None:
    inputs = _real_inputs()
    base, _ = build_election_derived_third_profile_v3(*inputs)
    for election_id in base["election_id"]:
        changed_results = inputs[3].copy()
        target = changed_results["election_id"].astype(str).eq(str(election_id))
        changed_results.loc[target, "votes"] = 999_999_999.0
        if "vote_share" in changed_results.columns:
            changed_results.loc[target, "vote_share"] = 1.0
        changed, _ = build_election_derived_third_profile_v3(
            inputs[0],
            inputs[1],
            inputs[2],
            changed_results,
            inputs[4],
            inputs[5],
            inputs[6],
        )
        columns = [
            "election_id",
            "slot",
            "viability",
            "centrist_appeal",
            "anti_major_party_appeal",
            "regional_base_overlap",
        ]
        expected = base.loc[base["election_id"].eq(election_id), columns]
        observed = changed.loc[changed["election_id"].eq(election_id), columns]
        assert_frame_equal(
            expected.reset_index(drop=True),
            observed.reset_index(drop=True),
            check_exact=True,
        )
