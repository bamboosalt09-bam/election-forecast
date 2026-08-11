"""Export helpers for the existing election_forecast input schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_election_issue_scores(input_path: str | Path, output_path: str | Path) -> int:
    frame = pd.read_csv(input_path)
    if frame.empty:
        out = pd.DataFrame(
            columns=[
                "date",
                "candidate_id",
                "issue_name",
                "salience_score",
                "direction_score",
                "candidate_link_score",
                "media_reliability_score",
                "final_issue_score",
                "available_date",
            ]
        )
    else:
        out = pd.DataFrame(
            {
                "date": frame["date"],
                "candidate_id": frame["candidate_id"],
                "issue_name": frame["issue_name"],
                "salience_score": frame["volume_z_score"],
                "direction_score": frame["weighted_stance"],
                "candidate_link_score": frame["avg_candidate_link_score"],
                "media_reliability_score": frame["source_reliability_avg"],
                "final_issue_score": frame["final_issue_score"],
                "available_date": frame["available_date"],
            }
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, encoding="utf-8")
    return len(out)
