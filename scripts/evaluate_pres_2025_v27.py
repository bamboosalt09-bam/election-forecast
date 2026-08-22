"""Score the already-frozen V27 D-1 forecast against the later 2025 result.

This script is evaluation-only. It does not import, fit, select, or modify the
forecast engine. Headline metrics normalize the actual result to the same A/B/C
three-candidate composition predicted by V27.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "outputs/prospective_pres_2025_v27/prospective_predictions.csv"
NATIONAL_FORECAST = ROOT / "outputs/prospective_pres_2025_v27/national_summary.csv"
EVALUATION_DIR = ROOT / "evaluations/pres_2025_v27"
ACTUAL = EVALUATION_DIR / "official_results.csv"
EXPECTED_FORECAST_SHA256 = "2df2b980504e7d8a88c54ceb52267a42dfe0faf2d1593fa29343fb3d946a5238"
EXPECTED_NATIONAL_SHA256 = "8d32b52141bd64afb37acf6bd9493de856b6f7d6e513882340610f849252b4b4"

CANDIDATES = {
    "A": ("이재명", "lee_jaemyung_votes"),
    "B": ("김문수", "kim_moonsu_votes"),
    "C": ("이준석", "lee_junseok_votes"),
}
OTHER_COLUMNS = ("kwon_yeongguk_votes", "song_jinho_votes")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate() -> dict[str, object]:
    if _sha256(FORECAST) != EXPECTED_FORECAST_SHA256:
        raise ValueError("V27 prospective forecast hash drift")
    if _sha256(NATIONAL_FORECAST) != EXPECTED_NATIONAL_SHA256:
        raise ValueError("V27 frozen national forecast hash drift")
    forecast = pd.read_csv(FORECAST, encoding="utf-8-sig")
    frozen_national = pd.read_csv(NATIONAL_FORECAST, encoding="utf-8-sig").set_index("slot")
    actual_wide = pd.read_csv(ACTUAL, encoding="utf-8-sig")
    if len(actual_wide) != 17 or actual_wide["region_id"].nunique() != 17:
        raise ValueError("official result must contain exactly 17 regions")

    actual_rows = []
    for row in actual_wide.itertuples(index=False):
        contest_votes = sum(float(getattr(row, column)) for _, column in CANDIDATES.values())
        all_valid_votes = contest_votes + sum(float(getattr(row, column)) for column in OTHER_COLUMNS)
        for slot, (candidate, column) in CANDIDATES.items():
            votes = float(getattr(row, column))
            actual_rows.append({
                "region_id": row.region_id,
                "region_name": row.region_name,
                "slot": slot,
                "candidate_name": candidate,
                "actual_votes": votes,
                "contest_votes": contest_votes,
                "all_valid_votes": all_valid_votes,
                "actual_contest_share": votes / contest_votes,
                "actual_raw_share": votes / all_valid_votes,
            })
    actual = pd.DataFrame(actual_rows)
    scored = forecast.merge(
        actual,
        on=["region_id", "slot", "candidate_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != 51:
        raise ValueError(f"forecast/result join produced {len(scored)} rows, expected 51")
    scored["error_pp"] = (scored["predicted_share"] - scored["actual_contest_share"]) * 100
    scored["absolute_error_pp"] = scored["error_pp"].abs()

    regional_weighted_mae = float(np.average(scored["absolute_error_pp"], weights=scored["contest_votes"]))
    regional_equal_region_mae = float(scored["absolute_error_pp"].mean())

    region_weights = actual_wide.assign(
        contest_votes=actual_wide[[column for _, column in CANDIDATES.values()]].sum(axis=1)
    ).set_index("region_id")["contest_votes"]
    national_rows = []
    total_contest_votes = float(region_weights.sum())
    total_all_valid_votes = float(actual_wide[[column for _, column in CANDIDATES.values()] + list(OTHER_COLUMNS)].sum().sum())
    for slot, (candidate, column) in CANDIDATES.items():
        prediction = forecast.loc[forecast["slot"].eq(slot)].set_index("region_id")["predicted_share"]
        ex_post_reaggregated_share = float((prediction * region_weights).sum() / total_contest_votes)
        predicted_share = float(frozen_national.loc[slot, "predicted_share"])
        actual_votes = float(actual_wide[column].sum())
        actual_contest_share = actual_votes / total_contest_votes
        national_rows.append({
            "slot": slot,
            "candidate_name": candidate,
            "predicted_share": predicted_share,
            "actual_contest_share": actual_contest_share,
            "actual_raw_share": actual_votes / total_all_valid_votes,
            "error_pp": (predicted_share - actual_contest_share) * 100,
            "absolute_error_pp": abs(predicted_share - actual_contest_share) * 100,
            "ex_post_actual_volume_reaggregated_prediction": ex_post_reaggregated_share,
        })
    national = pd.DataFrame(national_rows)
    national_mae = float(national["absolute_error_pp"].mean())

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(EVALUATION_DIR / "regional_scored.csv", index=False, encoding="utf-8-sig")
    national.to_csv(EVALUATION_DIR / "national_scored.csv", index=False, encoding="utf-8-sig")
    summary = {
        "schema": "pres_2025_v27_post_election_evaluation_v1",
        "status": "post_election_evaluation_not_model_selection",
        "forecast_cutoff": "2025-06-02",
        "forecast_sha256": _sha256(FORECAST),
        "national_forecast_sha256": _sha256(NATIONAL_FORECAST),
        "official_result_sha256": _sha256(ACTUAL),
        "scope": "A/B/C contest-normalized",
        "rows": 51,
        "regions": 17,
        "regional_contest_vote_weighted_point_mae_pp": regional_weighted_mae,
        "regional_equal_region_point_mae_pp": regional_equal_region_mae,
        "national_frozen_forecast_point_mae_pp": national_mae,
        "contest_votes": int(total_contest_votes),
        "all_valid_votes": int(total_all_valid_votes),
        "outcome_use": "evaluation only; no fitting, tuning, selection, or forecast mutation",
    }
    (EVALUATION_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
