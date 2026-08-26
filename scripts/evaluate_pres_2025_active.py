"""Score the active version's frozen 2025 D-1 forecast against the result.

`evaluate_pres_2025_v27.py` did this once, for V27, and was left alone while the
model moved to V31. That left a V27-era score published beside a V31 model,
which is the same defect this repository keeps finding elsewhere: a figure that
is checkable and no longer true of the thing it sits next to.

The declared boundary permits this. It says the realised outcome is "not added
to the model inputs, training panel, stage selection, thresholds, or
parameters" and "is read only by" the evaluation script. A read-only
post-election score is the disclosed exception, not a breach of it - and each
version since V27 was selected on structural grounds recorded *before* any 2025
score existed.

What must not happen is this score becoming a selection criterion. It is
published to be read, not to choose versions by, and no promotion in this
repository has ever cited it.

The evaluation itself is unchanged from V27's: the actual result is normalised
to the same A/B/C composition the model predicts, because the model forecasts
three shares summing to one and the ballot had more names on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POINTER = ROOT / "data/config/current_presidential_model.json"
#: The official count, transcribed once for the V27 evaluation. It is a
#: property of the election rather than of any model version, so it is read
#: from where it already lives instead of being copied per version.
ACTUAL = ROOT / "evaluations/pres_2025_v27/official_results.csv"

CANDIDATES = {
    "A": ("이재명", "lee_jaemyung_votes"),
    "B": ("김문수", "kim_moonsu_votes"),
    "C": ("이준석", "lee_junseok_votes"),
}
OTHER_COLUMNS = ("kwon_yeongguk_votes", "song_jinho_votes")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_state() -> dict[str, object]:
    """The commit the scored forecast was read at, and whether it was clean.

    An evaluation is only interpretable against a specific artifact. Recording
    the commit lets a reader recover which forecast produced the score without
    trusting that the file has not moved since.
    """

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD") or "unknown",
        "working_tree_clean": status == "",
        "uncommitted_paths": len(status.splitlines()) if status else 0,
    }


def _active_version() -> str:
    return str(json.loads(POINTER.read_text(encoding="utf-8"))["active_version"])


def evaluate(version: str | None = None) -> dict[str, object]:
    version = version or _active_version()
    forecast_dir = ROOT / f"outputs/prospective_pres_2025_{version}"
    forecast_path = forecast_dir / "prospective_predictions.csv"
    national_path = forecast_dir / "national_summary.csv"
    manifest_path = forecast_dir / "run_manifest.json"
    for path in (forecast_path, national_path, manifest_path, ACTUAL):
        if not path.is_file():
            raise FileNotFoundError(f"required file is missing: {path}")

    # the forecast being scored must itself declare that it never read the
    # outcome; scoring one that did would be circular
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("performance_metrics_computed") is not False:
        raise ValueError(f"{version} forecast manifest does not declare outcome-free generation")

    forecast = pd.read_csv(forecast_path, encoding="utf-8-sig")
    frozen_national = pd.read_csv(national_path, encoding="utf-8-sig").set_index("slot")
    actual_wide = pd.read_csv(ACTUAL, encoding="utf-8-sig")
    if len(actual_wide) != 17 or actual_wide["region_id"].nunique() != 17:
        raise ValueError("official result must contain exactly 17 regions")

    rows = []
    for row in actual_wide.itertuples(index=False):
        contest_votes = sum(float(getattr(row, column)) for _, column in CANDIDATES.values())
        all_valid = contest_votes + sum(float(getattr(row, column)) for column in OTHER_COLUMNS)
        for slot, (candidate, column) in CANDIDATES.items():
            votes = float(getattr(row, column))
            rows.append(
                {
                    "region_id": row.region_id,
                    "region_name": row.region_name,
                    "slot": slot,
                    "candidate_name": candidate,
                    "actual_votes": votes,
                    "contest_votes": contest_votes,
                    "all_valid_votes": all_valid,
                    "actual_contest_share": votes / contest_votes,
                    "actual_raw_share": votes / all_valid,
                }
            )
    actual = pd.DataFrame(rows)

    scored = forecast.merge(
        actual, on=["region_id", "slot", "candidate_name"], how="inner", validate="one_to_one"
    )
    if len(scored) != 51:
        raise ValueError(f"forecast/result join produced {len(scored)} rows, expected 51")
    scored["error_pp"] = (scored["predicted_share"] - scored["actual_contest_share"]) * 100
    scored["absolute_error_pp"] = scored["error_pp"].abs()

    regional_weighted = float(
        np.average(scored["absolute_error_pp"], weights=scored["contest_votes"])
    )
    regional_equal = float(scored["absolute_error_pp"].mean())

    region_weights = actual_wide.assign(
        contest_votes=actual_wide[[c for _, c in CANDIDATES.values()]].sum(axis=1)
    ).set_index("region_id")["contest_votes"]
    total_contest = float(region_weights.sum())
    total_all_valid = float(
        actual_wide[[c for _, c in CANDIDATES.values()] + list(OTHER_COLUMNS)].sum().sum()
    )

    national_rows = []
    for slot, (candidate, column) in CANDIDATES.items():
        prediction = forecast.loc[forecast["slot"].eq(slot)].set_index("region_id")["predicted_share"]
        predicted = float(frozen_national.loc[slot, "predicted_share"])
        actual_votes = float(actual_wide[column].sum())
        actual_share = actual_votes / total_contest
        national_rows.append(
            {
                "slot": slot,
                "candidate_name": candidate,
                "predicted_share": predicted,
                "actual_contest_share": actual_share,
                "actual_raw_share": actual_votes / total_all_valid,
                "error_pp": (predicted - actual_share) * 100,
                "absolute_error_pp": abs(predicted - actual_share) * 100,
                "ex_post_actual_volume_reaggregated_prediction": float(
                    (prediction * region_weights).sum() / total_contest
                ),
            }
        )
    national = pd.DataFrame(national_rows)

    out_dir = ROOT / f"evaluations/pres_2025_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "regional_scored.csv", index=False, encoding="utf-8-sig")
    national.to_csv(out_dir / "national_scored.csv", index=False, encoding="utf-8-sig")
    summary = {
        "schema": "pres_2025_post_election_evaluation_v3",
        "version": version,
        "status": "post_election_evaluation_not_model_selection",
        # The boundary, stated in the artifact rather than only in prose.
        "boundary": (
            f"{version.upper()} was frozen before this evaluation; 2025 outcomes "
            f"were not used for {version.upper()} model selection, "
            "parameterization, or promotion."
        ),
        "scored_forecast": {
            "artifact": f"outputs/prospective_pres_2025_{version}",
            "prospective_predictions_sha256": _sha256(forecast_path),
            "national_summary_sha256": _sha256(national_path),
            "run_manifest_sha256": _sha256(manifest_path),
            "repository": _repository_state(),
        },
        "forecast_cutoff": "2025-06-02",
        "forecast_sha256": _sha256(forecast_path),
        "national_forecast_sha256": _sha256(national_path),
        "official_result_sha256": _sha256(ACTUAL),
        "scope": "A/B/C contest-normalized",
        "rows": 51,
        "regions": 17,
        "regional_contest_vote_weighted_point_mae_pp": regional_weighted,
        "regional_equal_region_point_mae_pp": regional_equal,
        "national_frozen_forecast_point_mae_pp": float(national["absolute_error_pp"].mean()),
        "contest_votes": int(total_contest),
        "all_valid_votes": int(total_all_valid),
        "outcome_use": "evaluation only; no fitting, tuning, selection, or forecast mutation",
        "selection_note": (
            "no promotion in this repository has cited a 2025 score; each version "
            "was selected on structural grounds recorded before this was computed"
        ),
    }
    (out_dir / "summary.json").write_bytes(
        (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="defaults to the active pointer version")
    print(json.dumps(evaluate(parser.parse_args().version), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
