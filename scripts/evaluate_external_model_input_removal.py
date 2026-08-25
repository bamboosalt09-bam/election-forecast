"""Re-measure what removing the external-model-derived aggregate costs.

`data/raw/auto_issue_seed/candidate_issue_profile.csv` is the one active input
that descends from an external encoder. It is retained and disclosed rather
than dropped, and the stated reason is that removal is not cost-free. The
number backing that claim was measured on V27 in
`EXPERIMENT_REMOVE_EXTERNAL_MODEL_OVERLAY_20260822.md` and never repeated, so
by V30 the documents were quoting a cost against a baseline that no longer
existed.

The removal is a *schema-only injection*: the file is replaced by one with the
same header and no rows. That is stricter than deleting it, because a missing
file changes control flow while an empty one exercises the same code path with
no evidence in it.

It is done by swapping the file on disk rather than by patching the paths that
read it. The original V27 experiment first reported no change at all, because
several readers loaded the same file independently and patching one of them
left the others live. Swapping the file cannot miss a reader.

Read-only with respect to the model: nothing here promotes, and the frozen V30
artifact is not touched. Run:

    python scripts/evaluate_external_model_input_removal.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "data" / "raw" / "auto_issue_seed" / "candidate_issue_profile.csv"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v30"
OUTPUT_DIR = ROOT / "outputs" / "external_model_input_removal"
RUNNER = "scripts/run_active_presidential_model_v30.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_only(source: Path, destination: Path) -> None:
    """Same columns, no rows."""

    frame = pd.read_csv(source, encoding="utf-8-sig")
    frame.head(0).to_csv(destination, index=False, encoding="utf-8-sig")


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    frame = frame.copy()
    frame["abs_pp"] = (frame["layer_pred"] - frame["actual"]).abs() * 100.0
    regional: list[float] = []
    national: list[float] = []
    winners = 0
    for _, group in frame.groupby("election_id"):
        regional.append(float(np.average(group["abs_pp"], weights=group["contest_votes"])))
        errors = []
        levels: dict[str, float] = {}
        actuals: dict[str, float] = {}
        for slot, rows in group.groupby("slot"):
            weights = rows["contest_votes"]
            predicted = float(np.average(rows["layer_pred"], weights=weights))
            realised = float(np.average(rows["actual"], weights=weights))
            errors.append(abs(predicted - realised) * 100.0)
            levels[str(slot)] = predicted
            actuals[str(slot)] = realised
        national.append(float(np.mean(errors)))
        winners += int(max(levels, key=levels.get) == max(actuals, key=actuals.get))
    return {
        "regional_equal_election_macro_mae_pp": float(np.mean(regional)),
        "national_equal_election_macro_mae_pp": float(np.mean(national)),
        "winner_accuracy": winners / len(regional),
        "rows": int(len(frame)),
    }


def _run(output_dir: Path) -> pd.DataFrame:
    result = subprocess.run(
        [sys.executable, RUNNER, "--output-dir", str(output_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"removal run failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
    return pd.read_csv(
        output_dir / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-run", action="store_true", help="retain the removal run directory")
    args = parser.parse_args()

    if not ACTIVE_DIR.joinpath("nested_predictions.csv").is_file():
        raise SystemExit("the active V30 artifact is missing; nothing to compare against")

    retained = _metrics(
        pd.read_csv(ACTIVE_DIR / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False)
    )

    original_hash = _sha256(PROFILE)
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        backup = scratch_path / "candidate_issue_profile.original.csv"
        shutil.copy2(PROFILE, backup)
        run_dir = (
            OUTPUT_DIR / "removed_run" if args.keep_run else scratch_path / "removed_run"
        )
        try:
            _schema_only(backup, PROFILE)
            removed = _metrics(_run(run_dir))
        finally:
            shutil.copy2(backup, PROFILE)
        # the input is a tracked file; prove it came back untouched
        if _sha256(PROFILE) != original_hash:
            raise RuntimeError("the profile was not restored; restore it from Git before rerunning")

    report = {
        "schema": "external_model_input_removal_v1",
        "measured_on": "v30",
        "input": "data/raw/auto_issue_seed/candidate_issue_profile.csv",
        "input_sha256": original_hash,
        "method": "schema-only injection: same header, zero rows, swapped on disk",
        "post_2022_outcomes_used": False,
        "retained": retained,
        "removed": removed,
        "change": {
            key: removed[key] - retained[key]
            for key in ("regional_equal_election_macro_mae_pp",
                        "national_equal_election_macro_mae_pp",
                        "winner_accuracy")
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "removal_summary.json").write_bytes(
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
