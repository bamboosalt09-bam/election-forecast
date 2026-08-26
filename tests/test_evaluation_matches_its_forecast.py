"""A published evaluation must still describe the forecast it names.

`evaluations/pres_2025_v32/` sat tracked in this repository scoring a forecast
that had been regenerated after the run: it recorded `forecast_sha256:
e19ef33d…` while the committed artifact hashed to `d3932936…`. Nothing read the
field back, so a score belonging to no surviving forecast was published.

The evaluator has always recorded provenance. What it lacked was anything that
reads it. That is the general shape of the defect - a field written once and
never checked is indistinguishable from a field that is wrong.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = ROOT / "evaluations"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summaries() -> list[Path]:
    if not EVALUATIONS.is_dir():
        return []
    return sorted(EVALUATIONS.glob("pres_2025_*/summary.json"))


@pytest.mark.parametrize("summary_path", _summaries(), ids=lambda p: p.parent.name)
def test_the_evaluation_still_matches_the_forecast_it_scored(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # the directory name carries the version on every schema; the `version`
    # key only appeared in v2, and V27's evaluation predates it
    version = summary.get("version") or summary_path.parent.name.rsplit("_", 1)[-1]
    forecast = ROOT / f"outputs/prospective_pres_2025_{version}/prospective_predictions.csv"
    if not forecast.is_file():
        pytest.skip(f"the {version} forecast artifact is not present in this tree")

    recorded = str(summary["forecast_sha256"])
    actual = _sha256(forecast)
    assert recorded == actual, (
        f"{summary_path.parent.name} scores a forecast that no longer exists: it "
        f"records {recorded[:12]} and {forecast.relative_to(ROOT).as_posix()} "
        f"hashes to {actual[:12]}. Re-run "
        f"`python scripts/evaluate_pres_2025_active.py --version {version}` or "
        "restore the artifact; do not leave a published score attached to a "
        "forecast that was regenerated under it."
    )


@pytest.mark.parametrize("summary_path", _summaries(), ids=lambda p: p.parent.name)
def test_the_evaluation_declares_its_outcome_boundary(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "post_election_evaluation_not_model_selection"
    assert "no fitting" in str(summary["outcome_use"])


def test_the_current_evaluation_records_the_commit_it_ran_at() -> None:
    """Only the newest schema carries this; older ones are preserved as written."""

    summary_path = EVALUATIONS / "pres_2025_v32" / "summary.json"
    if not summary_path.is_file():
        pytest.skip("the V32 evaluation is not present in this tree")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema"] == "pres_2025_post_election_evaluation_v3"
    scored = summary["scored_forecast"]
    assert scored["artifact"] == "outputs/prospective_pres_2025_v32"
    assert len(str(scored["repository"]["commit"])) == 40
    boundary = str(summary["boundary"])
    assert "frozen before this evaluation" in boundary
    assert "not used for V32 model selection" in boundary


def test_v31s_evaluation_is_not_replaced_by_v32s() -> None:
    """Two forecasts, two evaluations. Neither supersedes the other."""

    for version in ("v31", "v32"):
        path = EVALUATIONS / f"pres_2025_{version}" / "summary.json"
        if not path.is_file():
            pytest.skip(f"the {version} evaluation is not present in this tree")
    v31 = json.loads((EVALUATIONS / "pres_2025_v31" / "summary.json").read_text(encoding="utf-8"))
    v32 = json.loads((EVALUATIONS / "pres_2025_v32" / "summary.json").read_text(encoding="utf-8"))
    assert v31["version"] == "v31" and v32["version"] == "v32"
    assert v31["forecast_sha256"] != v32["forecast_sha256"], (
        "the two evaluations score the same bytes, so one of them names the "
        "wrong forecast"
    )
    # the same official count on both sides, or the comparison is meaningless
    assert v31["official_result_sha256"] == v32["official_result_sha256"]
