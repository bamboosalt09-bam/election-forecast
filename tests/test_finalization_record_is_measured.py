"""The finalization manifest must report the artifact, not assert about it.

V31's manifest shipped two false fields, both hardcoded in the finalizer so
re-running reproduced them:

    "feasibility_capped_elections": ["pres_2017"]
    "national_macro_not_worse_than_v30": True

The first names a cap V31 exists to remove — its own audit checks that no cap
column survives. The second claims a direction the experiment record beside it
contradicts: V31's national macro is `0.724291` against V30's `0.720437`, a
cost taken deliberately.

Neither touched a prediction. Both are worse than a stale number for that
reason: they are provenance, they are checkable, and they were false while
every other declaration agreed.

These tests read the manifest back against the artifacts it describes, so a
claim that stops being true fails here rather than being read by someone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"


def _active_dir() -> Path:
    return ROOT / str(json.loads(POINTER.read_text(encoding="utf-8"))["output"])


def _manifest() -> dict:
    return json.loads((_active_dir() / "finalization_manifest.json").read_text(encoding="utf-8"))


def test_the_manifest_claims_no_cap_that_the_artifact_does_not_show() -> None:
    manifest = _manifest()
    expansion = manifest.get("multiplicative_dispersion_expansion")
    if expansion is None:  # a future version may name its transform differently
        return
    audit = pd.read_csv(
        _active_dir() / "multiplicative_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    capped_in_artifact = (
        sorted(audit.loc[audit["feasibility_capped"].astype(bool), "election_id"])
        if "feasibility_capped" in audit.columns
        else []
    )
    assert list(expansion.get("feasibility_capped_elections", [])) == capped_in_artifact, (
        "the manifest names capped elections the audit does not show"
    )


def test_the_preserved_flags_match_the_measurement() -> None:
    manifest = _manifest()
    expansion = manifest.get("multiplicative_dispersion_expansion")
    if expansion is None:
        return
    audit = pd.read_csv(
        _active_dir() / "multiplicative_dispersion_expansion_audit.csv", encoding="utf-8-sig"
    )
    worst = float(audit["max_candidate_level_shift_pp"].abs().max())
    assert expansion["candidate_national_level_preserved"] == (worst < 1e-9)
    assert float(expansion["worst_candidate_level_shift_pp"]) == worst

    predictions = pd.read_csv(
        _active_dir() / "nested_predictions.csv", encoding="utf-8-sig", low_memory=False
    )
    sums_hold = bool(
        predictions.groupby(["election_id", "region_id"])["layer_pred"]
        .sum()
        .sub(1.0)
        .abs()
        .lt(1e-12)
        .all()
    )
    assert expansion["regional_composition_preserved"] == sums_hold
    assert float(expansion["minimum_predicted_share"]) == float(predictions["layer_pred"].min())


def test_the_verification_block_states_changes_rather_than_directions() -> None:
    """A boolean asserting 'not worse' is the shape that went false."""

    verification = _manifest()["verification"]
    forbidden = [key for key in verification if "not_worse" in key or "improved" in key]
    assert not forbidden, (
        f"{forbidden} assert a direction; record the measured change instead"
    )

    metrics = _manifest()["metrics"]
    for axis in ("national", "regional"):
        key = f"{axis}_macro_change_vs_v30_pp"
        assert key in verification, f"{key} is missing"
        assert isinstance(verification[key], float)
    # and the recorded change must reconstruct the published figure
    assert (
        abs(
            (0.7204374174124484 + verification["national_macro_change_vs_v30_pp"])
            - metrics["national_equal_election_macro_mae_pp"]
        )
        < 1e-12
    )
