"""Audit the frozen public V26 pointer and the V23/V24/V25 rollback boundaries.

The input, prediction, interval and finalization checks are identical in shape
to V25's, so they are reused by repointing that module's ``ACTIVE_DIR`` rather
than copied. Only the pointer expectations and the rollback set are version
specific, and the rollback set grows: promoting V26 makes V25 a rollback
reference, so its prediction hash is now checked here too.
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_public_active_presidential_model_v25 as v25_audit  # noqa: E402

ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v26"
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
V23_SHA256 = v25_audit.V23_SHA256
V24_SHA256 = v25_audit.V24_SHA256
V25_SHA256 = v25_audit.V25_SHA256
CANONICAL_DOCUMENT = "docs/FINAL_MODEL_V26_20260822.md"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _active_v26() -> Iterator[None]:
    """Run the shared V25 checks against the V26 output directory."""

    original = v25_audit.ACTIVE_DIR
    v25_audit.ACTIVE_DIR = ACTIVE_DIR
    try:
        yield
    finally:
        v25_audit.ACTIVE_DIR = original


def _audit_rollbacks() -> dict[str, object]:
    for version, expected in (
        ("v23", V23_SHA256),
        ("v24", V24_SHA256),
        ("v25", V25_SHA256),
    ):
        path = ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv"
        _require(_sha256(path) == expected, f"{version.upper()} rollback prediction drift")
    return {
        "v23_rollback_sha256": V23_SHA256,
        "v24_rollback_sha256": V24_SHA256,
        "v25_rollback_sha256": V25_SHA256,
    }


def _audit_pointer() -> dict[str, object]:
    pointer = json.loads(POINTER.read_text(encoding="utf-8"))
    expected = {
        "active_version": "v26",
        "runner": "scripts/run_active_presidential_model_v26.py",
        "output": "outputs/active_presidential_nested_v26",
        "canonical_document": CANONICAL_DOCUMENT,
        "finalization_manifest": "outputs/active_presidential_nested_v26/finalization_manifest.json",
        "config": "data/config/active_presidential_model_v23.json",
        "base_config_version": "v23",
        "predecessor": "v25",
    }
    for key, value in expected.items():
        _require(pointer.get(key) == value, f"active pointer mismatch: {key}")
    _require(
        pointer.get("post_2022_outcomes_used") is False,
        "pointer permits post-2022 outcomes",
    )
    return {"active_version": "v26"}


def _audit_graded_intensity() -> dict[str, object]:
    """The two changes that define V26 must both be present and one-sided."""

    import pandas as pd

    from presidential_issue_engine.mega_issue_intensity_ladder import CRISIS_INTENSITY

    base = pd.read_csv(
        ROOT / "outputs/automatic_controls_v23/mega_issue_intensity.csv",
        encoding="utf-8-sig",
    ).set_index("election_id")["mega_issue_intensity"]
    graded = pd.read_csv(
        ROOT / "outputs/automatic_controls_v26/mega_issue_intensity.csv",
        encoding="utf-8-sig",
    ).set_index("election_id")["mega_issue_intensity"]
    _require(set(base.index) == set(graded.index), "graded intensity changed the panel")
    for election in base.index:
        floor, raised = float(base[election]), float(graded[election])
        _require(raised >= floor - 1e-9, f"graded intensity lowered {election}")
        _require(raised <= CRISIS_INTENSITY + 1e-9, f"graded intensity exceeded ceiling: {election}")
        if floor >= CRISIS_INTENSITY - 1e-9:
            _require(
                abs(raised - floor) <= 1e-9,
                f"graded intensity moved an election already at the ceiling: {election}",
            )
    intermediate = sum(1 for e in base.index if 1.0 < float(graded[e]) < CRISIS_INTENSITY)
    _require(intermediate > 0, "graded intensity reached no intermediate rung")
    return {"graded_intermediate_elections": intermediate}


def _audit_predictions() -> dict[str, object]:
    """Same shape as V25's, against the V26 hash recorded at finalization.

    V25's version pins its own prediction hash, so it cannot simply be
    repointed; the panel invariants below are the version-agnostic part.
    """

    import numpy as np
    import pandas as pd

    finalization = json.loads(
        (ACTIVE_DIR / "finalization_manifest.json").read_text(encoding="utf-8")
    )
    expected = str(finalization["verification"]["v26_prediction_hash"])
    path = ACTIVE_DIR / "nested_predictions.csv"
    _require(_sha256(path) == expected, "V26 canonical prediction drift")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    _require(len(frame) == 232, "V26 prediction row count is not 232")
    _require(
        set(frame["election_id"].astype(str)) == set(v25_audit.ORDER),
        "V26 election panel drift",
    )
    _require(
        not frame["election_id"].astype(str).str.contains("2025").any(),
        "2025 row in V26 history",
    )
    totals = frame.groupby(["election_id", "region_id"])["layer_pred"].sum()
    _require(np.allclose(totals, 1.0, atol=1e-12), "V26 predictions are not compositional")
    summary = json.loads((ACTIVE_DIR / "summary.json").read_text(encoding="utf-8"))
    _require(summary["metrics"]["rows"] == 232, "V26 summary row drift")
    _require(np.isclose(summary["metrics"]["winner_accuracy"], 0.8), "V26 winner gate drift")
    _require(
        summary["metrics"]["variant"] == "v26_graded_mega_intensity_event_aligned",
        "V26 variant stamp drift",
    )
    refusal = pd.read_csv(
        ACTIVE_DIR / "weak_same_lane_refusal_audit.csv", encoding="utf-8-sig"
    )
    _require(
        set(refusal["recipient_weight_mode"].astype(str)) == {"prediction_tilted"},
        "weak-C route drift",
    )
    return {"v26_prediction_rows": 232, "v26_prediction_sha256": expected}


def _audit_intervals() -> dict[str, object]:
    """V25's version pins its own model_version, so the checks are repeated here."""

    import pandas as pd

    manifest = json.loads(
        (ACTIVE_DIR / "predictive_interval_manifest.json").read_text(encoding="utf-8")
    )
    _require(manifest["model_version"] == "v26", "interval model version drift")
    _require(manifest["post_2022_outcomes_used"] is False, "post-2022 interval outcome used")
    _require(
        manifest["target_outcomes_used_to_construct_bounds"] is False,
        "target built interval",
    )
    detail = pd.read_csv(ACTIVE_DIR / "national_predictive_intervals.csv")
    _require(len(detail) == 44, "interval row count is not 44")
    _require(
        set(detail["nominal_level"].round(2)) == {0.5, 0.8, 0.9, 0.95},
        "interval levels drift",
    )
    position = {election: index for index, election in enumerate(v25_audit.ORDER)}
    for row in detail.itertuples(index=False):
        _require(
            all(
                position[election] < position[row.election_id]
                for election in str(row.training_elections).split("|")
            ),
            f"nonchronological interval training: {row.election_id}",
        )
    return {"interval_candidate_level_rows": len(detail)}


def _audit_finalization() -> dict[str, object]:
    """V25's version pins its own active_version, so the checks are repeated here."""

    manifest = json.loads(
        (ACTIVE_DIR / "finalization_manifest.json").read_text(encoding="utf-8")
    )
    _require(manifest["active_version"] == "v26", "finalization version drift")
    _require(
        manifest["post_2022_outcomes_used"] is False,
        "finalization outcome boundary drift",
    )
    for record in manifest["artifacts"]:
        path = ROOT / record["path"]
        _require(path.is_file(), f"finalized artifact missing: {record['path']}")
        _require(
            v25_audit._artifact_sha256(path, str(record.get("hash_mode", "raw_bytes")))
            == record["sha256"],
            f"finalized artifact drift: {record['path']}",
        )
    return {"finalized_artifacts": len(manifest["artifacts"])}


def main() -> None:
    results: dict[str, object] = {}
    results.update(_audit_rollbacks())
    results.update(_audit_pointer())
    results.update(_audit_graded_intensity())
    results.update(_audit_predictions())
    results.update(_audit_intervals())
    results.update(_audit_finalization())
    with _active_v26():
        results.update(v25_audit._audit_inputs())
    print("[active V26 public audit: PASS]")
    for key, value in results.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
