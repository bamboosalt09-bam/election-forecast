"""Guards for the V26 promotion: pointer, boundary, and the two changes.

V26 differs from V25 in exactly two places - a graded mega-issue intensity and
the event-class alignment on the scored path. These tests pin that the pointer
records the promotion, that the frozen predecessors did not move, and that the
graded table is one-sided and reaches the intermediate rungs the promotion
exists to make reachable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from presidential_issue_engine.mega_issue_intensity_ladder import (
    CRISIS_INTENSITY,
    ladder_intensity,
)

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"
ACTIVE_DIR = ROOT / "outputs" / "active_presidential_nested_v26"
V23_SHA256 = "dbcf596308abf026b35a007b121d13e4bef35755aa4d4a9fe47cc95c1484204b"
V24_SHA256 = "edefb5e0f24cfa1ad4d2d5e7934e7158de2113cdf9cb11e42853e208cd00726a"
V25_SHA256 = "218e5d6c732f65c5c9259b38aabff0f381f2df9ced970a136d1a954a2fb51a1b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer() -> dict[str, object]:
    return json.loads(POINTER.read_text(encoding="utf-8"))


def test_v26_remains_in_the_declared_rollback_chain() -> None:
    pointer = _pointer()
    assert pointer["active_version"] == "v31"
    assert pointer["predecessor"] == "v30"
    assert pointer["rollback_pointer"] == (
        "outputs/active_presidential_nested_v30/finalization_manifest.json"
    )
    # walk the chain rather than naming one link, so a further promotion keeps
    # this test honest instead of silently only checking the newest hop
    version = "v28"
    while version != "v26":
        manifest = json.loads(
            (ROOT / f"outputs/active_presidential_nested_{version}/finalization_manifest.json")
            .read_text(encoding="utf-8")
        )
        version = manifest["rollback"]["version"]
    assert version == "v26"
    assert _sha256(ACTIVE_DIR / "nested_predictions.csv") == (
        "9b66b813f97c3c2804a178ebb5b9104fa4a58553c75812f75affbb3b17773dd3"
    )


def test_promoting_v26_did_not_move_any_frozen_predecessor() -> None:
    for version, expected in (
        ("v23", V23_SHA256),
        ("v24", V24_SHA256),
        ("v25", V25_SHA256),
    ):
        path = ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv"
        assert _sha256(path) == expected, version


def test_v26_improves_the_panel_without_losing_a_winner() -> None:
    pointer = _pointer()
    assert pointer["prediction_rows"] == 232
    assert pointer["winner_accuracy"] == pytest.approx(0.8)
    # V25's published figures; V26 must not be worse on either metric
    assert pointer["regional_equal_election_macro_mae_pp"] < 2.773943232022332
    assert pointer["national_equal_election_macro_mae_pp"] < 0.9896196354753938


def test_the_graded_control_matches_the_ladder_the_engine_produces() -> None:
    """The written control must be reproducible from the module, not hand-edited."""

    diagnostics = pd.read_csv(
        ROOT / "outputs/automatic_controls_v22/mega_issue_taxonomy_audit.csv",
        encoding="utf-8-sig",
    )
    base = pd.read_csv(
        ROOT / "outputs/automatic_controls_v23/mega_issue_intensity.csv",
        encoding="utf-8-sig",
    )
    written = pd.read_csv(
        ROOT / "outputs/automatic_controls_v26/mega_issue_intensity.csv",
        encoding="utf-8-sig",
    )
    expected = ladder_intensity(base, diagnostics)
    pd.testing.assert_series_equal(
        written.set_index("election_id")["mega_issue_intensity"].sort_index(),
        expected.set_index("election_id")["mega_issue_intensity"].sort_index(),
        check_names=False,
    )


def test_the_graded_control_is_one_sided_and_reaches_intermediate_rungs() -> None:
    base = pd.read_csv(
        ROOT / "outputs/automatic_controls_v23/mega_issue_intensity.csv",
        encoding="utf-8-sig",
    ).set_index("election_id")["mega_issue_intensity"]
    graded = pd.read_csv(
        ROOT / "outputs/automatic_controls_v26/mega_issue_intensity.csv",
        encoding="utf-8-sig",
    ).set_index("election_id")["mega_issue_intensity"]

    assert set(base.index) == set(graded.index)
    for election in base.index:
        floor, raised = float(base[election]), float(graded[election])
        assert raised >= floor - 1e-9, election
        assert raised <= CRISIS_INTENSITY + 1e-9, election
        if floor >= CRISIS_INTENSITY - 1e-9:
            assert raised == pytest.approx(floor), f"{election} was already at the ceiling"

    intermediate = [e for e in base.index if 1.0 < float(graded[e]) < CRISIS_INTENSITY]
    assert intermediate, "the promotion exists to make intermediate rungs reachable"
    # 2017 is the one election already saturated; it must not be among them
    assert "pres_2017" not in intermediate


def test_the_v26_runner_restores_v25_state_on_exit() -> None:
    """Importing and running V26 must not leave V25 patched for other callers."""

    from presidential_issue_engine import mega_issue_adjustment
    from scripts import run_active_presidential_model_v25 as v25
    from scripts import run_active_presidential_model_v26 as v26

    before = (
        v25.AUTOMATIC_DIR,
        v25.FINAL_VARIANT,
        mega_issue_adjustment.compile_direct_mega_scores,
    )
    with v26.graded_mega_runtime():
        assert v25.AUTOMATIC_DIR == v26.AUTOMATIC_DIR
        assert v25.FINAL_VARIANT == v26.FINAL_VARIANT
        assert mega_issue_adjustment.compile_direct_mega_scores is not before[2]
    assert (
        v25.AUTOMATIC_DIR,
        v25.FINAL_VARIANT,
        mega_issue_adjustment.compile_direct_mega_scores,
    ) == before


def test_the_finalization_record_declares_the_selection_disclosure() -> None:
    """The promotion was selected on the scored panel; the record must say so."""

    promotion = json.loads(
        (ACTIVE_DIR / "promotion_manifest.json").read_text(encoding="utf-8")
    )
    assert promotion["active_version"] == "v26"
    assert promotion["predecessor"] == "v25"
    assert promotion["post_2022_outcomes_used"] is False
    assert "selection_disclosure" in promotion
    assert "scored outcomes" in str(promotion["selection_disclosure"])
    assert "graded_intensity_without_event_class_alignment" in promotion["rejected_scope"]
