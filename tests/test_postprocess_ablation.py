"""Guards for the postprocess ablation harness and what it found.

The durable finding is structural rather than numeric: the three layers barely
co-occur, so most of the apparent commutativity in the results table is an
artifact of the panel rather than a property of the transforms. These tests pin
the grid's shape and that co-occurrence structure, so a future change that makes
the layers overlap more - or less - becomes visible rather than silent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import evaluate_postprocess_ablation as ablation
from scripts.active_model_pointer import active_output_dir

SCORED_DIR = active_output_dir()
AUDITS = {
    "veto": "strong_incumbent_veto_audit.csv",
    "ceiling": "third_candidate_lineage_ceiling_audit.csv",
    "refusal": "weak_same_lane_refusal_audit.csv",
}


def _fired(directory: Path, name: str) -> set[str]:
    path = directory / AUDITS[name]
    if not path.exists():
        pytest.skip(f"{path.name} is not present")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return set()
    return {str(value) for value in frame["election_id"]}


def test_the_grid_covers_every_subset_and_every_ordering() -> None:
    cells = ablation.cells()
    labels = [label for label, _ in cells]
    assert len(cells) == 16, "1 empty + 3 singletons + 3 pairs x 2 + 6 triples"
    assert len(set(labels)) == 16, "cell labels must be unique"
    assert "none" in labels
    assert ">".join(ablation.LAYER_ORDER) in labels, "the shipped order must be a cell"

    triples = [order for _, order in cells if len(order) == 3]
    assert len(triples) == 6
    assert len({frozenset(order) for order in triples}) == 1, "one subset, six orders"


def test_every_ordering_is_a_permutation_of_its_subset() -> None:
    for label, order in ablation.cells():
        assert len(set(order)) == len(order), f"{label} repeats a layer"
        assert set(order).issubset(set(ablation.LAYER_ORDER)), label


def test_the_layers_barely_co_occur_on_the_scored_panel() -> None:
    """The panel has one two-layer election and no three-layer election.

    This is why the veto appears to commute with the other two: it never shares
    an election with them, so no ordering involving it can be exercised.
    """

    fired = {name: _fired(SCORED_DIR, name) for name in AUDITS}
    counts: dict[str, int] = {}
    for name, elections in fired.items():
        for election in elections:
            counts[election] = counts.get(election, 0) + 1

    assert max(counts.values()) == 2, "a three-layer scored election would change this"
    two_layer = sorted(e for e, n in counts.items() if n == 2)
    assert two_layer == ["pres_2002"]

    # the veto shares no scored election with either other layer
    assert not fired["veto"] & fired["ceiling"]
    assert not fired["veto"] & fired["refusal"]
    # ceiling and refusal are the one genuinely exercised pair
    assert fired["ceiling"] & fired["refusal"] == {"pres_2002"}


def test_each_structural_layer_rests_on_at_most_two_scored_elections() -> None:
    """The activation counts are the honest measure of the evidence behind each rule."""

    for name in AUDITS:
        elections = _fired(SCORED_DIR, name)
        assert len(elections) <= 2, f"{name} fires on {sorted(elections)}"
        assert elections, f"{name} never fires; it cannot be evidenced at all"


def test_applying_no_layers_returns_the_frame_unchanged() -> None:
    frame = pd.DataFrame(
        {
            "election_id": ["pres_2002"] * 3,
            "region_id": ["sido_11"] * 3,
            "slot": ["A", "B", "C"],
            "source_slot": ["A", "B", "C"],
            "candidate_name": ["a", "b", "c"],
            "layer_pred": [0.45, 0.40, 0.15],
        }
    )
    result = ablation.apply_sequence(frame, ())
    pd.testing.assert_frame_equal(result, frame)
