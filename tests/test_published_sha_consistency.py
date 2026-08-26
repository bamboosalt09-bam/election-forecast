"""Every place that publishes the active prediction hash must publish the same one.

The version audit pinned the hash in three source files. It did not pin the one
`REPRODUCIBILITY.md` prints for a reader, and that document spent a release
naming V30's hash under a "V31 active" heading while every other declaration
agreed with the pointer. Fixing that one line fixed one line.

This makes it a property instead: whatever the pointer's artifact hashes to,
every disclosure surface must name that value, and a rollback hash may appear
only where it is labelled as a rollback. Adding a new disclosure surface later
means adding it here, which is the point - a surface nobody checks is how the
last one went wrong.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "data" / "config" / "current_presidential_model.json"


def _pointer() -> dict:
    return json.loads(POINTER.read_text(encoding="utf-8"))


def canonical_hash() -> str:
    pointer = _pointer()
    artifact = ROOT / str(pointer["output"]) / "nested_predictions.csv"
    return hashlib.sha256(artifact.read_bytes()).hexdigest()


def rollback_hashes() -> set[str]:
    """Hashes that may legitimately appear, labelled as predecessors."""

    found: set[str] = set()
    for directory in sorted((ROOT / "outputs").glob("active_presidential_nested_v*")):
        predictions = directory / "nested_predictions.csv"
        if predictions.is_file():
            found.add(hashlib.sha256(predictions.read_bytes()).hexdigest())
    return found - {canonical_hash()}


def test_the_pointer_publishes_the_hash_of_its_own_artifact() -> None:
    assert _pointer()["prediction_sha256"] == canonical_hash()


def test_the_finalization_manifest_agrees() -> None:
    pointer = _pointer()
    manifest = json.loads(
        (ROOT / str(pointer["output"]) / "finalization_manifest.json").read_text(encoding="utf-8")
    )
    published = manifest["verification"].get(f"{pointer['active_version']}_prediction_hash")
    assert published == canonical_hash(), "the finalization manifest names another hash"


@pytest.mark.parametrize(
    "relative,pattern",
    [
        ("setup.py", r'"frozen_prediction_sha256": "([0-9a-f]{64})"'),
        ("scripts/audit_distribution_artifacts.py", r'FROZEN_V\d+_SHA256 = \(\s*"([0-9a-f]{64})"'),
        ("scripts/audit_current_public_surface.py", r'V\d+_SHA256 = "([0-9a-f]{64})"'),
        ("docs/REPRODUCIBILITY.md", r"active:\n[^\n]+\nSHA-256: ([0-9a-f]{64})"),
    ],
)
def test_each_disclosure_surface_names_the_canonical_hash(relative: str, pattern: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8")
    found = re.search(pattern, text)
    assert found, f"{relative} publishes no active prediction hash"
    assert found.group(1) == canonical_hash(), (
        f"{relative} publishes {found.group(1)[:12]}; the pointer's artifact "
        f"hashes to {canonical_hash()[:12]}"
    )


def test_a_rollback_hash_never_appears_as_the_active_one() -> None:
    """The exact failure: a predecessor's hash under an 'active' label."""

    rollbacks = rollback_hashes()
    assert rollbacks, "no rollback artifacts found; the check would be vacuous"

    text = (ROOT / "docs/REPRODUCIBILITY.md").read_text(encoding="utf-8")
    found = re.search(r"active:\n[^\n]+\nSHA-256: ([0-9a-f]{64})", text)
    assert found and found.group(1) not in rollbacks, (
        "the document labels a rollback hash as the active artifact"
    )


def test_rollback_hashes_are_still_disclosed_somewhere() -> None:
    """Removing them would be the opposite mistake."""

    text = (ROOT / "docs/REPRODUCIBILITY.md").read_text(encoding="utf-8")
    published = set(re.findall(r"SHA-256: ([0-9a-f]{64})", text))
    assert rollback_hashes() & published, (
        "no rollback prediction hash is published; the chain is the evidence"
    )
