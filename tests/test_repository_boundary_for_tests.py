"""Fail locally when a test depends on a file the public checkout does not have.

Bulk sources, caches and most of ``outputs/`` are deliberately untracked, so a
test that reads one of them passes on the machine that produced it and fails in
CI with FileNotFoundError. That has happened twice.

Both times the path came from a module constant rather than a literal, so this
guard resolves references of the form ``module.CONSTANT`` as well as literal
strings: it imports each test module, finds the ``Path`` constants reachable
through the modules it imported, and requires every one that is untracked and
inside a partly-tracked directory to be guarded by an explicit existence check.

It reads no data - it inspects the test sources and the imported namespaces
only - so it stays cheap.
"""

from __future__ import annotations

import importlib
import re
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
# Directories whose contents are only partly tracked. A path inside one of
# these is the risk; everything else is source or committed fixture data.
WATCHED_ROOTS = ("outputs", "archives")
WATCHED_NESTED = (Path("data") / "raw" / "official_sources",)
LITERAL = re.compile(r'"((?:outputs|data|archives)/[^"\n]+?\.(?:csv|json|parquet))"')
REFERENCE = re.compile(r"\b([a-zA-Z_][\w]*)\.([A-Z][A-Z0-9_]{2,})\b")
# A module that guards its risky reads with one of these is fine.
SKIP_MARKERS = ("pytest.skip", ".exists()", "skipif")


def _tracked() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    )
    return {
        ROOT / entry
        for entry in result.stdout.decode("utf-8").split("\0")
        if entry
    }


def _watched(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        return False
    if relative.parts and relative.parts[0] in WATCHED_ROOTS:
        return True
    return any(
        relative.is_relative_to(nested) for nested in WATCHED_NESTED
    )


def _is_tracked(path: Path, tracked: set[Path]) -> bool:
    """A file must be tracked; a directory needs only tracked content beneath it.

    ``git ls-files`` names files, so a directory constant would otherwise always
    look untracked even when the checkout carries everything the test reads.
    """

    if path in tracked:
        return True
    return any(entry.is_relative_to(path) for entry in tracked)


def _referenced_paths(module: ModuleType, text: str) -> set[Path]:
    """Paths reachable as ``alias.CONSTANT`` from the test module's namespace."""

    found: set[Path] = set()
    for alias, constant in REFERENCE.findall(text):
        target = getattr(module, alias, None)
        if not isinstance(target, ModuleType):
            continue
        value = getattr(target, constant, None)
        if isinstance(value, Path):
            found.add(value)
    return found


@pytest.mark.parametrize(
    "source", sorted(TESTS.glob("test_*.py")), ids=lambda path: path.name
)
def test_test_modules_do_not_read_untracked_repository_paths(source: Path) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    try:
        module = importlib.import_module(f"tests.{source.stem}")
    except Exception:  # noqa: BLE001 - an unimportable module fails on its own
        module = None

    # A literal carrying a format placeholder is a template, not a path: it
    # cannot be resolved statically, so it is left to the module's own guards.
    candidates = {
        ROOT / literal
        for literal in LITERAL.findall(text)
        if "{" not in literal and "}" not in literal
    }
    if module is not None:
        candidates |= _referenced_paths(module, text)

    tracked = _tracked()
    untracked = sorted(
        str(path.resolve().relative_to(ROOT))
        for path in candidates
        if _watched(path) and not _is_tracked(path.resolve(), tracked)
    )
    if not untracked:
        return
    assert any(marker in text for marker in SKIP_MARKERS), (
        f"{source.name} reads untracked paths without an existence guard: "
        + ", ".join(untracked)
    )


def test_the_guard_resolves_both_literals_and_module_constants() -> None:
    """A guard that silently matched nothing would make the parametrised test vacuous."""

    assert LITERAL.findall('pd.read_csv("outputs/dir/file.csv")') == [
        "outputs/dir/file.csv"
    ]
    assert _watched(ROOT / "outputs" / "dir" / "file.csv")
    assert _watched(ROOT / "data" / "raw" / "official_sources" / "x.csv")
    assert not _watched(ROOT / "data" / "raw" / "candidates.csv")

    from scripts import evaluate_v25_intensity_ladder as ladder

    namespace = ModuleType("sample")
    namespace.ladder = ladder  # type: ignore[attr-defined]
    resolved = _referenced_paths(namespace, "frame = read(ladder.DIAGNOSTICS)")
    assert ladder.DIAGNOSTICS in resolved


def test_a_directory_counts_as_tracked_when_it_holds_tracked_files() -> None:
    """git ls-files names files, so a directory constant needs the prefix rule."""

    tracked = {ROOT / "outputs" / "dir" / "file.csv"}
    assert _is_tracked(ROOT / "outputs" / "dir", tracked)
    assert _is_tracked(ROOT / "outputs" / "dir" / "file.csv", tracked)
    assert not _is_tracked(ROOT / "outputs" / "other", tracked)


def test_a_format_template_is_not_mistaken_for_a_path() -> None:
    """f-string templates cannot be resolved statically and must not be flagged."""

    template = 'path = ROOT / f"outputs/active_presidential_nested_{version}/nested_predictions.csv"'
    matched = LITERAL.findall(template)
    assert matched, "the regex should still see the literal"
    assert all("{" in literal for literal in matched), "the match is a template"
