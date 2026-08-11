"""Verify the GitHub repository boundary and frozen active-model hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs" / "GITHUB_BASELINE_20260810.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    ]


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    for relative, expected in baseline["expected_hashes"].items():
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"required baseline artifact is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen baseline hash drift: {relative}: {actual} != {expected}"
            )

    tracked = _tracked_files()
    forbidden_prefixes = (
        ".venv-stance/",
        ".pytest_cache/",
        "archives/",
        "backups/",
        "data/cache/",
        "data/raw_lake/",
        "data/shadow/",
        "stats_competition/",
    )
    allowed_outputs = tuple(baseline["allowed_output_prefixes"])
    max_bytes = int(baseline["tracked_file_max_bytes"])

    violations: list[str] = []
    for relative in tracked:
        if relative.startswith(forbidden_prefixes):
            violations.append(f"forbidden tracked path: {relative}")
        if relative.startswith("outputs/") and not relative.startswith(allowed_outputs):
            violations.append(f"noncanonical output tracked: {relative}")
        path = ROOT / relative
        if path.is_file() and path.stat().st_size > max_bytes:
            violations.append(
                f"tracked file exceeds {max_bytes} bytes: {relative}"
            )

    if violations:
        raise RuntimeError("\n".join(violations))

    print("[GitHub baseline audit: PASS]")
    print(f"tracked_files={len(tracked)}")
    print(f"active_version={baseline['active_version']}")
    print("post_2022_outcomes_used=false")


if __name__ == "__main__":
    main()
